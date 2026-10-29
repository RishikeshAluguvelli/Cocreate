"""FastAPI app: shoots, photos, chat (SSE), manual adjustments, exports, static frontend."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import zipfile

import numpy as np
from PIL import Image
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import auth, demo_shoots, store
from .agent.loop import archive_chat, run_turn
from .config import APP_DIR, settings
from .models import Photo, Shoot
from .report import build_report_html
from .agent.executor import _source_path
from .tools import brightness, composite, imaging, pick

app = FastAPI(title="Cocreate · conversational real-estate photo editing")
# Same-origin by default (the SPA is served by this app; the Vite dev server proxies /api). Only an explicitly configured
# public origin is allowed cross-origin, never a wildcard, so the session cookie cannot be used from another site.
if settings.public_base_url:
    app.add_middleware(CORSMiddleware, allow_origins=[settings.public_base_url.rstrip("/")], allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_FILES = 60
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_CHAT_CHARS = 4000
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin"}


@app.middleware("http")
async def private_cache(request: Request, call_next):
    """Everything under /api and /files belongs to one user: mark it private so a CDN in front of the app (Render,
    Cloudflare) never serves one tenant's photo or JSON to the next visitor from its cache. Browsers may still keep
    versioned images for an hour; the built frontend assets under /assets stay publicly cacheable."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/files/"):
        response.headers["Cache-Control"] = "private, max-age=3600"
    elif path.startswith("/api/") and "Cache-Control" not in response.headers:
        response.headers["Cache-Control"] = "private, no-store"
    if path.startswith(("/files/", "/api/")):
        response.headers["Vary"] = "Cookie"
    return response


def _checked_uploads(files: list[UploadFile]) -> None:
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(400, f"At most {MAX_UPLOAD_FILES} photos per upload.")
    for f in files:
        if f.content_type and not f.content_type.startswith("image/"):
            raise HTTPException(400, f"{f.filename or 'file'} is not an image.")


def _checked_bytes(name: str, data: bytes) -> bytes:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"{name} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return data

COOKIE = "cc_session"


# ---------------- accounts / tenant isolation ----------------
def _is_https(request: Request) -> bool:
    return settings.cookie_secure or request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "") == "https"


def _set_session(response: Response, request: Request, token: str) -> None:
    response.set_cookie(COOKIE, token, max_age=auth.SESSION_TTL, httponly=True, samesite="lax", secure=_is_https(request), path="/")


def _user_from_request(request: Request) -> dict | None:
    return auth.session_user(request.cookies.get(COOKIE))


def current_user(request: Request) -> dict | None:
    """The logged-in account, if any (open mode never requires one)."""
    return _user_from_request(request)


def require_user(request: Request) -> dict | None:
    """In guest and required modes every shoot call needs a session (a guest one is enough); open mode passes it along."""
    user = _user_from_request(request)
    if settings.auth_mode in ("guest", "required") and user is None:
        raise HTTPException(401, "Please log in.")
    return user


def _owns(s: Shoot, user: dict | None) -> bool:
    if settings.auth_mode == "open":
        return True
    return user is not None and s.owner_id == user["id"]


def _claim_orphans(user: dict) -> int:
    """The first real account inherits shoots made before accounts existed (single-user -> multi-user migration)."""
    n = 0
    for sh in store.list_shoots():
        if sh.owner_id is None and not sh.name.startswith("eval ") and sh.name != "stream test":
            sh.owner_id = user["id"]
            store.save(sh)
            n += 1
    return n


def _user_json(u: dict | None) -> dict:
    return {"auth_mode": settings.auth_mode,
            "user": None if u is None else {"id": u["id"], "email": u["email"], "is_guest": u["is_guest"]}}


class Credentials(BaseModel):
    email: str
    password: str


@app.get("/api/auth/me")
def auth_me(request: Request, response: Response, user: dict | None = Depends(current_user)):
    """In guest mode a first visit silently gets its own private guest workspace."""
    if user is None and settings.auth_mode == "guest":
        user = auth.create_guest()
        _set_session(response, request, auth.create_session(user["id"], request.headers.get("user-agent", "")))
    return _user_json(user)


@app.post("/api/auth/register")
def auth_register(body: Credentials, request: Request, response: Response, user: dict | None = Depends(current_user)):
    try:
        u = auth.register(body.email, body.password, upgrade_user_id=user["id"] if user and user["is_guest"] else None)
    except auth.AuthError as e:
        raise HTTPException(400, str(e))
    auth.delete_session(request.cookies.get(COOKIE))
    _set_session(response, request, auth.create_session(u["id"], request.headers.get("user-agent", "")))
    out = _user_json(u)
    if auth.user_count() == 1:
        out["claimed_shoots"] = _claim_orphans(u)
    return out


@app.post("/api/auth/login")
def auth_login(body: Credentials, request: Request, response: Response):
    try:
        u = auth.authenticate(body.email, body.password, ip=request.client.host if request.client else "")
    except auth.AuthError as e:
        raise HTTPException(401, str(e))
    _set_session(response, request, auth.create_session(u["id"], request.headers.get("user-agent", "")))
    return _user_json(u)


@app.post("/api/auth/guest")
def auth_guest(request: Request, response: Response):
    """One-click isolated workspace; can be turned into a full account later with /register."""
    u = auth.create_guest()
    _set_session(response, request, auth.create_session(u["id"], request.headers.get("user-agent", "")))
    return _user_json(u)


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    auth.delete_session(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/change-password")
def auth_change_password(body: PasswordChange, request: Request, response: Response, user: dict | None = Depends(current_user)):
    if user is None:
        raise HTTPException(401, "Please log in.")
    try:
        auth.change_password(user["id"], body.current_password, body.new_password)
    except auth.AuthError as e:
        raise HTTPException(400, str(e))
    _set_session(response, request, auth.create_session(user["id"], request.headers.get("user-agent", "")))  # others are logged out
    return {"ok": True}


class ForgotIn(BaseModel):
    email: str


@app.post("/api/auth/forgot")
def auth_forgot(body: ForgotIn, request: Request):
    """Always answers the same way so the endpoint cannot be used to find out which emails have accounts."""
    try:
        token, u = auth.create_reset_token(body.email)
    except auth.AuthError as e:
        raise HTTPException(429, str(e))
    out: dict = {"ok": True, "sent": False}
    if token and u:
        base = settings.public_base_url.rstrip("/") or str(request.base_url).rstrip("/")
        link = f"{base}/#reset={token}"
        try:
            out["sent"] = auth.send_reset_email(u["email"], link)
        except Exception as e:  # noqa: BLE001
            print(f"[auth] reset mail failed: {e}")
        if not out["sent"]:
            print(f"[auth] password reset link for {u['email']}: {link}")
            if settings.auth_show_reset_link:
                out["reset_link"] = link
    return out


class ResetIn(BaseModel):
    token: str
    new_password: str


@app.post("/api/auth/reset")
def auth_reset(body: ResetIn, request: Request, response: Response):
    try:
        u = auth.reset_password(body.token, body.new_password)
    except auth.AuthError as e:
        raise HTTPException(400, str(e))
    _set_session(response, request, auth.create_session(u["id"], request.headers.get("user-agent", "")))
    return _user_json(u)


@app.get("/files/{path:path}")
def files(path: str, request: Request):
    """Photos, versions and masks. In required mode a shoot's files are only served to its owner."""
    base = settings.data_dir.resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        raise HTTPException(404)
    parts = Path(path).parts
    if settings.auth_mode != "open" and parts and parts[0] == "shoots":
        s = store.get(parts[1]) if len(parts) > 1 else None
        if s is None or not _owns(s, _user_from_request(request)):
            raise HTTPException(404)
    return FileResponse(target, headers=SECURITY_HEADERS)


def _photo_json(p: Photo) -> dict:
    return {
        "id": p.id, "name": p.name, "width": p.width, "height": p.height,
        "original_url": store.rel_url(p.original_path),
        "head_version_id": p.head_version_id,
        "brackets": [store.rel_url(b) for b in p.bracket_paths],
        "references": [store.rel_url(r) for r in p.reference_paths],
        "dataset_request": p.dataset_request,
        "versions": [{
            "id": v.id, "index": v.index, "parent_id": v.parent_id, "tool": v.tool, "summary": v.summary,
            "params": v.params, "url": store.rel_url(v.path), "created_at": v.created_at, "cost_usd": v.cost_usd,
            "mask_preview_url": store.rel_url(Path(v.mask_path).with_name(Path(v.mask_path).stem.replace("_mask", "_maskpreview") + ".jpg"))
            if v.mask_path and Path(v.mask_path).with_name(Path(v.mask_path).stem.replace("_mask", "_maskpreview") + ".jpg").exists() else None,
        } for v in p.versions],
    }


def _shoot_json(s: Shoot) -> dict:
    return {"id": s.id, "name": s.name, "created_at": s.created_at, "total_cost_usd": round(s.total_cost_usd, 3),
            "has_logo": bool(s.watermark_logo_path), "photos": [_photo_json(p) for p in s.photos],
            "chat": [m.model_dump() for m in s.chat], "archived_chats": s.archived_chats,
            "memory_notes": s.memory_notes, "logo_url": store.rel_url(s.watermark_logo_path) if s.watermark_logo_path else None}


def _get(shoot_id: str, user: dict | None = None) -> Shoot:
    s = store.get(shoot_id)
    if not s or not _owns(s, user):
        raise HTTPException(404, "shoot not found")
    return s


@app.get("/api/health")
def health():
    mock = os.environ.get("COCREATE_MOCK", "") not in ("", "0", "false")
    return {"ok": True, "planner": settings.planner != "mock", "planner_provider": settings.planner,
            "planner_model": settings.planner_model_name, "image_models": settings.image_backend != "mock",
            "image_backend": settings.image_backend, "mock": mock,
            "edit_model": {"fal": settings.fal_edit_endpoint, "gemini": settings.gemini_edit_model}.get(settings.image_backend, "mock"),
            "auth_mode": settings.auth_mode}


@app.get("/api/demo-shoots")
def demo_templates():
    return demo_shoots.list_templates()


@app.post("/api/shoots/demo/{template_id}")
async def create_demo(template_id: str, user: dict | None = Depends(require_user)):
    try:
        s = await demo_shoots.create_demo_shoot(template_id, owner_id=user["id"] if user else None)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return _shoot_json(s)


@app.post("/api/shoots")
async def create_shoot(name: str = Form("Uploaded shoot"), files: list[UploadFile] = File(...),
                       references: list[UploadFile] | None = File(None), user: dict | None = Depends(require_user)):
    _checked_uploads(files)
    data = [(f.filename or f"photo_{i}.jpg", _checked_bytes(f.filename or "photo", await f.read())) for i, f in enumerate(files)]
    refs = [_checked_bytes(r.filename or "reference", await r.read()) for r in (references or [])]
    s = demo_shoots.create_shoot_from_files(name, data, references=refs, owner_id=user["id"] if user else None)
    return _shoot_json(s)


@app.get("/api/shoots")
def list_shoots(user: dict | None = Depends(require_user)):
    out = []
    for s in store.list_shoots():
        if s.name.startswith("eval ") or s.name == "stream test":
            continue  # harness shoots are not user work
        if not _owns(s, user):
            continue
        edits = sum(max(0, len(p.versions) - 1) for p in s.photos)
        last = max([s.created_at, *(v.created_at for p in s.photos for v in p.versions), *(m.created_at for m in s.chat)])
        out.append({"id": s.id, "name": s.name, "created_at": s.created_at, "updated_at": last, "photos": len(s.photos),
                    "edits": edits, "messages": len(s.chat), "archived_chats": len(s.archived_chats),
                    "cover": store.rel_url(s.photos[0].head.path) if s.photos else None})
    return sorted(out, key=lambda x: -x["updated_at"])


@app.get("/api/shoots/{shoot_id}")
def get_shoot(shoot_id: str, user: dict | None = Depends(require_user)):
    return _shoot_json(_get(shoot_id, user))


@app.delete("/api/shoots/{shoot_id}")
def delete_shoot(shoot_id: str, user: dict | None = Depends(require_user)):
    _get(shoot_id, user)
    store.delete(shoot_id)
    return {"ok": True}


@app.post("/api/shoots/{shoot_id}/logo")
async def upload_logo(shoot_id: str, user: dict | None = Depends(require_user), file: UploadFile = File(...)):
    s = _get(shoot_id, user)
    if Path(file.filename or "logo.png").suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "Logo must be PNG, JPEG or WebP.")
    p = store.shoot_dir(s.id) / ("logo" + Path(file.filename or "logo.png").suffix.lower())
    p.write_bytes(_checked_bytes("logo", await file.read()))
    s.watermark_logo_path = str(p)
    store.save(s)
    return {"ok": True, "logo_url": store.rel_url(p)}


class ChatIn(BaseModel):
    text: str
    selected_photo_id: str | None = None
    points: list[list[float]] | None = None
    include_dataset_requests: bool = False
    scope: str = "shoot"  # "shoot" = agent decides per photo; "photo" = only the selected photo
    confirm_first: bool = False  # describe the plan and wait for approval before running edit tools
    mask: dict | None = None  # {"photo_id", "png" (base64 PNG, white = painted), "area"}: a region the user painted on one photo
    picked_color: dict | None = None  # {"hex", "from"}: the colour the user picked with the eyedropper, if any


@app.post("/api/shoots/{shoot_id}/chat/new")
def new_chat(shoot_id: str, user: dict | None = Depends(require_user)):
    """Archive the current conversation and start a fresh one (photo versions are kept)."""
    s = _get(shoot_id, user)
    archive_chat(s)
    return _shoot_json(s)


@app.post("/api/shoots/{shoot_id}/photos")
async def add_photos(shoot_id: str, user: dict | None = Depends(require_user), files: list[UploadFile] = File(...)):
    """Add more finished photos to an existing shoot."""
    s = _get(shoot_id, user)
    _checked_uploads(files)
    data = [(f.filename or f"photo_{i}.jpg", _checked_bytes(f.filename or "photo", await f.read())) for i, f in enumerate(files)]
    demo_shoots.add_photos_to_shoot(s, data)
    return _shoot_json(s)


@app.post("/api/shoots/{shoot_id}/chat")
async def chat(shoot_id: str, body: ChatIn, user: dict | None = Depends(require_user)):
    s = _get(shoot_id, user)
    if not body.text.strip():
        raise HTTPException(400, "Empty request.")
    if len(body.text) > MAX_CHAT_CHARS:
        raise HTTPException(400, f"Please keep a request under {MAX_CHAT_CHARS} characters.")
    if body.mask and len(str(body.mask.get("png", ""))) > 3_000_000:
        raise HTTPException(400, "The painted mask is too large.")

    async def gen():
        async for ev in run_turn(s, body.text, body.model_dump()):
            yield {"event": ev["type"], "data": json.dumps(ev["data"], ensure_ascii=False)}

    return EventSourceResponse(gen(), ping=15)


class RevertIn(BaseModel):
    version_id: str


@app.post("/api/shoots/{shoot_id}/photos/{photo_id}/revert")
def revert(shoot_id: str, photo_id: str, body: RevertIn, user: dict | None = Depends(require_user)):
    s = _get(shoot_id, user)
    p = s.photo(photo_id)
    if not p or not p.version(body.version_id):
        raise HTTPException(404, "version not found")
    p.head_version_id = body.version_id
    store.save(s)
    return _photo_json(p)


class ManualAdjustIn(BaseModel):
    brightness: float = 0
    contrast: float = 0
    saturation: float = 0
    temperature: float = 0
    shadows: float = 0
    highlights: float = 0
    sharpen: float = 0
    # The version the slider values are relative to. The UI passes the pre-adjustment image when the head is an
    # adjustment, so dragging a slider re-renders from the base instead of stacking on top of the saved result.
    base_version_id: str | None = None


@app.post("/api/shoots/{shoot_id}/photos/{photo_id}/adjust")
def manual_adjust(shoot_id: str, photo_id: str, body: ManualAdjustIn, user: dict | None = Depends(require_user)):
    """Manual sliders (customers can also adjust by hand)."""
    s = _get(shoot_id, user)
    p = s.photo(photo_id)
    if not p:
        raise HTTPException(404, "photo not found")
    base = p.version(body.base_version_id) if body.base_version_id else p.head
    if base is None:
        raise HTTPException(404, "base version not found")
    vals = body.model_dump(exclude={"base_version_id"})
    sharpen = vals.pop("sharpen", 0)
    changed = {k: v for k, v in {**vals, "sharpen": sharpen}.items() if v}
    if not changed:
        # every slider back at zero: the result is the base image itself, so just make it the head again
        p.head_version_id = base.id
        store.save(s)
        return _photo_json(p)
    rgb = imaging.load_rgb(base.path)
    out = brightness.adjust(rgb, **vals)
    if sharpen:
        out = brightness.sharpen(out, sharpen)
    idx = len(p.versions)
    path = imaging.save_rgb(out, store.photo_dir(p) / f"v{idx:02d}_manual.jpg")
    store.add_version(s, p, path, "manual_adjust", changed,
                      "Sliders: " + ", ".join(f"{k} {v:+.0f}" for k, v in changed.items()), parent=base)
    return _photo_json(p)


# ---------------- eyedropper + one-click recolour (deterministic, no planner) ----------------
class PointIn(BaseModel):
    x: float
    y: float
    source: str = "head"          # head | original | bracket | reference : which image the click refers to
    index: int | None = None      # bracket / reference index
    tolerance: float = 50         # 5..100, how far the colour band reaches


class RegionIn(PointIn):
    mode: str = "auto"            # auto | surface | object
    points: list[list[float]] | None = None   # several clicks on the same surface: their regions are unioned (x, y is the last one)
    follow_light: bool = False    # extend into parts of the surface that fade into strong light (far end of a floor)
    # hand corrections drawn in the editor, each a PNG (base64 or data URL) at any resolution where opaque / white =
    # painted: "add" is unioned with the detected region, "erase" is cut out of it
    mask_edit: dict | None = None


class ApplyColorIn(RegionIn):
    hex: str
    strength: float = 1.0         # 0..1 hue/saturation move toward the colour
    luminance_follow: float = 0.7  # 0..1 brightness move toward the colour's brightness


def _hex_ok(h: str) -> str:
    h = h.strip()
    if not (len(h) == 7 and h[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in h[1:])):
        raise HTTPException(400, "Colour must be a hex value like #3A5F6B.")
    return h.upper()


def _point_photo(shoot_id: str, photo_id: str, body: PointIn, user: dict | None) -> tuple[Shoot, Photo, str, str]:
    s = _get(shoot_id, user)
    p = s.photo(photo_id)
    if not p:
        raise HTTPException(404, "photo not found")
    if not (0 <= body.x <= 1 and 0 <= body.y <= 1):
        raise HTTPException(400, "Point must be inside the photo.")
    try:
        path, label = _source_path(p, body.source, body.index)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return s, p, path, label


@app.post("/api/shoots/{shoot_id}/photos/{photo_id}/color-at")
async def color_at(shoot_id: str, photo_id: str, body: PointIn, user: dict | None = Depends(require_user)):
    """Eyedropper: the paint colour of the wall or object under the click (median of the grown region, not one pixel)."""
    s, p, path, label = _point_photo(shoot_id, photo_id, body, user)
    rgb = await asyncio.to_thread(imaging.load_rgb, path)
    r = await asyncio.to_thread(pick.pick_color, rgb, body.x, body.y, body.tolerance)
    region = r.pop("region")
    prev = await asyncio.to_thread(pick.overlay_png, region, store.photo_dir(p) / "pick_color.png", 1024, (52, 211, 153))
    r.update({"photo": p.name, "sampled_from": label, "source": body.source, "index": body.index,
              "preview_url": f"{store.rel_url(prev)}?t={int(time.time() * 1000)}"})
    return r


MAX_MASK_PNG_CHARS = 3_000_000


def _decode_mask_png(data, size_wh: tuple[int, int], what: str) -> np.ndarray | None:
    """A PNG drawn in the browser (alpha channel if present, else luminance) -> float mask at the photo's size."""
    if not data:
        return None
    if not isinstance(data, str) or len(data) > MAX_MASK_PNG_CHARS:
        raise HTTPException(400, f"The {what} drawing is too large.")
    try:
        raw = base64.b64decode(data.split(",", 1)[-1], validate=False)
        im = Image.open(io.BytesIO(raw))
        chan = im.getchannel("A") if im.mode in ("RGBA", "LA") else im.convert("L")
        chan = chan.resize(size_wh, Image.BILINEAR)
        return (np.asarray(chan).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    except Exception:
        raise HTTPException(400, f"The {what} drawing could not be read.")


def _region_union(rgb, body: RegionIn):
    pts = [[float(x), float(y)] for x, y in (body.points or [[body.x, body.y]]) if 0 <= x <= 1 and 0 <= y <= 1][:12] or [[body.x, body.y]]
    region, used = None, []
    for x, y in pts:
        m, u = pick.region_for_apply(rgb, x, y, body.tolerance, body.mode, body.follow_light)
        region = m if region is None else np.maximum(region, m)
        used.append(u)
    edit = body.mask_edit or {}
    if edit:
        H, W = region.shape
        add = _decode_mask_png(edit.get("add"), (W, H), "added-area")
        erase = _decode_mask_png(edit.get("erase"), (W, H), "erased-area")
        if add is not None and add.any():
            region = np.maximum(region, add)
            used.append("hand-added")
        if erase is not None and erase.any():
            region = region * (1.0 - erase)
            used.append("hand-erased")
    return region, " + ".join(dict.fromkeys(used)), pts


@app.post("/api/shoots/{shoot_id}/photos/{photo_id}/region-at")
async def region_at(shoot_id: str, photo_id: str, body: RegionIn, user: dict | None = Depends(require_user)):
    """Preview of the surface / object that a click would recolour (same algorithm as apply-color)."""
    s, p, path, label = _point_photo(shoot_id, photo_id, body, user)
    rgb = await asyncio.to_thread(imaging.load_rgb, path)
    region, used, _ = await asyncio.to_thread(_region_union, rgb, body)
    prev = await asyncio.to_thread(pick.overlay_png, region, store.photo_dir(p) / "pick_region.png")
    bb = composite.mask_bbox(region)
    H, W = region.shape
    return {"area_pct": round(float((region > 0.5).mean()) * 100, 2), "mode_used": used,
            "box": [round(bb[0] / W, 3), round(bb[1] / H, 3), round(bb[2] / W, 3), round(bb[3] / H, 3)] if bb else None,
            "preview_url": f"{store.rel_url(prev)}?t={int(time.time() * 1000)}"}


@app.post("/api/shoots/{shoot_id}/photos/{photo_id}/apply-color")
async def apply_color(shoot_id: str, photo_id: str, body: ApplyColorIn, user: dict | None = Depends(require_user)):
    """Recolour the surface / object under the click on the current version; keeps texture and shading; new version."""
    hex_ = _hex_ok(body.hex)
    s, p, _, _ = _point_photo(shoot_id, photo_id, body, user)
    head = p.head
    rgb = await asyncio.to_thread(imaging.load_rgb, head.path)
    region, used, pts = await asyncio.to_thread(_region_union, rgb, body)
    area = float((region > 0.5).mean())
    if area < 0.0005:
        raise HTTPException(400, "Nothing left to recolour under that point: the area is empty (or you erased all of it). Click a surface, lower the erase, or widen the band.")
    out = await asyncio.to_thread(pick.apply_color, rgb, region, hex_, body.strength, body.luminance_follow)
    pdir = store.photo_dir(p)
    idx = len(p.versions)
    out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_recolor.jpg")
    mask_path = imaging.save_mask(region, pdir / f"v{idx:02d}_mask.png")
    imaging.save_rgb(composite.overlay_preview(rgb, region), pdir / f"v{idx:02d}_maskpreview.jpg")
    store.add_version(s, p, out_path, "recolor",
                      {"hex": hex_, "points": [[round(x, 4), round(y, 4)] for x, y in pts], "mode": used, "tolerance": body.tolerance, "follow_light": body.follow_light,
                       "hand_edited": bool(body.mask_edit and (body.mask_edit.get("add") or body.mask_edit.get("erase"))),
                       "strength": body.strength, "luminance_follow": body.luminance_follow, "area_pct": round(area * 100, 2)},
                      f"Recoloured the clicked {used} to {hex_}{'' if body.strength >= 1 else f' at {int(body.strength * 100)}%'} ({area * 100:.1f}% of frame)",
                      mask_path=mask_path)
    return _photo_json(p)


@app.get("/api/shoots/{shoot_id}/photos/{photo_id}/download")
def download_photo(shoot_id: str, photo_id: str, user: dict | None = Depends(require_user)):
    s = _get(shoot_id, user)
    p = s.photo(photo_id)
    if not p:
        raise HTTPException(404)
    return FileResponse(p.head.path, media_type="image/jpeg", filename=f"{p.name}.jpg")


@app.get("/api/shoots/{shoot_id}/report.html")
def shoot_report(shoot_id: str, user: dict | None = Depends(require_user)):
    """One self-contained HTML page: requests, every edit with its QC verdict, before/after per photo."""
    s = _get(shoot_id, user)
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in s.name)[:40].strip() or "shoot"
    return HTMLResponse(build_report_html(s), headers={**SECURITY_HEADERS, "Content-Disposition": f'inline; filename="{safe}-report.html"'})


@app.get("/api/shoots/{shoot_id}/download.zip")
def download_zip(shoot_id: str, user: dict | None = Depends(require_user)):
    s = _get(shoot_id, user)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for p in s.photos:
            z.write(p.head.path, f"{p.name}.jpg")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{s.name[:40]}.zip"'})


# ---- static frontend (built by `npm run build`, copied to app/static) ----
STATIC = APP_DIR / "static"
if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC / "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = STATIC / path
        if path and f.is_file():
            return FileResponse(f, headers=SECURITY_HEADERS)
        return FileResponse(STATIC / "index.html", headers=SECURITY_HEADERS)
