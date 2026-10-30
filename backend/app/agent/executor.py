"""Executes planner tool calls against the store. Every function returns a JSON-serialisable dict
(plus an optional image block for inspect_photo) and emits progress events for the UI."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import io
from pathlib import Path

import numpy as np
from typing import Any, Awaitable, Callable

from .. import store
from ..config import settings
from ..models import Photo, Shoot
from ..tools import brightness, color, composite, extras, imaging, lot_lines, masks, watermark
from ..tools.qc import qc_check
from ..tools.generative import generative_edit
from ..tools.remove import remove_object
from ..tools.tv_screen import tv_screen

Emit = Callable[[str, dict], Awaitable[None]]
_SEM = asyncio.Semaphore(max(1, settings.edit_concurrency))   # EDIT_CONCURRENCY
# the UI payload of the request being served (scope, points, painted mask); set per turn in agent.loop
CURRENT_UI: contextvars.ContextVar[dict] = contextvars.ContextVar("cocreate_ui", default={})


def painted_mask_for(photo: Photo) -> "np.ndarray | None":
    """The region the user painted on this photo in the current request, as a float mask at photo size (or None)."""
    ui = CURRENT_UI.get() or {}
    m = ui.get("mask") or {}
    if not m or m.get("photo_id") != photo.id or not m.get("png"):
        return None
    cache = ui.setdefault("_painted_cache", {})
    if photo.id in cache:
        return cache[photo.id]
    try:
        from PIL import Image
        import numpy as np
        raw = base64.b64decode(str(m["png"]).split(",", 1)[-1])
        im = Image.open(io.BytesIO(raw))
        chan = im.getchannel("A") if im.mode in ("RGBA", "LA") else im.convert("L")
        chan = chan.resize((photo.width, photo.height), Image.BILINEAR)
        arr = (np.asarray(chan).astype("float32") / 255.0 > 0.5).astype("float32")
    except Exception:
        return None
    cache[photo.id] = arr
    return arr


def _photos(shoot: Shoot, ids: list[str] | str) -> list[Photo]:
    if isinstance(ids, str) and ids != "all":
        ids = [ids]
    if ids == "all" or (isinstance(ids, list) and ("all" in ids or not ids)):
        return list(shoot.photos)
    out = []
    for i in ids:
        p = shoot.photo(i)
        if p is None:
            # allow matching by name
            p = next((q for q in shoot.photos if q.name == i), None)
        if p:
            out.append(p)
    return out


def _source_path(photo: Photo, source: str | None, index: int | None) -> tuple[str, str]:
    source = source or "head"
    if source == "head":
        return photo.head.path, "current version"
    if source == "original":
        return photo.original_path, "original finished photo"
    if source == "bracket":
        if not photo.bracket_paths:
            raise ValueError("this photo has no original brackets")
        i = len(photo.bracket_paths) // 2 if index is None else index % len(photo.bracket_paths)
        return photo.bracket_paths[i], f"bracket {i + 1} of {len(photo.bracket_paths)}"
    if source == "reference":
        if not photo.reference_paths:
            raise ValueError("this photo has no customer reference images")
        i = 0 if index is None else index % len(photo.reference_paths)
        return photo.reference_paths[i], f"customer reference {i + 1}"
    raise ValueError(f"unknown source {source}")


def _version_payload(photo: Photo) -> dict:
    v = photo.head
    return {"photo_id": photo.id, "photo_name": photo.name, "version_id": v.id, "version_index": v.index,
            "summary": v.summary, "result_url": store.rel_url(v.path), "cost_usd": round(v.cost_usd, 3),
            "qc": v.params.get("qc") if isinstance(v.params, dict) else None,
            "mask_preview_url": store.rel_url(Path(v.mask_path).with_name(Path(v.mask_path).stem.replace('_mask', '_maskpreview') + '.jpg')) if v.mask_path else None}


# ---------------- tools ----------------
async def inspect_photo(shoot: Shoot, emit: Emit, photo_id: str, source: str = "head", index: int | None = None,
                        box: list[float] | None = None) -> tuple[dict, dict | None]:
    photo = shoot.photo(photo_id)
    if not photo:
        raise ValueError(f"unknown photo {photo_id}")
    path, label = _source_path(photo, source, index)
    im = imaging.load_pil(path)
    if box:
        im = imaging.crop_norm(im, box)
    im.thumbnail((settings.inspect_long_side, settings.inspect_long_side))
    b64 = imaging.pil_to_b64_jpeg(im, 88)
    text = {"photo": photo.name, "showing": label, "crop": box, "width": photo.width, "height": photo.height}
    return text, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


async def sample_color(shoot: Shoot, emit: Emit, photo_id: str, target: str, source: str = "head",
                       index: int | None = None, box: list[float] | None = None, points: list[list[float]] | None = None) -> dict:
    photo = shoot.photo(photo_id)
    if not photo:
        raise ValueError(f"unknown photo {photo_id}")
    path, label = _source_path(photo, source, index)
    rgb = imaging.load_rgb(path)
    H, W = rgb.shape[:2]
    em = await masks.text_mask(path, (W, H), target, points=points, box=box, dilate_pct=0.0, feather_pct=0.0)
    if em is None:
        return {"found": False, "message": f"could not find '{target}' in {label} of {photo.name}"}
    core = composite.erode(em.raw, max(2, int(0.004 * max(W, H))))
    if core.sum() < 50:
        core = em.raw
    s = color.sample_color(rgb, core)
    pdir = store.photo_dir(photo)
    prev = imaging.save_rgb(composite.overlay_preview(rgb, em.raw, color=(0.2, 1.0, 0.4)), pdir / f"sample_{abs(hash(target + label)) % 10**6}.jpg")
    out = {"found": True, "photo": photo.name, "sampled_from": label, "target": target, "hex": s["hex"],
           "lab": [round(v, 1) for v in s["lab"]], "pixels": s["pixels"], "mask_preview_url": store.rel_url(prev),
           "instances": em.instances}
    if em.instance_masks and len(em.instance_masks) > 1:
        cands = []
        for m in em.instance_masks:
            if (m > 0.5).sum() < 200:
                continue
            try:
                c = color.sample_color(rgb, m)
            except ValueError:
                continue
            ys, xs = np.where(m > 0.5)
            cands.append({"hex": c["hex"], "pixels": int((m > 0.5).sum()),
                          "box": [round(xs.min() / W, 2), round(ys.min() / H, 2), round(xs.max() / W, 2), round(ys.max() / H, 2)]})
        # merge near-identical colours, keep distinct ones
        distinct: list[dict] = []
        for c in sorted(cands, key=lambda c: -c["pixels"]):
            if all(sum(abs(int(c["hex"][i:i + 2], 16) - int(d["hex"][i:i + 2], 16)) for i in (1, 3, 5)) > 36 for d in distinct):
                distinct.append(c)
        if len(distinct) > 1:
            out["candidates"] = distinct[:4]
            out["note_candidates"] = ("several surfaces matched; 'hex' is the largest. If the customer meant a specific one "
                                      "(e.g. the teal accent wall), pick the candidate whose box matches or re-sample with a "
                                      "more specific target.")
    if source not in (None, "head"):
        # brackets / references are exposed differently: also give the sampled hue+saturation at the
        # finished photo's brightness, which is what "restore the original colour" usually means
        head_rgb = imaging.load_rgb(photo.head.path)
        Hh, Wh = head_rgb.shape[:2]
        em_h = await masks.text_mask(photo.head.path, (Wh, Hh), target, box=box, dilate_pct=0.0, feather_pct=0.0)
        if em_h is not None:
            core_h = composite.erode(em_h.raw, max(2, int(0.004 * max(Wh, Hh))))
            if core_h.sum() < 50:
                core_h = em_h.raw
            s_h = color.sample_color(head_rgb, core_h)
            matched = color.lab_to_rgb(np.array([[[s_h["lab"][0], s["lab"][1], s["lab"][2]]]], dtype=np.float32))[0, 0]
            out["current_hex_in_finished_photo"] = s_h["hex"]
            out["hex_exposure_matched"] = color.rgb01_to_hex(matched)
            out["note"] = ("hex is the raw colour in the source image; hex_exposure_matched keeps that hue and saturation "
                           "at the finished photo's brightness. Use hex_exposure_matched to restore a drifted colour; use hex "
                           "when the customer wants the exact source colour including its brightness.")
    return out


async def repaint(shoot: Shoot, emit: Emit, photo_ids, target: str, hex: str, luminance_follow: float = 0.7,
                  box: list[float] | None = None, strength: float = 1.0, painted_area: bool = False,
                  points: list[list[float]] | None = None) -> dict:
    results = []
    photos = _photos(shoot, photo_ids)
    if points and len(photos) != 1:
        points = None   # clicked points belong to one photo
    for photo in photos:
        async with _SEM:
            head = photo.head
            rgb = imaging.load_rgb(head.path)
            H, W = rgb.shape[:2]
            await emit("progress", {"photo_id": photo.id, "message": f"segmenting {target} in {photo.name}"})
            em = await masks.text_mask(head.path, (W, H), target, points=points, box=box, dilate_pct=0.002, feather_pct=0.004)
            um = painted_mask_for(photo) if painted_area else None
            if um is not None:
                em = masks.merge_user_mask(em, um, (W, H), dilate_pct=0.002, feather_pct=0.004)
            if em is None:
                results.append({"photo_id": photo.id, "photo_name": photo.name, "found": False})
                continue
            out = await asyncio.to_thread(color.recolor, rgb, em.raw, hex, luminance_follow=luminance_follow, feather_px=max(2, int(0.004 * max(W, H))),
                                strength=float(strength if strength is not None else 1.0))
            pdir = store.photo_dir(photo)
            idx = len(photo.versions)
            out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_repaint.jpg")
            mask_path = imaging.save_mask(em.raw, pdir / f"v{idx:02d}_mask.png")
            imaging.save_rgb(composite.overlay_preview(rgb, em.raw), pdir / f"v{idx:02d}_maskpreview.jpg")
            v = store.add_version(shoot, photo, out_path, "repaint", {"target": target, "hex": hex, "strength": strength, "luminance_follow": luminance_follow},
                                  f"Repainted {target} to {hex.upper()}{'' if (strength or 1) >= 1 else f' at {int(strength*100)}%'} ({em.area*100:.1f}% of frame)", mask_path=mask_path, cost_usd=0.01)
            # same vision QC as the generative tools: catches a mask that bled onto artwork, windows or furniture
            await emit("progress", {"photo_id": photo.id, "message": f"QC check on {photo.name}"})
            verdict = await qc_check(f"Repaint only the {target} to {hex}, keeping its texture and shading; everything that is not {target} "
                                     f"(windows, artwork, furniture, floor, ceiling) must stay identical.", head.path, str(out_path))
            v.params["qc"] = verdict
            v.cost_usd += verdict.get("cost_usd", 0); shoot.total_cost_usd += verdict.get("cost_usd", 0)
            store.save(shoot)
            r = _version_payload(photo); r["found"] = True; r["area_pct"] = round(em.area * 100, 1)
            if em.area > 0.45:
                r["warning"] = (f"the {target} mask covers {em.area*100:.0f}% of the frame; check the Mask view. If too much was repainted, "
                                f"retry with a more specific target (e.g. 'back wall') or a box, or ask the user to click points on it.")
            results.append(r)
            await emit("version", r)
    return {"results": results}


async def adjust(shoot: Shoot, emit: Emit, photo_ids, brightness_: float = 0, **kw) -> dict:
    results = []
    region = kw.pop("region", None)
    sharpen_amt = float(kw.pop("sharpen", 0) or 0)
    sliders = {k: float(kw.get(k, 0) or 0) for k in ("contrast", "saturation", "temperature", "shadows", "highlights")}
    sliders["brightness"] = float(brightness_ or 0)
    for photo in _photos(shoot, photo_ids):
        head = photo.head
        rgb = imaging.load_rgb(head.path)
        H, W = rgb.shape[:2]
        region_mask = None
        if region:
            em = await masks.text_mask(head.path, (W, H), region, dilate_pct=0.01, feather_pct=0.0)
            if em is None:
                results.append({"photo_id": photo.id, "photo_name": photo.name, "found": False, "region": region})
                continue
            region_mask = em.hard
        out = await asyncio.to_thread(brightness.adjust, rgb, region_mask=region_mask, **sliders)
        if sharpen_amt:
            out = brightness.sharpen(out, sharpen_amt)
        pdir = store.photo_dir(photo)
        idx = len(photo.versions)
        out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_adjust.jpg")
        mask_path = imaging.save_mask(region_mask, pdir / f"v{idx:02d}_mask.png") if region_mask is not None else None
        if region_mask is not None:
            imaging.save_rgb(composite.overlay_preview(rgb, region_mask), pdir / f"v{idx:02d}_maskpreview.jpg")
        changed = {k: v for k, v in {**sliders, "sharpen": sharpen_amt}.items() if v}
        desc = ", ".join(f"{k} {v:+.0f}" for k, v in changed.items()) or "no-op"
        store.add_version(shoot, photo, out_path, "adjust", {**changed, "region": region},
                          f"Adjusted {desc}" + (f" on {region}" if region else ""), mask_path=mask_path)
        r = _version_payload(photo); r["found"] = True
        results.append(r)
        await emit("version", r)
    return {"results": results}


async def draw_lot_lines(shoot: Shoot, emit: Emit, photo_id: str, points: list[list[float]], color_: str = "#FFD60A",
                         style: str = "dashed", fill_opacity: float = 0.10, label: str | None = None, closed: bool = True) -> dict:
    photo = shoot.photo(photo_id)
    if not photo:
        raise ValueError(f"unknown photo {photo_id}")
    rgb = imaging.load_rgb(photo.head.path)
    out = lot_lines.draw_lot_lines(rgb, points, color=color_ or "#FFD60A", style=style or "dashed",
                                   fill_opacity=fill_opacity if fill_opacity is not None else 0.10, label=label, closed=closed)
    pdir = store.photo_dir(photo)
    idx = len(photo.versions)
    out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_lotlines.jpg")
    store.add_version(shoot, photo, out_path, "draw_lot_lines", {"points": points, "color": color_, "style": style, "label": label},
                      f"Drew lot lines ({len(points)} corners, {style})")
    r = _version_payload(photo); r["points"] = points
    await emit("version", r)
    return r


async def add_watermark(shoot: Shoot, emit: Emit, photo_ids, text: str | None = None, use_logo: bool = False,
                        position: str = "bottom_right", opacity: float = 0.6, scale: float = 0.22, color_: str = "#FFFFFF",
                        preset: str | None = None) -> dict:
    logo = shoot.watermark_logo_path if use_logo else None
    if use_logo and not logo:
        return {"error": "no logo uploaded for this shoot; ask the user to upload one or use text"}
    if preset == "disclosure":  # MLS / California AB 723 style "digitally altered" label
        text, position, opacity, scale, logo = text or "Digitally altered", position or "bottom_left", 0.75, 0.16, None
    results = []
    for photo in _photos(shoot, photo_ids):
        rgb = imaging.load_rgb(photo.head.path)
        out = watermark.add_watermark(rgb, text=text, logo_path=logo, position=position or "bottom_right",
                                      opacity=opacity if opacity is not None else 0.6, scale=scale or 0.22, color=color_ or "#FFFFFF")
        pdir = store.photo_dir(photo)
        idx = len(photo.versions)
        out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_watermark.jpg")
        store.add_version(shoot, photo, out_path, "add_watermark", {"text": text, "logo": bool(logo), "position": position, "opacity": opacity, "scale": scale},
                          f"Watermark {'logo' if logo else repr(text)} at {position}")
        r = _version_payload(photo)
        results.append(r)
        await emit("version", r)
    return {"results": results}


async def remove_object_tool(shoot: Shoot, emit: Emit, photo_ids, target: str, method: str = "auto",
                             points=None, box=None, fill_prompt: str | None = None, painted_area: bool = False) -> dict:
    photos = _photos(shoot, photo_ids)

    async def one(photo: Photo):
        async def progress(msg: str) -> None:
            await emit("progress", {"photo_id": photo.id, "message": msg})

        async with _SEM:
            await progress(f"looking for {target} in {photo.name}")
            r = await remove_object(shoot, photo, target, method=method or "auto",
                                    points=points if len(photos) == 1 else None, box=box if len(photos) == 1 else None,
                                    user_mask=painted_mask_for(photo) if painted_area else None,
                                    fill_prompt=fill_prompt, progress=progress)
            r["photo_name"] = photo.name
            if r.get("found") and r.get("changed", True):
                await emit("version", {**_version_payload(photo), "mask_preview_url": r["mask_preview_url"]})
            else:
                await emit("skipped", {"photo_id": photo.id, "photo_name": photo.name, "reason": r.get("message")})
            return r

    results = await asyncio.gather(*(one(p) for p in photos), return_exceptions=True)
    out = []
    for p, r in zip(photos, results):
        if isinstance(r, Exception):
            out.append({"photo_id": p.id, "photo_name": p.name, "error": str(r)[:300]})
        else:
            out.append(r)
    return {"results": out}


async def generative_edit_tool(shoot: Shoot, emit: Emit, photo_ids, instruction: str, references=None, protect: str = "auto",
                               painted_area: bool = False) -> dict:
    photos = _photos(shoot, photo_ids)

    async def one(photo: Photo):
        async def progress(msg: str) -> None:
            await emit("progress", {"photo_id": photo.id, "message": msg})

        async with _SEM:
            await progress(f"editing {photo.name}: {instruction[:60]}")
            r = await generative_edit(shoot, photo, instruction, references=references, protect=protect or "auto", progress=progress,
                                      user_mask=painted_mask_for(photo) if painted_area else None)
            r["photo_name"] = photo.name
            if r.get("changed", True):
                await emit("version", {**_version_payload(photo), "mask_preview_url": r["mask_preview_url"]})
            else:
                await emit("skipped", {"photo_id": photo.id, "photo_name": photo.name, "reason": r.get("message")})
            return r

    results = await asyncio.gather(*(one(p) for p in photos), return_exceptions=True)
    out = []
    for p, r in zip(photos, results):
        out.append({"photo_id": p.id, "photo_name": p.name, "error": str(r)[:300]} if isinstance(r, Exception) else r)
    return {"results": out}


async def tv_screen_tool(shoot: Shoot, emit: Emit, photo_ids, reference_index: int = 0, mode: str = "reference",
                         brightness_: float = 0.92, box=None) -> dict:
    photos = _photos(shoot, photo_ids)

    async def one(photo: Photo):
        async def progress(msg: str) -> None:
            await emit("progress", {"photo_id": photo.id, "message": msg})

        async with _SEM:
            r = await tv_screen(shoot, photo, reference_index=reference_index or 0, mode=mode or "reference",
                                brightness=brightness_ if brightness_ is not None else 0.92, box=box if len(photos) == 1 else None, progress=progress)
            r["photo_name"] = photo.name
            if r.get("found"):
                await emit("version", {**_version_payload(photo), "mask_preview_url": r["mask_preview_url"]})
            else:
                await emit("skipped", {"photo_id": photo.id, "photo_name": photo.name, "reason": r.get("message")})
            return r

    results = await asyncio.gather(*(one(p) for p in photos), return_exceptions=True)
    return {"results": [{"photo_id": p.id, "photo_name": p.name, "error": str(r)[:300]} if isinstance(r, Exception) else r
                        for p, r in zip(photos, results)]}


async def blur(shoot: Shoot, emit: Emit, photo_ids, target: str, strength: float = 60, pixelate: bool = False,
               points=None, box=None) -> dict:
    results = []
    photos = _photos(shoot, photo_ids)
    for photo in photos:
        async with _SEM:
            head = photo.head
            rgb = imaging.load_rgb(head.path)
            H, W = rgb.shape[:2]
            await emit("progress", {"photo_id": photo.id, "message": f"finding {target} in {photo.name}"})
            em = await masks.text_mask(head.path, (W, H), target, points=points if len(photos) == 1 else None,
                                       box=box if len(photos) == 1 else None, dilate_pct=0.008, feather_pct=0.0)
            if em is None:
                results.append({"photo_id": photo.id, "photo_name": photo.name, "found": False})
                await emit("skipped", {"photo_id": photo.id, "photo_name": photo.name, "reason": f"no {target} found"})
                continue
            out = extras.blur_region(rgb, em.hard, strength=strength if strength is not None else 60, pixelate=bool(pixelate))
            pdir = store.photo_dir(photo)
            idx = len(photo.versions)
            out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_blur.jpg")
            mask_path = imaging.save_mask(em.hard, pdir / f"v{idx:02d}_mask.png")
            imaging.save_rgb(composite.overlay_preview(rgb, em.hard), pdir / f"v{idx:02d}_maskpreview.jpg")
            store.add_version(shoot, photo, out_path, "blur", {"target": target, "strength": strength, "pixelate": pixelate, "instances": em.instances},
                              f"Blurred {target} ({em.instances} region{'s' if em.instances != 1 else ''})", mask_path=mask_path, cost_usd=0.001)
            r = _version_payload(photo); r["found"] = True; r["instances"] = em.instances
            results.append(r)
            await emit("version", r)
    return {"results": results}


async def match_look(shoot: Shoot, emit: Emit, photo_ids, reference_photo_id: str, strength: float = 0.8) -> dict:
    ref = shoot.photo(reference_photo_id) or next((q for q in shoot.photos if q.name == reference_photo_id), None)
    if not ref:
        raise ValueError(f"unknown reference photo {reference_photo_id}")
    ref_rgb = imaging.load_rgb(ref.head.path)
    results = []
    for photo in _photos(shoot, photo_ids):
        if photo.id == ref.id:
            continue
        rgb = imaging.load_rgb(photo.head.path)
        out = extras.match_look(rgb, ref_rgb, strength=strength if strength is not None else 0.8)
        pdir = store.photo_dir(photo)
        idx = len(photo.versions)
        out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_match.jpg")
        store.add_version(shoot, photo, out_path, "match_look", {"reference_photo_id": ref.id, "strength": strength},
                          f"Matched colour and tone to {ref.name}")
        r = _version_payload(photo)
        results.append(r)
        await emit("version", r)
    return {"results": results}


async def crop_rotate(shoot: Shoot, emit: Emit, photo_id: str, rotate_deg: float = 0.0, box: list[float] | None = None) -> dict:
    photo = shoot.photo(photo_id)
    if not photo:
        raise ValueError(f"unknown photo {photo_id}")
    rgb = imaging.load_rgb(photo.head.path)
    out = extras.crop_rotate(rgb, rotate_deg=rotate_deg or 0.0, box=box)
    pdir = store.photo_dir(photo)
    idx = len(photo.versions)
    out_path = imaging.save_rgb(out, pdir / f"v{idx:02d}_crop.jpg")
    parts = []
    if rotate_deg:
        parts.append(f"straightened {rotate_deg:+.1f}°")
    if box:
        parts.append("cropped to " + ", ".join(f"{v:.2f}" for v in box))
    store.add_version(shoot, photo, out_path, "crop_rotate", {"rotate_deg": rotate_deg, "box": box}, "; ".join(parts).capitalize() or "No-op crop")
    r = _version_payload(photo)
    await emit("version", r)
    return r


async def undo(shoot: Shoot, emit: Emit, photo_ids, steps: int = 1) -> dict:
    results = []
    for photo in _photos(shoot, photo_ids):
        v = photo.head
        for _ in range(max(1, steps or 1)):
            if v.parent_id:
                v = photo.version(v.parent_id) or v
        photo.head_version_id = v.id
        store.save(shoot)
        r = _version_payload(photo)
        results.append(r)
        await emit("version", r)
    return {"results": results}


async def revert_to(shoot: Shoot, emit: Emit, photo_id: str, version_index: int) -> dict:
    photo = shoot.photo(photo_id)
    if not photo:
        raise ValueError(f"unknown photo {photo_id}")
    v = next((x for x in photo.versions if x.index == version_index), None)
    if not v:
        raise ValueError(f"no version {version_index}")
    photo.head_version_id = v.id
    store.save(shoot)
    r = _version_payload(photo)
    await emit("version", r)
    return r


_photo_locks: dict[str, asyncio.Lock] = {}
READ_ONLY_TOOLS = {"inspect_photo", "sample_color"}


def _photo_lock(photo_id: str) -> asyncio.Lock:
    lock = _photo_locks.get(photo_id)
    if lock is None:
        lock = _photo_locks[photo_id] = asyncio.Lock()
    return lock


@contextlib.asynccontextmanager
async def _hold_photos(shoot: Shoot, inp: dict[str, Any]):
    """The planner runs independent tool calls in parallel. Two calls that edit the SAME photo (put the reference on the
    TV, turn on the lamps) would both start from the current version and the second one saved would silently throw the
    first one's pixels away. Each editing call therefore holds a lock on every photo it touches, taken in a fixed order so
    two calls can never deadlock; calls on disjoint photos still run side by side."""
    ids = inp.get("photo_ids")
    if ids is None and inp.get("photo_id"):
        ids = [inp["photo_id"]]
    photos = sorted(_photos(shoot, ids), key=lambda p: p.id) if ids else []
    held = []
    try:
        for ph in photos:
            await _photo_lock(ph.id).acquire()
            held.append(ph.id)
        yield
    finally:
        for pid in held:
            _photo_lock(pid).release()


async def dispatch(shoot: Shoot, emit: Emit, name: str, inp: dict[str, Any]) -> tuple[dict, dict | None]:
    """Returns (json_result, optional_image_block)."""
    inp = dict(inp or {})
    if name in READ_ONLY_TOOLS:
        return await _dispatch(shoot, emit, name, inp)
    async with _hold_photos(shoot, inp):
        return await _dispatch(shoot, emit, name, inp)


async def _dispatch(shoot: Shoot, emit: Emit, name: str, inp: dict[str, Any]) -> tuple[dict, dict | None]:
    if name == "inspect_photo":
        return await inspect_photo(shoot, emit, **inp)
    if name == "sample_color":
        return await sample_color(shoot, emit, **inp), None
    if name == "repaint":
        return await repaint(shoot, emit, **inp), None
    if name == "adjust":
        b = inp.pop("brightness", 0)
        return await adjust(shoot, emit, inp.pop("photo_ids"), brightness_=b, **inp), None
    if name == "draw_lot_lines":
        c = inp.pop("color", "#FFD60A")
        return await draw_lot_lines(shoot, emit, color_=c, **inp), None
    if name == "add_watermark":
        c = inp.pop("color", "#FFFFFF")
        return await add_watermark(shoot, emit, color_=c, **inp), None
    if name == "remove_object":
        return await remove_object_tool(shoot, emit, **inp), None
    if name == "generative_edit":
        return await generative_edit_tool(shoot, emit, **inp), None
    if name == "tv_screen":
        b = inp.pop("brightness", None)
        return await tv_screen_tool(shoot, emit, brightness_=b, **inp), None
    if name == "blur":
        return await blur(shoot, emit, **inp), None
    if name == "match_look":
        return await match_look(shoot, emit, **inp), None
    if name == "crop_rotate":
        return await crop_rotate(shoot, emit, **inp), None
    if name == "undo":
        return await undo(shoot, emit, **inp), None
    if name == "revert_to":
        return await revert_to(shoot, emit, **inp), None
    raise ValueError(f"unknown tool {name}")
