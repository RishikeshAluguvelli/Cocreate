"""TV screen replacement: find the screen quadrilateral, perspective-warp the customer's reference onto it (or black it
out). Deterministic, pixel-exact, free - the reference is never re-imagined by a generative model."""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

import cv2
import numpy as np

from .. import store
from ..config import settings
from ..models import Photo, Shoot
from ..providers.gemini import gemini_provider
from . import composite, imaging
from .qc import qc_check, qc_score

Progress = Callable[[str], Awaitable[None]] | None


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_screen_quad(rgb: np.ndarray, box: list[float]) -> tuple[np.ndarray, str]:
    """Inside the detection box, find the (usually dark, glossy) screen panel as a quadrilateral.
    Returns (4x2 pixel coords, how)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    pad = 0.04
    bx0, by0 = int(max(0, (x0 - pad * (x1 - x0)) * W)), int(max(0, (y0 - pad * (y1 - y0)) * H))
    bx1, by1 = int(min(W, (x1 + pad * (x1 - x0)) * W)), int(min(H, (y1 + pad * (y1 - y0)) * H))
    crop = rgb[by0:by1, bx0:bx1]
    if crop.size == 0:
        return np.zeros((4, 2), np.float32), "empty"
    gray = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # screens are the darkest smooth region of the TV; threshold adaptively (Otsu) and take the dark side
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, int(0.01 * max(crop.shape[:2])) | 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_area = crop.shape[0] * crop.shape[1]
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.25 * box_area:
            continue
        if best is None or a > cv2.contourArea(best):
            best = c
    quad = None
    if best is not None:
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        quad, how = None, ""
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2).astype(np.float32))
            how = "dark-panel contour"
        else:
            rect = cv2.minAreaRect(best)
            (rw, rh), ang = rect[1], rect[2]
            # a screen panel photographed in a room is close to axis-aligned; a strongly tilted minimum rectangle means
            # the dark contour is not the panel (a reflection, a stand, a cabinet) and must not be used as the screen
            tilt = min(abs(ang) % 90, 90 - abs(ang) % 90)
            if tilt <= 8.0 and min(rw, rh) > 0:
                quad = _order_quad(cv2.boxPoints(rect).astype(np.float32))
                how = "dark-panel min-rect"
    if best is not None and quad is not None:
        # sanity: convex, fills most of its own bbox, is a decent part of the TV box, and has roughly the TV's aspect
        # (a dark vertical cabinet or a reflection inside the box must not become a portrait "screen")
        qx0, qy0 = quad.min(axis=0); qx1, qy1 = quad.max(axis=0)
        bb = max(1.0, (qx1 - qx0) * (qy1 - qy0))
        q_aspect = max(1.0, qx1 - qx0) / max(1.0, qy1 - qy0)
        b_aspect = max(1.0, (x1 - x0) * W) / max(1.0, (y1 - y0) * H)
        plausible = (0.6 < q_aspect / b_aspect < 1.7) and cv2.contourArea(quad) >= 0.35 * box_area and not (b_aspect > 1.2 and q_aspect < 1.0)
        if plausible and cv2.isContourConvex(quad.astype(np.int32).reshape(-1, 1, 2)) and cv2.contourArea(quad) / bb > 0.7:
            quad[:, 0] += bx0; quad[:, 1] += by0
            return quad, how
    # fallback 1: the largest dark region's bounding box, when it is a plausible landscape panel. A TV behind glass
    # shelves or inside a cabinet gives a detection box far bigger than the screen; the whole box would overflow it.
    if best is not None:
        rx, ry, rw, rh = cv2.boundingRect(best)
        r_aspect = max(1.0, rw) / max(1.0, rh)
        b_aspect = max(1.0, (x1 - x0) * W) / max(1.0, (y1 - y0) * H)
        if rw * rh >= 0.15 * box_area and r_aspect >= 1.1 and 0.5 < r_aspect / b_aspect < 2.5:
            quad = np.array([[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]], np.float32)
            quad[:, 0] += bx0; quad[:, 1] += by0
            return quad, "dark-panel bounding box"
    # fallback 2: the detection box inset by 4% (bezel)
    ix, iy = 0.04 * (x1 - x0) * W, 0.04 * (y1 - y0) * H
    quad = np.array([[x0 * W + ix, y0 * H + iy], [x1 * W - ix, y0 * H + iy], [x1 * W - ix, y1 * H - iy], [x0 * W + ix, y1 * H - iy]], np.float32)
    return quad, "detection box (inset)"


def _cover_fit(ref: np.ndarray, aspect: float) -> np.ndarray:
    """Centre-crop the reference to the screen's aspect ratio (w/h) so it fills the panel without stretching."""
    h, w = ref.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        return ref[:, x0:x0 + nw]
    nh = int(round(w / aspect)); y0 = (h - nh) // 2
    return ref[y0:y0 + nh]


def render_screen(rgb: np.ndarray, quad: np.ndarray, content: np.ndarray | None, brightness: float = 0.92) -> tuple[np.ndarray, np.ndarray]:
    """Warp `content` (or black) into the quad. Returns (edited_rgb, hard_mask)."""
    H, W = rgb.shape[:2]
    tl, tr, br, bl = quad
    sw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    sh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    sw, sh = max(8, sw), max(8, sh)
    if content is None:
        panel = np.zeros((sh, sw, 3), np.float32) + np.array([0.035, 0.035, 0.04], np.float32)
    else:
        panel = _cover_fit(content, sw / sh)
        panel = imaging.fit_to(panel, (sw, sh)) * brightness
    # subtle screen look: slight vignette toward the panel edges + faint glass highlight carried over from the original
    yy, xx = np.mgrid[0:sh, 0:sw]
    vig = 1 - 0.12 * (((xx / sw - 0.5) ** 2 + (yy / sh - 0.5) ** 2) * 2.2)
    panel = np.clip(panel * vig[..., None], 0, 1)
    src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(panel, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((H, W), np.float32)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 1.0)
    mask = composite.erode(mask, max(1, int(0.0015 * max(W, H))))  # stay inside the bezel
    # carry a little of the original specular highlight so the screen still sits in the room's light
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    hi = np.clip((gray - 0.55) / 0.45, 0, 1)[..., None] * 0.25
    out = np.clip(warped * (1 - hi) + hi, 0, 1)
    return out, mask


async def _qc_closeup(instruction: str, before: np.ndarray, after: np.ndarray, box: list[float], pdir: Path, tag: str) -> dict:
    """QC on a crop around the TV box (padded, at least ~700 px): on the full frame a misaligned screen is a few pixels
    and the judge waves it through; on the close-up it is obvious."""
    H, W = before.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * W, (y1 - y0) * H
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    side = max(700.0, 1.6 * max(bw, bh))
    cw, ch = min(W, side), min(H, side)
    ax0 = int(np.clip(cx - cw / 2, 0, W - cw)); ay0 = int(np.clip(cy - ch / 2, 0, H - ch))
    sl = (slice(ay0, int(ay0 + ch)), slice(ax0, int(ax0 + cw)))
    b_path = imaging.save_rgb(before[sl], pdir / f"_qc_{tag}_before.jpg", quality=90)
    a_path = imaging.save_rgb(after[sl], pdir / f"_qc_{tag}_after.jpg", quality=90)
    try:
        return await qc_check(instruction, str(b_path), str(a_path))
    finally:
        Path(b_path).unlink(missing_ok=True); Path(a_path).unlink(missing_ok=True)


async def tv_screen(shoot: Shoot, photo: Photo, reference_index: int = 0, mode: str = "reference", brightness: float = 0.92,
                    box: list[float] | None = None, progress: Progress = None) -> dict:
    head = photo.head
    rgb = imaging.load_rgb(head.path)
    H, W = rgb.shape[:2]
    if progress:
        await progress(f"finding the TV screen in {photo.name}")
    dets = await gemini_provider.detect(rgb, "television screen (the TV panel)", restrict_box=box, max_objects=3) if gemini_provider.enabled else []
    if box and not dets:
        dets = [{"box": box}]
    if not dets:
        return {"found": False, "photo_id": photo.id, "message": f"No TV screen detected in {photo.name}; left unchanged."}
    d = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    quad, how = find_screen_quad(rgb, d["box"])
    content = None
    if mode != "blackout":
        if not photo.reference_paths:
            return {"found": False, "photo_id": photo.id, "message": f"{photo.name} has no customer reference image to put on the TV."}
        content = imaging.load_rgb(photo.reference_paths[reference_index % len(photo.reference_paths)])
    if progress:
        await progress(f"fitting {'black panel' if content is None else 'reference'} to the screen ({how})")
    edited, hard = render_screen(rgb, quad, content, brightness=brightness)
    soft = composite.feather(hard, max(1, int(0.0012 * max(W, H))))
    final = composite.composite(rgb, edited, soft)
    pdir = store.photo_dir(photo)
    idx = len(photo.versions)
    out_path = imaging.save_rgb(final, pdir / f"v{idx:02d}_tv.jpg")
    cost = 0.001
    method = f"perspective fit via {how}"
    verdict: dict | None = None
    params: dict = {"mode": mode, "reference_index": reference_index, "quad": quad.round(1).tolist(), "fit": how}
    # QC-gated generative fallback: a TV behind glass shelves, cut by the frame edge or inside a cabinet defeats the
    # geometric fit. When the judge flags the fit (or it had to fall back to the detection box), ask the image model to
    # put the reference on the screen and keep whichever result the judge scores higher.
    if content is not None and settings.generative_fallback and settings.image_backend == "gemini":
        qc_instr = ("Show the customer's reference image on the TV screen so that it fills the screen panel exactly, inside the "
                    "bezel, with the screen's own perspective. It must not spill over the frame, stand, shelves or wall, must not be "
                    "a flat rectangle pasted at the wrong angle, and must not cover only part of the screen. Everything that is not "
                    "the screen must stay identical. This is a close-up of the TV area.")
        verdict = await _qc_closeup(qc_instr, rgb, final, d["box"], pdir, "fit")
        cost += verdict.get("cost_usd", 0)
        weak = how == "detection box (inset)"
        if verdict.get("verdict") == "needs_review" or weak:
            from .generative import generative_edit   # local import: generative imports masks, which imports nothing of ours here
            if progress:
                await progress(f"{photo.name}: screen fit flagged by QC, trying a generative fit")
            try:
                g = await generative_edit(shoot, photo, "Display the customer's reference image (image 2) on the television screen so it fills the "
                                          "screen exactly inside its bezel with the correct perspective, as if the TV were showing it. Keep the TV "
                                          "frame, stand, reflections on the bezel and everything else in the room unchanged.",
                                          references=[f"reference:{reference_index}"], protect="television screen", progress=progress, qc=False, save=False)
            except Exception as e:  # noqa: BLE001 - the fallback must never break the deterministic result
                g = {"changed": False, "message": str(e)[:200]}
            if g.get("changed"):
                g_path = imaging.save_rgb(g["final"], pdir / f"v{idx:02d}_cand_gen.jpg")
                gv = await _qc_closeup(qc_instr, rgb, g["final"], d["box"], pdir, "gen")
                cost += gv.get("cost_usd", 0) + float(g.get("cost_usd", 0))
                params["fallback"] = {"tried": True, "fit_qc": {k: verdict.get(k) for k in ("verdict", "fulfilled", "collateral", "artifacts")},
                                      "generative_qc": {k: gv.get(k) for k in ("verdict", "fulfilled", "collateral", "artifacts")}}
                if qc_score(gv) > qc_score(verdict):
                    final, hard, verdict = g["final"], g["hard"], gv
                    Path(out_path).unlink(missing_ok=True); Path(g_path).rename(out_path)
                    method = f"generative fit (Nano Banana), chosen by QC over the {how} fit"
                    params["fallback"]["chosen"] = "generative"
                else:
                    Path(g_path).unlink(missing_ok=True)
                    params["fallback"]["chosen"] = "perspective fit"
            else:
                params["fallback"] = {"tried": True, "generative_changed": False}
    mask_path = imaging.save_mask(hard, pdir / f"v{idx:02d}_mask.png")
    imaging.save_rgb(composite.overlay_preview(rgb, hard, color=(0.2, 0.6, 1.0)), pdir / f"v{idx:02d}_maskpreview.jpg")
    area = composite.mask_area_fraction(hard)
    label = "Blacked out the TV screen" if content is None else f"Put customer reference {reference_index + 1} on the TV screen"
    if verdict is not None:
        params["qc"] = verdict
    v = store.add_version(shoot, photo, out_path, "tv_screen", params, f"{label} ({method}, {area*100:.1f}% of frame)", mask_path=mask_path, cost_usd=cost)
    out = {"found": True, "photo_id": photo.id, "version_id": v.id, "version_index": v.index, "fit": how, "method": method,
           "area_pct": round(area * 100, 2), "result_url": store.rel_url(out_path),
           "mask_preview_url": store.rel_url(pdir / f"v{idx:02d}_maskpreview.jpg"), "cost_usd": round(cost, 3)}
    if verdict is not None:
        out["qc"] = {k: verdict.get(k) for k in ("verdict", "fulfilled", "collateral", "artifacts", "reason")}
    if params.get("fallback"):
        out["fallback"] = params["fallback"]
    return out
