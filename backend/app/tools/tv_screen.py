"""TV screen replacement: find the screen quadrilateral, perspective-warp the customer's reference onto it (or black it
out). Deterministic, pixel-exact, free - the reference is never re-imagined by a generative model."""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

import cv2
import numpy as np

from .. import store
from ..models import Photo, Shoot
from ..providers.gemini import gemini_provider
from . import composite, imaging

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
    if best is not None:
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2).astype(np.float32))
            how = "dark-panel contour"
        else:
            rect = cv2.minAreaRect(best)
            quad = _order_quad(cv2.boxPoints(rect).astype(np.float32))
            how = "dark-panel min-rect"
        # sanity: must be convex, fill most of its own bbox and not be degenerate
        qx0, qy0 = quad.min(axis=0); qx1, qy1 = quad.max(axis=0)
        bb = max(1.0, (qx1 - qx0) * (qy1 - qy0))
        if cv2.isContourConvex(quad.astype(np.int32).reshape(-1, 1, 2)) and cv2.contourArea(quad) / bb > 0.7:
            quad[:, 0] += bx0; quad[:, 1] += by0
            return quad, how
    # fallback: the detection box inset by 4% (bezel)
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
    mask_path = imaging.save_mask(hard, pdir / f"v{idx:02d}_mask.png")
    imaging.save_rgb(composite.overlay_preview(rgb, hard, color=(0.2, 0.6, 1.0)), pdir / f"v{idx:02d}_maskpreview.jpg")
    area = composite.mask_area_fraction(hard)
    label = "Blacked out the TV screen" if content is None else f"Put customer reference {reference_index + 1} on the TV screen"
    v = store.add_version(shoot, photo, out_path, "tv_screen", {"mode": mode, "reference_index": reference_index, "quad": quad.round(1).tolist(), "fit": how},
                          f"{label} (perspective fit via {how}, {area*100:.1f}% of frame)", mask_path=mask_path, cost_usd=0.001)
    return {"found": True, "photo_id": photo.id, "version_id": v.id, "version_index": v.index, "fit": how,
            "area_pct": round(area * 100, 2), "result_url": store.rel_url(out_path),
            "mask_preview_url": store.rel_url(pdir / f"v{idx:02d}_maskpreview.jpg"), "cost_usd": 0.001}
