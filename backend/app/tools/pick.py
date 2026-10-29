"""Eyedropper and one-click recolour, driven from the editor without the planner.

`pick_color` reads the paint colour of the wall or object under a click (region grown from the click, median of its
mid-lit pixels, so shading and highlights do not bias it). `apply_color` recolours the surface or object under a click
with `color.recolor`, keeping texture and shading. Both are deterministic OpenCV work: no model calls, sub-second.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import color, composite, masks


def _hex_at(lab_med: np.ndarray) -> str:
    return color.rgb01_to_hex(color.lab_to_rgb(np.asarray(lab_med, dtype=np.float32).reshape(1, 1, 3))[0, 0])


def pick_color(rgb: np.ndarray, x: float, y: float, tolerance: float = 50.0) -> dict:
    """Colour of the surface under (x, y). Returns hex (surface median), hex_at_click (the exact spot), lighter and
    darker variants (the same paint seen in light and in shadow), the grown region and its size."""
    H, W = rgb.shape[:2]
    region, used = masks.region_at_point(rgb, x, y, tolerance, mode="surface")
    if (region > 0.5).sum() < 200:
        region = np.zeros((H, W), np.float32)
        cv2.circle(region, (int(x * W), int(y * H)), max(4, int(0.01 * max(W, H))), 1.0, -1)
        used = "point"
    core = composite.erode(region, max(2, int(0.004 * max(W, H))))
    if core.sum() < 50:
        core = region
    s = color.sample_color(rgb, core)
    lab = color.rgb_to_lab(rgb)
    r = max(2, int(0.004 * max(W, H)))
    px, py = int(np.clip(x * W, 0, W - 1)), int(np.clip(y * H, 0, H - 1))
    spot = np.median(lab[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1].reshape(-1, 3), axis=0)
    Ls = lab[..., 0][region > 0.5]
    lo, hi = (np.percentile(Ls, [20, 80]) if len(Ls) > 50 else (s["lab"][0], s["lab"][0]))
    bb = composite.mask_bbox(region)
    return {"hex": s["hex"], "hex_at_click": _hex_at(spot), "lab": [round(float(v), 1) for v in s["lab"]],
            "hex_light": _hex_at([hi, s["lab"][1], s["lab"][2]]), "hex_shade": _hex_at([lo, s["lab"][1], s["lab"][2]]),
            "pixels": s["pixels"], "area_pct": round(float((region > 0.5).mean()) * 100, 2), "region_mode": used,
            "box": [round(bb[0] / W, 3), round(bb[1] / H, 3), round(bb[2] / W, 3), round(bb[3] / H, 3)] if bb else None,
            "region": region}


def region_for_apply(rgb: np.ndarray, x: float, y: float, tolerance: float = 50.0, mode: str = "auto", follow_light: bool = False) -> tuple[np.ndarray, str]:
    region, used = masks.region_at_point(rgb, x, y, tolerance, mode)
    if follow_light and used == "surface":
        region = masks.follow_lighting(rgb, region, tolerance)
        used = "surface, following the lighting"
    return region, used


def apply_color(rgb: np.ndarray, region: np.ndarray, hex_: str, strength: float = 1.0, luminance_follow: float = 0.7) -> np.ndarray:
    H, W = rgb.shape[:2]
    return color.recolor(rgb, region, hex_, luminance_follow=float(luminance_follow), feather_px=max(2, int(0.004 * max(W, H))),
                         strength=float(strength))


def overlay_png(mask: np.ndarray, path: str | Path, long_side: int = 1024, rgb_color=(251, 191, 36), fill_alpha: int = 105) -> Path:
    """Transparent PNG the browser lays over the photo: tinted fill plus a crisp outline of the region."""
    H, W = mask.shape
    scale = min(1.0, long_side / max(W, H))
    m = (mask > 0.5).astype(np.uint8)
    if scale < 1.0:
        m = cv2.resize(m, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_NEAREST)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out = np.zeros((*m.shape, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = rgb_color
    out[..., 3] = m * fill_alpha
    out[edge, 3] = 235
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(path, "PNG", optimize=True)
    return path
