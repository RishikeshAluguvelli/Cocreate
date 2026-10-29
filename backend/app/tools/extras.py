"""Small deterministic tools: privacy blur, look matching across photos, crop/straighten."""
from __future__ import annotations

import cv2
import numpy as np

from . import composite
from .color import lab_to_rgb, rgb_to_lab


def blur_region(rgb: np.ndarray, mask: np.ndarray, strength: float = 60.0, pixelate: bool = False) -> np.ndarray:
    """Gaussian blur (or pixelation) confined to the mask; strength 0..100 -> radius 0.5%..4% of the long side."""
    H, W = rgb.shape[:2]
    r = max(2, int(max(W, H) * (0.005 + 0.035 * np.clip(strength, 0, 100) / 100)))
    if pixelate:
        small = cv2.resize(rgb, (max(1, W // (r * 2)), max(1, H // (r * 2))), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        blurred = cv2.GaussianBlur(rgb, (0, 0), r)
    soft = composite.feather(mask, max(1, r // 2))
    return composite.composite(rgb, blurred, soft)


def match_look(rgb: np.ndarray, reference: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """Reinhard colour transfer in CIELAB: match the mean and spread of L, a, b to the reference photo so a shoot
    looks consistent. strength 0..1 blends toward the matched result."""
    src = rgb_to_lab(rgb)
    ref = rgb_to_lab(reference)
    out = src.copy()
    for ch in range(3):
        ms, ss = float(src[..., ch].mean()), float(src[..., ch].std()) + 1e-4
        mr, sr = float(ref[..., ch].mean()), float(ref[..., ch].std()) + 1e-4
        # cap the spread change so the result cannot blow out contrast
        scale = float(np.clip(sr / ss, 0.7, 1.4))
        out[..., ch] = (src[..., ch] - ms) * scale + mr
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    matched = lab_to_rgb(out)
    return np.clip(rgb * (1 - strength) + matched * strength, 0, 1)


def crop_rotate(rgb: np.ndarray, rotate_deg: float = 0.0, box: list[float] | None = None) -> np.ndarray:
    """Straighten by a small angle (auto-cropping the black wedges) and/or crop to a normalized box. The result is
    resampled back to the original frame size so versions stay comparable."""
    H, W = rgb.shape[:2]
    out = rgb
    if rotate_deg:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rotate_deg, 1.0)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        # largest axis-aligned rectangle inside the rotated frame
        a = abs(np.deg2rad(rotate_deg))
        sin, cos = np.sin(a), np.cos(a)
        if W >= H:
            side_short = H
            half = (side_short / 2) / (sin * (W / H) + cos) if sin * 2 * (W / H) > cos else None
        else:
            side_short = W
            half = None
        if half is None:
            cos2 = np.cos(2 * a)
            wr = (W * cos - H * sin) / cos2 if cos2 > 1e-6 else W * 0.8
            hr = (H * cos - W * sin) / cos2 if cos2 > 1e-6 else H * 0.8
        else:
            wr, hr = 2 * half * (W / H), 2 * half
        wr, hr = max(8, min(W, wr)), max(8, min(H, hr))
        x0, y0 = int((W - wr) / 2), int((H - hr) / 2)
        out = out[y0:y0 + int(hr), x0:x0 + int(wr)]
    if box:
        h, w = out.shape[:2]
        x0, y0, x1, y1 = box
        # force the frame's aspect ratio so nothing is stretched
        bw, bh = (x1 - x0) * w, (y1 - y0) * h
        target = W / H
        if bw / max(1, bh) > target:
            bh = bw / target
        else:
            bw = bh * target
        cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
        bw, bh = min(bw, w), min(bh, h)
        xx0, yy0 = int(np.clip(cx - bw / 2, 0, w - bw)), int(np.clip(cy - bh / 2, 0, h - bh))
        out = out[yy0:yy0 + int(bh), xx0:xx0 + int(bw)]
    return cv2.resize(np.ascontiguousarray(out), (W, H), interpolation=cv2.INTER_LANCZOS4)
