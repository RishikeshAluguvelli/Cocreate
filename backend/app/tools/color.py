"""Color sampling and texture-preserving recolor (wall repaint, floor tint) in CIELAB."""
from __future__ import annotations

import cv2
import numpy as np

from .composite import feather


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def hex_to_rgb01(hexstr: str) -> np.ndarray:
    s = hexstr.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=np.float32) / 255.0


def rgb01_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def sample_color(rgb: np.ndarray, mask: np.ndarray, trim: float = 0.2) -> dict:
    """Robust paint colour inside a mask: median chroma over the mid-lit pixels (drops shadows/highlights).

    Returns hex + lab + how many pixels were used.
    """
    sel = mask > 0.5
    if sel.sum() < 50:
        raise ValueError("mask too small to sample a colour")
    lab = rgb_to_lab(rgb)[sel]
    L = lab[:, 0]
    lo, hi = np.percentile(L, [trim * 100, (1 - trim) * 100])
    mid = (L >= lo) & (L <= hi)
    if mid.sum() < 50:
        mid = np.ones_like(L, dtype=bool)
    med = np.median(lab[mid], axis=0)
    rgb_med = lab_to_rgb(med.reshape(1, 1, 3).astype(np.float32))[0, 0]
    return {"hex": rgb01_to_hex(rgb_med), "lab": [float(v) for v in med], "pixels": int(sel.sum()),
            "mid_pixels": int(mid.sum())}


def recolor(rgb: np.ndarray, mask: np.ndarray, target_hex: str, luminance_follow: float = 0.7,
            feather_px: int = 6, keep_texture: float = 1.0, strength: float = 1.0, chroma_texture: float = 0.35,
            highlight_keep: float = 0.7) -> np.ndarray:
    """Repaint the masked surface to target colour while keeping shading, texture and shadows.

    - chroma (a,b) inside the mask moves to the target's chroma; `chroma_texture` keeps a fraction of the surface's own
      chroma variation around its median (wood grain, tile pattern, grass) so the result reads as a material, not a flat tint
    - luminance is shifted so the surface's median L matches the target's L (scaled by luminance_follow) while local
      variation (texture, shadows, highlights) is preserved; when a light target would clip, the bright deviations are
      compressed instead of flattening into a white blob
    - pixels far brighter than the surface (specular highlights, light falling through a window) keep less of the new
      chroma (`highlight_keep`), so highlights stay bright instead of turning into saturated patches
    - strength < 1 nudges the existing chroma toward the target instead of replacing it ("a bit bluer")
    """
    lab = rgb_to_lab(rgb)
    tgt = rgb_to_lab(hex_to_rgb01(target_hex).reshape(1, 1, 3))[0, 0]
    sel = mask > 0.5
    if sel.sum() < 10:
        return rgb
    L = lab[..., 0]
    med = np.median(lab[sel].reshape(-1, 3), axis=0)
    med_L = float(med[0])
    delta = (tgt[0] - med_L) * luminance_follow
    dev = (L - med_L) * keep_texture
    headroom = 100.0 - (med_L + delta)
    up = float(np.percentile(dev[sel], 97)) if sel.sum() > 50 else float(dev[sel].max())
    if up > headroom > 0:   # light target: compress the bright deviations so the texture survives instead of clipping
        pos = dev > 0
        dev = np.where(pos, dev * (headroom / up), dev)
    newL = np.clip(med_L + dev + delta, 0, 100)
    out = lab.copy()
    out[..., 0] = newL
    spec = np.clip((L - med_L - 25.0) / 15.0, 0, 1)                  # much brighter than the surface: speculars, glare
    keep_chroma = 1.0 - highlight_keep * spec
    tgt_a = (tgt[1] + (lab[..., 1] - med[1]) * chroma_texture) * keep_chroma
    tgt_b = (tgt[2] + (lab[..., 2] - med[2]) * chroma_texture) * keep_chroma
    out[..., 1] = lab[..., 1] * (1 - strength) + tgt_a * strength
    out[..., 2] = lab[..., 2] * (1 - strength) + tgt_b * strength
    out_rgb = lab_to_rgb(out)
    a = feather(mask, feather_px)[..., None]
    return rgb * (1 - a) + out_rgb * a
