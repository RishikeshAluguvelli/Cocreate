"""Mask utilities + the one rule that keeps 'unrelated parts intact':
every generative result is pasted back into the ORIGINAL pixels through a feathered mask."""
from __future__ import annotations

import cv2
import numpy as np


def _odd(n: int) -> int:
    n = int(max(1, n))
    return n if n % 2 == 1 else n + 1


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * radius_px + 1), _odd(2 * radius_px + 1)))
    return cv2.dilate((mask > 0.5).astype(np.uint8), k).astype(np.float32)


def erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * radius_px + 1), _odd(2 * radius_px + 1)))
    return cv2.erode((mask > 0.5).astype(np.uint8), k).astype(np.float32)


def feather(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.clip(mask, 0, 1).astype(np.float32)
    k = _odd(2 * radius_px + 1)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0), 0, 1)


def prepare_edit_mask(mask: np.ndarray, long_side: int, dilate_pct: float, feather_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (hard_mask_for_model, soft_mask_for_composite)."""
    hard = dilate(mask, int(round(long_side * dilate_pct)))
    soft = feather(hard, int(round(long_side * feather_pct)))
    return hard, soft


def composite(original: np.ndarray, edited: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """original/edited: HxWx3 float, soft_mask: HxW float in [0,1]. Inside mask -> edited, outside -> original."""
    a = soft_mask[..., None]
    return original * (1.0 - a) + edited * a


def mask_area_fraction(mask: np.ndarray) -> float:
    return float((mask > 0.5).mean())


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def change_mask(a: np.ndarray, b: np.ndarray, thresh: float = 0.06, min_blob_px: int = 64) -> np.ndarray:
    """Where did a generative model actually change pixels? Perceptual-ish diff, cleaned up."""
    diff = np.abs(a - b).max(axis=2)
    m = (cv2.GaussianBlur(diff, (7, 7), 0) > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
            keep[labels == i] = 1
    return keep.astype(np.float32)


def overlay_preview(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.2, 0.2), alpha: float = 0.45) -> np.ndarray:
    """Tint the masked region for a visual check."""
    tint = np.array(color, dtype=np.float32)[None, None, :]
    a = (mask > 0.5).astype(np.float32)[..., None] * alpha
    out = rgb * (1 - a) + tint * a
    # outline
    edges = cv2.Canny((mask > 0.5).astype(np.uint8) * 255, 50, 150) > 0
    out[edges] = tint[0, 0]
    return out


def diff_overlay(before: np.ndarray, after: np.ndarray, thresh: float = 0.06) -> np.ndarray:
    """`after` with every changed pixel tinted magenta - lets a reviewer (human or model) see exactly where an edit
    touched the photo."""
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
    m = change_mask(before, after, thresh=thresh, min_blob_px=16)
    return overlay_preview(after, m, color=(1.0, 0.15, 0.8), alpha=0.5)
