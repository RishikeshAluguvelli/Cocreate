"""Lot / property line overlay for aerial photos. Points are normalized (0..1) polygon vertices."""
from __future__ import annotations

import cv2
import numpy as np


def _hex_bgr(hexstr: str) -> tuple[int, int, int]:
    s = hexstr.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return b, g, r


def draw_lot_lines(rgb: np.ndarray, points: list[list[float]], color: str = "#FFD60A", width_pct: float = 0.004,
                   style: str = "dashed", fill_opacity: float = 0.10, closed: bool = True, label: str | None = None,
                   corner_markers: bool = True) -> np.ndarray:
    """Draw anti-aliased property lines. Rendered at 2x then downsampled for smooth edges.

    points: [[x,y], ...] normalized to the image; >= 2 points. style: solid | dashed | dotted.
    """
    if len(points) < 2:
        raise ValueError("need at least 2 points")
    H, W = rgb.shape[:2]
    S = 2  # supersample
    canvas = cv2.resize((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (W * S, H * S), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    pts = np.array([[int(p[0] * W * S), int(p[1] * H * S)] for p in points], dtype=np.int32)
    bgr = _hex_bgr(color)
    thickness = max(2, int(max(W, H) * S * width_pct))

    # translucent fill
    if closed and fill_opacity > 0 and len(points) >= 3:
        fill = canvas.copy()
        cv2.fillPoly(fill, [pts], bgr)
        canvas = cv2.addWeighted(fill, fill_opacity, canvas, 1 - fill_opacity, 0)

    # outline with dark halo for legibility on any background
    segs = list(zip(pts, np.roll(pts, -1, axis=0))) if closed else list(zip(pts[:-1], pts[1:]))
    halo = (0, 0, 0)

    def line(a, b, col, th):
        cv2.line(canvas, tuple(a), tuple(b), col, th, lineType=cv2.LINE_AA)

    for a, b in segs:
        if style == "solid":
            line(a, b, halo, thickness + 4)
            line(a, b, bgr, thickness)
        else:
            dash = thickness * (6 if style == "dashed" else 2)
            gap = thickness * (4 if style == "dashed" else 3)
            v = b - a
            L = float(np.hypot(*v))
            if L < 1:
                continue
            u = v / L
            pos = 0.0
            while pos < L:
                p0 = a + u * pos
                p1 = a + u * min(L, pos + dash)
                line(p0.astype(int), p1.astype(int), halo, thickness + 4)
                line(p0.astype(int), p1.astype(int), bgr, thickness)
                pos += dash + gap

    if corner_markers:
        r = thickness * 2
        for p in pts:
            cv2.circle(canvas, tuple(p), r + 2, halo, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tuple(p), r, bgr, -1, lineType=cv2.LINE_AA)

    if label and len(points) >= 3:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        fs = max(0.8, max(W, H) * S / 1600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        org = (int(cx - tw / 2), int(cy + th / 2))
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_DUPLEX, fs, halo, 6, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    out = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
