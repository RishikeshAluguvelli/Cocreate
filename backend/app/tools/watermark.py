"""Watermark overlay (text or logo), applied identically across a shoot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
POSITIONS = ("bottom_right", "bottom_left", "top_right", "top_left", "center", "bottom_center", "top_center", "tile")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        p = FONT_DIR / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()  # type: ignore[return-value]


def _anchor(pos: str, W: int, H: int, w: int, h: int, margin: int) -> tuple[int, int]:
    return {
        "bottom_right": (W - w - margin, H - h - margin),
        "bottom_left": (margin, H - h - margin),
        "top_right": (W - w - margin, margin),
        "top_left": (margin, margin),
        "center": ((W - w) // 2, (H - h) // 2),
        "bottom_center": ((W - w) // 2, H - h - margin),
        "top_center": ((W - w) // 2, margin),
    }.get(pos, (W - w - margin, H - h - margin))


def _hex_rgb(hexstr: str) -> tuple[int, int, int]:
    s = hexstr.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def add_watermark(rgb: np.ndarray, text: str | None = None, logo_path: str | Path | None = None,
                  position: str = "bottom_right", opacity: float = 0.6, scale: float = 0.22,
                  color: str = "#FFFFFF", margin_pct: float = 0.025, shadow: bool = True) -> np.ndarray:
    """scale: watermark width as a fraction of image width (text or logo). opacity 0..1."""
    base = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").convert("RGBA")
    W, H = base.size
    margin = int(min(W, H) * margin_pct)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    if logo_path:
        logo = Image.open(logo_path).convert("RGBA")
        target_w = max(32, int(W * scale))
        ratio = target_w / logo.width
        logo = logo.resize((target_w, max(1, int(logo.height * ratio))), Image.LANCZOS)
        a = logo.getchannel("A").point(lambda v: int(v * opacity))
        logo.putalpha(a)
        stamp = logo
    else:
        text = text or "© Preview"
        # fit font so text width ~= scale * W
        size = max(12, int(W * scale / max(1, len(text)) * 1.9))
        font = _font(size)
        d = ImageDraw.Draw(layer)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # shrink if too wide
        while tw > W * scale * 1.15 and size > 10:
            size = int(size * 0.9)
            font = _font(size)
            bbox = d.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(2, size // 8)
        stamp = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp)
        r, g, b = _hex_rgb(color)
        alpha = int(255 * opacity)
        if shadow:
            sd.text((pad - bbox[0] + max(1, size // 24), pad - bbox[1] + max(1, size // 24)), text, font=font,
                    fill=(0, 0, 0, int(alpha * 0.55)))
        sd.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(r, g, b, alpha))

    if position == "tile":
        sw, sh = stamp.size
        rot = stamp.rotate(30, expand=True, resample=Image.BICUBIC)
        rw, rh = rot.size
        step_x, step_y = int(rw * 1.6), int(rh * 1.8)
        for y in range(-rh, H + rh, step_y):
            for x in range(-rw, W + rw, step_x):
                layer.alpha_composite(rot, (x + (y // step_y % 2) * step_x // 2, y))
    else:
        x, y = _anchor(position, W, H, stamp.width, stamp.height, margin)
        layer.alpha_composite(stamp, (x, y))

    out = Image.alpha_composite(base, layer).convert("RGB")
    return np.asarray(out).astype(np.float32) / 255.0
