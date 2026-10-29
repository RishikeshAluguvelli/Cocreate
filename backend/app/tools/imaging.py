"""Image I/O helpers. All tools work on float32 RGB arrays in [0,1] unless stated otherwise."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

JPEG_QUALITY = 94


def load_rgb(path: str | Path) -> np.ndarray:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def load_pil(path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB")


def save_rgb(arr: np.ndarray, path: str | Path, quality: int = JPEG_QUALITY) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(arr).save(path, "JPEG", quality=quality, subsampling=0, optimize=True)
    return path


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """mask: float [0,1] or bool HxW -> 8-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = (np.clip(mask.astype(np.float32), 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(m, "L").save(path, "PNG")
    return path


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load any mask image as float [0,1] HxW. Uses alpha if present, else luminance."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        m = im.getchannel("A")
    else:
        m = im.convert("L")
    if size and m.size != size:
        m = m.resize(size, Image.BILINEAR)
    return np.asarray(m).astype(np.float32) / 255.0


def thumbnail(path: str | Path, long_side: int, out: str | Path | None = None, quality: int = 82) -> Image.Image:
    im = load_pil(path)
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        im.save(out, "JPEG", quality=quality)
    return im


def pil_to_b64_jpeg(im: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def crop_norm(im: Image.Image, box: list[float] | tuple[float, float, float, float]) -> Image.Image:
    """Crop with a normalized [x0,y0,x1,y1] box (0..1)."""
    w, h = im.size
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0.0, x0), min(1.0, x1)))
    y0, y1 = sorted((max(0.0, y0), min(1.0, y1)))
    return im.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 8), max(int(y1 * h), int(y0 * h) + 8)))


def fit_to(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize an RGB float array to exactly (w,h) with Lanczos."""
    w, h = size_wh
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    im = to_pil(arr).resize((w, h), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0
