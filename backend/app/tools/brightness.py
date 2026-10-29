"""Slider-style global/regional adjustments: exposure, contrast, saturation, temperature, shadows, highlights."""
from __future__ import annotations

import numpy as np

from .composite import feather


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def adjust(rgb: np.ndarray, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0,
           temperature: float = 0.0, shadows: float = 0.0, highlights: float = 0.0,
           region_mask: np.ndarray | None = None) -> np.ndarray:
    """All sliders are in [-100, 100] like a customer-facing UI; 0 = unchanged.

    brightness: exposure in linear light, +100 = +1.5 EV.
    contrast:   S-curve around mid-grey.
    saturation: chroma scale (-100 = greyscale).
    temperature: warm (+) / cool (-) white-balance shift.
    shadows/highlights: lift shadows / recover highlights.
    region_mask: optional HxW float mask; adjustments are applied only there (feathered).
    """
    x = rgb.astype(np.float32)
    lin = _srgb_to_linear(x)

    if brightness:
        lin = lin * (2.0 ** (1.5 * brightness / 100.0))

    if temperature:
        t = temperature / 100.0
        lin = lin * np.array([1 + 0.12 * t, 1 + 0.02 * t, 1 - 0.12 * t], dtype=np.float32)[None, None, :]

    if shadows or highlights:
        y = _luma(lin)
        if shadows:
            w = np.clip(1 - y / 0.35, 0, 1) ** 1.5
            lin = lin * (1 + 0.8 * (shadows / 100.0) * w)[..., None]
        if highlights:
            w = np.clip((y - 0.55) / 0.45, 0, 1) ** 1.2
            lin = lin * (1 - 0.6 * (highlights / 100.0) * w)[..., None]

    out = _linear_to_srgb(lin)

    if contrast:
        c = contrast / 100.0
        k = 1 + 0.9 * c if c > 0 else 1 + 0.6 * c
        out = 0.5 + (out - 0.5) * k

    if saturation:
        s = 1 + saturation / 100.0
        y = _luma(out)[..., None]
        out = y + (out - y) * s

    out = np.clip(out, 0, 1)
    if region_mask is not None:
        a = feather(region_mask, max(2, int(0.01 * max(rgb.shape[:2]))))[..., None]
        out = rgb * (1 - a) + out * a
    return out


def sharpen(rgb: np.ndarray, amount: float = 50.0) -> np.ndarray:
    import cv2
    a = amount / 100.0
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    return np.clip(rgb + (rgb - blur) * (1.2 * a), 0, 1)
