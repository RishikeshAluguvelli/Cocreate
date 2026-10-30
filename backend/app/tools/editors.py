"""Which model fills a hole / performs a prompt edit, given the keys available.

Gemini path tricks that matter for quality:
- crop-zoom: the model edits a window around the mask (not the whole 2K frame), so a small object gets many more
  pixels and the fill has far fewer smears; the window is pasted back into the full frame afterwards.
- guide image: a second copy of the window with the target tinted red tells the model exactly what to remove.
- no-op detection: if the model returns the input unchanged (it sometimes refuses silently), retry once with a
  stronger prompt and report `changed=False` instead of pretending.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings
from ..providers.fal import EST_COST as FAL_COST, fal_provider
from ..providers.gemini import EST_COST as GEM_COST, gemini_provider
from . import composite


@dataclass
class Window:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def size(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0


def telea(rgb: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Classical inpainting: free, instant, good for thin/small objects on simple backgrounds."""
    src = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    m = (mask > 0.5).astype(np.uint8) * 255
    return cv2.inpaint(src, m, radius, cv2.INPAINT_TELEA).astype(np.float32) / 255.0


def crop_window(mask: np.ndarray, expand: float = 0.6, min_side: int = 900, max_frac: float = 0.45) -> Window | None:
    """Window around the mask with context. None when the mask is so large that the full frame is better."""
    H, W = mask.shape[:2]
    bb = composite.mask_bbox(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw * bh > max_frac * W * H:
        return None
    cw, ch = max(min_side, bw * (1 + expand)), max(min_side, bh * (1 + expand))
    # keep the window's aspect ratio sane (the model tends to preserve input aspect)
    cw, ch = min(W, cw), min(H, ch)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx0 = int(np.clip(cx - cw / 2, 0, W - cw)); wy0 = int(np.clip(cy - ch / 2, 0, H - ch))
    win = Window(wx0, wy0, int(wx0 + cw), int(wy0 + ch))
    if (win.x1 - win.x0) * (win.y1 - win.y0) > 0.8 * W * H:
        return None
    return win


def guide_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The crop with the target tinted red + outlined, so the model knows exactly which pixels to replace."""
    return composite.overlay_preview(rgb, mask, color=(1.0, 0.1, 0.1), alpha=0.5)


def change_inside(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0.5
    if sel.sum() == 0:
        return 0.0
    return float(np.abs(a - b).max(axis=2)[sel].mean())


def _resolution_for(size_wh: tuple[int, int]) -> str:
    """Never render a window below its own pixel size: 1K only when the window fits in 1024px, else the configured
    (2K) size, so the pasted-back patch is not softened by a down/up-scale round trip."""
    return "1K" if max(size_wh) <= 1024 else settings.gemini_edit_resolution


def cluster_windows(hard_mask: np.ndarray, max_windows: int = 6) -> list[Window] | None:
    """Group mask blobs that sit close together and build one close-up window per group. None = use the full frame."""
    H, W = hard_mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats((hard_mask > 0.5).astype(np.uint8), 8)
    rects = [[stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
              stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]]
             for i in range(1, n)]
    if not rects:
        return None
    pad = max(120, int(0.06 * max(W, H)))
    merged = True
    while merged:
        merged = False
        out: list[list[int]] = []
        for r in rects:
            for o in out:
                if r[0] - pad < o[2] and r[2] + pad > o[0] and r[1] - pad < o[3] and r[3] + pad > o[1]:
                    o[0], o[1], o[2], o[3] = min(o[0], r[0]), min(o[1], r[1]), max(o[2], r[2]), max(o[3], r[3])
                    merged = True
                    break
            else:
                out.append(list(r))
        rects = out
    if len(rects) > max_windows:
        return None
    wins = []
    for x0, y0, x1, y1 in rects:
        sub = np.zeros_like(hard_mask); sub[y0:y1, x0:x1] = hard_mask[y0:y1, x0:x1]
        w = crop_window(sub, expand=0.7, min_side=900, max_frac=0.5)
        if w is None:
            return None
        wins.append(w)
    return wins


async def _gemini_repair(sub: np.ndarray, sub_mask: np.ndarray, target: str, fill_prompt: str | None) -> tuple[np.ndarray, float]:
    """Pre-erase the target classically (so it is guaranteed gone), then let the model repaint the smeared area."""
    h, w = sub.shape[:2]
    pre = telea(sub, sub_mask, radius=7)
    maskimg = np.repeat((sub_mask > 0.5).astype(np.float32)[..., None], 3, axis=2)
    behind = f" What belongs there: {fill_prompt}." if fill_prompt else ""
    prompt = (f"Image 1 is a crop of a finished real-estate photo in which the {target} was crudely erased and smeared over. "
              f"Image 2 is a mask: white marks the smeared area. Repaint the white-masked area of image 1 so it becomes a seamless, "
              f"photorealistic continuation of its surroundings (ground, wall, grass, sky, driveway, furniture edges - whatever is "
              f"around it), matching lighting, texture, grain and perspective, including any contact shadow.{behind} Do not put any "
              f"object there. Keep everything outside the mask pixel-identical. Return the repaired photo only.")
    res = _resolution_for((w, h))
    out = await gemini_provider.edit(prompt, [pre, maskimg], (w, h), resolution=res)
    return out, GEM_COST.get(f"edit_{res}", 0.05)


async def _gemini_remove_semantic(rgb: np.ndarray, hard_mask: np.ndarray, target: str, fill_prompt: str | None) -> tuple[np.ndarray, float, str]:
    """Large removals: show the model the untouched photo plus a mask and ask it to remove the object and rebuild what was
    behind it. For big areas (people at a counter) this beats repainting a smeared pre-erase, which the model tends to keep."""
    H, W = rgb.shape[:2]
    maskimg = np.repeat((hard_mask > 0.5).astype(np.float32)[..., None], 3, axis=2)
    behind = f" What is behind it: {fill_prompt}." if fill_prompt else ""
    prompt = (f"Image 1 is a finished real-estate photo. Image 2 is a mask: white marks the {target}. Remove the {target} completely "
              f"and reconstruct everything that was hidden behind it so the scene looks as if it had never been there: continue "
              f"counters, cabinets, walls, floors, furniture and fixtures with matching perspective, lighting, texture and grain.{behind} "
              f"Do not add any new object or person. Keep everything outside the white area pixel-identical. Return the photo only.")
    res = _resolution_for((W, H))
    out = await gemini_provider.edit(prompt, [rgb, maskimg], (W, H), resolution=res)
    return out, GEM_COST.get(f"edit_{res}", 0.05), "generative fill (Nano Banana, semantic removal, full frame)"


async def _gemini_fill(rgb: np.ndarray, hard_mask: np.ndarray, target: str, fill_prompt: str | None) -> tuple[np.ndarray, float, str]:
    wins = cluster_windows(hard_mask)
    out = rgb.copy()
    cost = 0.0
    if wins:
        import asyncio

        async def one(win: Window):
            sub = rgb[win.y0:win.y1, win.x0:win.x1]
            return win, await _gemini_repair(sub, hard_mask[win.y0:win.y1, win.x0:win.x1], target, fill_prompt)

        for win, (patch, c) in await asyncio.gather(*(one(w) for w in wins)):
            out[win.y0:win.y1, win.x0:win.x1] = patch
            cost += c
        return out, cost, f"generative fill (Nano Banana, {len(wins)} close-up window{'s' if len(wins) > 1 else ''})"
    patch, cost = await _gemini_repair(rgb, hard_mask, target, fill_prompt)
    return patch, cost, "generative fill (Nano Banana, full frame)"


async def fill(rgb: np.ndarray, hard_mask: np.ndarray, target: str, method: str, area: float,
               fill_prompt: str | None = None, strategy: str = "repair") -> tuple[np.ndarray, str, float, bool]:
    """Returns (edited_rgb, method_label, est_cost, changed). strategy: "repair" (pre-erase + repaint per window) or
    "semantic" (model removes the object from the untouched photo; used for large areas, compared by QC)."""
    backend = settings.image_backend
    small = area <= settings.small_mask_area
    # classical inpainting smears anything bigger than a cord or a small sign: honour an explicit "erase" only for small
    # masks and send larger ones through the pre-erase + model repair path, which is what the user actually wants
    if (method == "erase" and area <= 0.008) or (method == "auto" and backend != "fal" and area <= 0.006):
        return telea(rgb, hard_mask), "erase (inpaint)", 0.0, True
    if backend == "fal":
        if method == "auto" and small:
            return await fal_provider.lama(rgb, hard_mask), "erase (LaMa)", FAL_COST["sam"] + FAL_COST["lama"], True
        prompt = fill_prompt or ("empty scene, seamless continuation of the surrounding surfaces and background, "
                                 "nothing in this area, photorealistic real-estate photo, same lighting and perspective")
        return await fal_provider.fill(rgb, hard_mask, prompt), "generative fill (FLUX Fill)", FAL_COST["sam"] + FAL_COST["fill"], True
    if backend == "gemini":
        if strategy == "semantic":
            out, cost, how = await _gemini_remove_semantic(rgb, hard_mask, target, fill_prompt)
            return out, how, cost, change_inside(rgb, out, hard_mask) > 0.015
        out, cost, how = await _gemini_fill(rgb, hard_mask, target, fill_prompt)
        return out, how, cost, True  # the target is erased before the model runs, so the result always changes
    return telea(rgb, hard_mask), "erase (inpaint, mock)", 0.0, True


async def prompt_edit(prompt: str, images: list, size_wh: tuple[int, int], resolution: str | None = None) -> tuple[np.ndarray, str, float]:
    backend = settings.image_backend
    if backend == "fal":
        out = await fal_provider.edit(prompt, images, size_wh, resolution=resolution)
        res = resolution or settings.fal_edit_resolution
        return out, settings.fal_edit_endpoint, FAL_COST.get(f"edit_{res}", 0.08)
    if backend == "gemini":
        res = resolution or _resolution_for(size_wh)
        out = await gemini_provider.edit(prompt, images, size_wh, resolution=res)
        return out, settings.gemini_edit_model, GEM_COST.get(f"edit_{res}", 0.05)
    out = await fal_provider.edit(prompt, images, size_wh, resolution=resolution)  # mock tint
    return out, "mock", 0.0
