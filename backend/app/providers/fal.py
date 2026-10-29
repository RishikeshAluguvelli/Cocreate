"""fal.ai adapter: SAM-3 (text/point masks), LaMa (erase), FLUX Fill (generative fill), Nano Banana (prompt edits).

Every call returns numpy arrays; uploads go to fal's CDN so the demo server never needs a public URL.
Set COCREATE_MOCK=1 to get plausible placeholders without a key (for UI/agent development).
"""
from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from ..config import settings
from ..tools import imaging

MOCK = os.environ.get("COCREATE_MOCK", "") not in ("", "0", "false")

# rough per-call prices (USD) shown in the UI; refine with fal's pricing endpoint
EST_COST = {"sam": 0.01, "lama": 0.01, "fill": 0.05, "edit_1K": 0.05, "edit_2K": 0.08, "edit_4K": 0.15}


@dataclass
class MaskResult:
    mask: np.ndarray                       # HxW float union of accepted instances
    instances: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)  # normalized cxcywh when available

    @property
    def found(self) -> bool:
        return bool((self.mask > 0.5).any())


class FalProvider:
    def __init__(self) -> None:
        self.enabled = bool(settings.fal_key) and not MOCK
        if settings.fal_key:
            os.environ.setdefault("FAL_KEY", settings.fal_key)
        self._fal = None
        if self.enabled:
            import fal_client  # lazy so the app runs without the key
            self._fal = fal_client
        self._upload_cache: dict[str, str] = {}

    # ---------- uploads ----------
    async def upload_path(self, path: str | Path) -> str:
        key = str(Path(path).resolve()) + f":{Path(path).stat().st_mtime_ns}"
        if key in self._upload_cache:
            return self._upload_cache[key]
        if not self.enabled:
            return f"mock://{Path(path).name}"
        url = await self._fal.upload_file_async(str(path))
        self._upload_cache[key] = url
        return url

    async def upload_array(self, arr: np.ndarray, fmt: str = "png") -> str:
        if not self.enabled:
            return "mock://array"
        buf = io.BytesIO()
        if arr.ndim == 2:
            Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(buf, "PNG")
            ct = "image/png"
        else:
            im = imaging.to_pil(arr)
            if fmt == "png":
                im.save(buf, "PNG"); ct = "image/png"
            else:
                im.save(buf, "JPEG", quality=95); ct = "image/jpeg"
        return await self._fal.upload_async(buf.getvalue(), ct)

    async def _download_rgb(self, url: str) -> np.ndarray:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.get(url); r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        return np.asarray(im).astype(np.float32) / 255.0

    async def _download_mask(self, url: str, size_wh: tuple[int, int]) -> np.ndarray:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.get(url); r.raise_for_status()
        im = Image.open(io.BytesIO(r.content))
        m = im.getchannel("A") if im.mode in ("RGBA", "LA") else im.convert("L")
        if m.size != size_wh:
            m = m.resize(size_wh, Image.BILINEAR)
        return (np.asarray(m).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    async def _run(self, endpoint: str, arguments: dict) -> dict:
        return await self._fal.subscribe_async(endpoint, arguments=arguments, with_logs=False)

    # ---------- SAM-3: text / point / box prompted segmentation ----------
    async def segment(self, image_path: str | Path, prompt: str | None, size_wh: tuple[int, int],
                      points: list[dict] | None = None, boxes: list[dict] | None = None,
                      max_masks: int = 12, min_score: float = 0.35) -> MaskResult:
        W, H = size_wh
        if not self.enabled:
            # mock: a box in the lower-middle of the frame, or around the given points
            m = np.zeros((H, W), np.float32)
            if points:
                for p in points:
                    x, y = int(p["x"]), int(p["y"])
                    m[max(0, y - H // 12): y + H // 12, max(0, x - W // 12): x + W // 12] = 1
            elif boxes:
                for b in boxes:
                    m[int(b["y_min"]): int(b["y_max"]), int(b["x_min"]): int(b["x_max"])] = 1
            else:
                m[int(H * 0.55): int(H * 0.8), int(W * 0.4): int(W * 0.6)] = 1
            return MaskResult(mask=m, instances=[m], scores=[0.9], boxes=[])
        url = await self.upload_path(image_path)
        args: dict = {"image_url": url, "apply_mask": False, "output_format": "png",
                      "return_multiple_masks": True, "max_masks": max_masks,
                      "include_scores": True, "include_boxes": True}
        if prompt:
            args["prompt"] = prompt
        if points:
            args["point_prompts"] = points
        if boxes:
            args["box_prompts"] = boxes
        res = await self._run(settings.fal_sam_endpoint, args)
        masks = res.get("masks") or ([res["image"]] if res.get("image") else [])
        meta = res.get("metadata") or []
        scores = res.get("scores") or [m.get("score") for m in meta]
        bxs = res.get("boxes") or [m.get("box") for m in meta]
        inst, keep_scores, keep_boxes = [], [], []
        for i, m in enumerate(masks):
            sc = scores[i] if i < len(scores) and scores[i] is not None else 1.0
            if sc < min_score and len(masks) > 1:
                continue
            arr = await self._download_mask(m["url"], (W, H))
            if arr.sum() < 16:
                continue
            inst.append(arr); keep_scores.append(float(sc)); keep_boxes.append(bxs[i] if i < len(bxs) and bxs[i] else [])
        union = np.clip(np.sum(inst, axis=0), 0, 1).astype(np.float32) if inst else np.zeros((H, W), np.float32)
        return MaskResult(mask=union, instances=inst, scores=keep_scores, boxes=keep_boxes)

    # ---------- erasers ----------
    async def lama(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return _mock_fill(rgb, mask)
        img_url = await self.upload_array(rgb, "png")
        mask_url = await self.upload_array(mask, "png")
        res = await self._run(settings.fal_lama_endpoint, {"image_url": img_url, "mask_image_url": mask_url})
        out = await self._download_rgb(res["image"]["url"])
        return imaging.fit_to(out, (rgb.shape[1], rgb.shape[0]))

    async def fill(self, rgb: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        if not self.enabled:
            return _mock_fill(rgb, mask)
        img_url = await self.upload_array(rgb, "png")
        mask_url = await self.upload_array(mask, "png")
        res = await self._run(settings.fal_fill_endpoint, {"image_url": img_url, "mask_url": mask_url, "prompt": prompt,
                                                           "output_format": "png", "safety_tolerance": "5"})
        out = await self._download_rgb(res["images"][0]["url"])
        return imaging.fit_to(out, (rgb.shape[1], rgb.shape[0]))

    # ---------- prompt-driven edit (Nano Banana) ----------
    async def edit(self, prompt: str, image_paths: list[str | Path], size_wh: tuple[int, int],
                   resolution: str | None = None) -> np.ndarray:
        if not self.enabled:
            base = imaging.load_rgb(image_paths[0])
            return np.clip(base * np.array([1.04, 1.0, 0.96], np.float32), 0, 1)
        urls = [await self.upload_path(p) for p in image_paths]
        args = {"prompt": prompt, "image_urls": urls, "resolution": resolution or settings.fal_edit_resolution,
                "output_format": "png", "aspect_ratio": "auto", "num_images": 1, "safety_tolerance": "5"}
        res = await self._run(settings.fal_edit_endpoint, args)
        out = await self._download_rgb(res["images"][0]["url"])
        return imaging.fit_to(out, size_wh)


def _mock_fill(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Telea inpainting stand-in so the pipeline can be exercised offline."""
    import cv2
    src = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    m = (mask > 0.5).astype(np.uint8) * 255
    out = cv2.inpaint(src, m, 7, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0


fal_provider = FalProvider()
