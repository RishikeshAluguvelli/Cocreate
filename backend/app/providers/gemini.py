"""Google Gemini adapter (REST, no SDK): object detection -> boxes, Nano Banana image edits, and a chat/tool-calling
helper used by the Gemini planner. One API key covers all three."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from ..config import settings
from ..tools import imaging

BASE = "https://generativelanguage.googleapis.com/v1beta"
EST_COST = {"detect": 0.001, "edit_1K": 0.034, "edit_2K": 0.05, "edit_4K": 0.113, "planner_turn": 0.01, "qc": 0.004}


class GeminiError(RuntimeError):
    pass


def _b64_jpeg(img: np.ndarray | Image.Image | str | Path, long_side: int | None = None, quality: int = 90) -> str:
    if isinstance(img, (str, Path)):
        im = imaging.load_pil(img)
    elif isinstance(img, np.ndarray):
        im = imaging.to_pil(img)
    else:
        im = img.convert("RGB")
    if long_side:
        im = im.copy(); im.thumbnail((long_side, long_side))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_part(img, long_side: int | None = None, quality: int = 90) -> dict:
    return {"inlineData": {"mimeType": "image/jpeg", "data": _b64_jpeg(img, long_side, quality)}}


def _parse_json_list(txt: str) -> list:
    """Parse a JSON list from model output; salvages the individual objects when the list itself is malformed."""
    try:
        objs = json.loads(txt)
        return objs if isinstance(objs, list) else []
    except Exception:
        pass
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            objs = json.loads(m.group(0))
            if isinstance(objs, list):
                return objs
        except Exception:
            pass
    out = []
    for frag in re.findall(r"\{[^{}]*\}", txt, re.S):
        try:
            out.append(json.loads(frag))
        except Exception:
            continue
    return out


class GeminiProvider:
    def __init__(self) -> None:
        self.key = settings.gemini_api_key
        self.enabled = bool(self.key)

    async def generate(self, model: str, contents: list[dict], *, system: str | None = None, tools: list[dict] | None = None,
                       generation_config: dict | None = None, timeout: float = 240) -> dict:
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = tools
        if generation_config:
            body["generationConfig"] = generation_config
        async with httpx.AsyncClient(timeout=timeout) as c:
            for attempt in range(6):
                try:
                    r = await c.post(f"{BASE}/models/{model}:generateContent", params={"key": self.key}, json=body)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    if attempt < 5:
                        await asyncio.sleep(3 + 4 * attempt)
                        continue
                    raise GeminiError(f"Gemini {model}: network error {type(e).__name__}") from e
                if r.status_code == 200:
                    return r.json()
                msg = ""
                try:
                    msg = r.json().get("error", {}).get("message", "")
                except Exception:
                    msg = r.text[:300]
                # drop unsupported thinking config and retry once
                if r.status_code == 400 and "thinking" in msg.lower() and generation_config and "thinkingConfig" in generation_config:
                    generation_config.pop("thinkingConfig", None); body["generationConfig"] = generation_config
                    continue
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                    await asyncio.sleep(min(45, 3 * (2 ** attempt)))  # 3, 6, 12, 24, 45 s
                    continue
                raise GeminiError(f"Gemini {model} HTTP {r.status_code}: {msg[:300]}")
        raise GeminiError(f"Gemini {model}: retries exhausted")

    async def generate_stream(self, model: str, contents: list[dict], *, system: str | None = None, tools: list[dict] | None = None,
                              generation_config: dict | None = None, timeout: float = 240, on_delta=None) -> dict:
        """streamGenerateContent (SSE). `on_delta(kind, text)` is awaited for every text chunk, kind = 'text' | 'thinking'.
        Returns the same shape as generate(): {'candidates': [{'content': {...}, 'finishReason': ...}]}. Falls back to the
        non-streaming call when the stream cannot be opened."""
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = tools
        if generation_config:
            body["generationConfig"] = generation_config
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=20)) as c:
            for attempt in range(5):
                try:
                    async with c.stream("POST", f"{BASE}/models/{model}:streamGenerateContent",
                                        params={"key": self.key, "alt": "sse"}, json=body) as r:
                        if r.status_code != 200:
                            await r.aread()
                            try:
                                msg = r.json().get("error", {}).get("message", "")
                            except Exception:
                                msg = r.text[:300]
                            if r.status_code == 400 and "thinking" in msg.lower() and generation_config and "thinkingConfig" in generation_config:
                                generation_config.pop("thinkingConfig", None); body["generationConfig"] = generation_config
                                continue
                            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                                await asyncio.sleep(min(45, 3 * (2 ** attempt)))
                                continue
                            raise GeminiError(f"Gemini {model} HTTP {r.status_code}: {msg[:300]}")
                        parts: list[dict] = []
                        finish = None
                        usage = None
                        feedback = None
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if not payload:
                                continue
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            cand = (chunk.get("candidates") or [{}])[0]
                            for p in cand.get("content", {}).get("parts", []) or []:
                                if "text" in p:
                                    is_thought = bool(p.get("thought"))
                                    prev = parts[-1] if parts else None
                                    if prev is not None and "text" in prev and bool(prev.get("thought")) == is_thought and "thoughtSignature" not in prev:
                                        prev["text"] += p["text"]
                                        if "thoughtSignature" in p:
                                            prev["thoughtSignature"] = p["thoughtSignature"]
                                    else:
                                        parts.append(dict(p))
                                    if on_delta and p["text"]:
                                        await on_delta("thinking" if is_thought else "text", p["text"])
                                else:
                                    parts.append(p)
                            if cand.get("finishReason"):
                                finish = cand["finishReason"]
                            usage = chunk.get("usageMetadata") or usage
                            feedback = chunk.get("promptFeedback") or feedback
                        return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
                                "usageMetadata": usage, "promptFeedback": feedback}
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    if attempt < 4:
                        await asyncio.sleep(3 + 4 * attempt)
                        continue
                    raise GeminiError(f"Gemini {model}: network error {type(e).__name__}") from e
        raise GeminiError(f"Gemini {model}: retries exhausted")

    # ---------- detection: text -> boxes ----------
    async def detect(self, rgb: np.ndarray, target: str, restrict_box: list[float] | None = None, max_objects: int = 24) -> list[dict]:
        """Returns [{'box': [x0,y0,x1,y1] normalized 0..1, 'label': str}] for every instance of `target`."""
        H, W = rgb.shape[:2]
        where = ""
        if restrict_box:
            where = (f" Only consider the region with normalized box [x0={restrict_box[0]:.2f}, y0={restrict_box[1]:.2f}, "
                     f"x1={restrict_box[2]:.2f}, y1={restrict_box[3]:.2f}].")
        prompt = (f"Detect every instance of: {target}. Be precise and tight; include partially visible ones; "
                  f"do not include things that are not {target}.{where} Return a JSON list where each item has "
                  f"\"box_2d\": [ymin, xmin, ymax, xmax] normalized to 0-1000 and \"label\": short text. "
                  f"Return [] if there is none. At most {max_objects} items.")
        t0 = time.time()
        d = await self.generate(settings.gemini_vision_model,
                                [{"role": "user", "parts": [image_part(rgb, 1024, 88), {"text": prompt}]}],
                                generation_config={"responseMimeType": "application/json", "maxOutputTokens": 4000,
                                                   "thinkingConfig": {"thinkingLevel": "minimal"}})
        txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        objs = _parse_json_list(txt)
        out = []
        for o in objs:
            b = o.get("box_2d") if isinstance(o, dict) else None
            if not b or len(b) != 4:
                continue
            ymin, xmin, ymax, xmax = [max(0.0, min(1.0, float(v) / 1000.0)) for v in b]
            if xmax - xmin < 0.002 or ymax - ymin < 0.002:
                continue
            out.append({"box": [xmin, ymin, xmax, ymax], "label": str(o.get("label", target))})
        return out

    # ---------- Nano Banana edit ----------
    async def edit(self, prompt: str, images: list, size_wh: tuple[int, int], resolution: str | None = None) -> np.ndarray:
        """images: list of ndarray/path; the first is the photo to edit. Returns RGB float array resized to size_wh."""
        res = resolution or settings.gemini_edit_resolution
        parts = [image_part(img, None if i == 0 else 1024, 92) for i, img in enumerate(images)] + [{"text": prompt}]
        d = await self.generate(settings.gemini_edit_model, [{"role": "user", "parts": parts}],
                                generation_config={"responseModalities": ["IMAGE"], "imageConfig": {"imageSize": res}},
                                timeout=300)
        cand = d.get("candidates", [{}])[0]
        for p in cand.get("content", {}).get("parts", []):
            if "inlineData" in p:
                raw = base64.b64decode(p["inlineData"]["data"])
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                arr = np.asarray(im).astype(np.float32) / 255.0
                return _match_frame(arr, size_wh)
        reason = cand.get("finishReason") or d.get("promptFeedback", {}).get("blockReason") or "no image returned"
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))[:200]
        raise GeminiError(f"edit produced no image ({reason}) {text}")


def _match_frame(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize the model output to the original frame. If the aspect ratio drifted slightly, centre-crop first
    so pixels line up instead of stretching."""
    W, H = size_wh
    h, w = arr.shape[:2]
    ar_in, ar_out = W / H, w / h
    if abs(ar_in - ar_out) / ar_in > 0.01:
        if ar_out > ar_in:  # output too wide -> crop width
            nw = int(round(h * ar_in)); x0 = (w - nw) // 2; arr = arr[:, x0:x0 + nw]
        else:
            nh = int(round(w / ar_in)); y0 = (h - nh) // 2; arr = arr[y0:y0 + nh]
    return imaging.fit_to(arr, (W, H))


gemini_provider = GeminiProvider()
