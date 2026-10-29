"""Automatic quality control after generative edits: a vision model compares before/after against the instruction and
returns a verdict the planner (and the UI) can act on. Cheap (~$0.004) and fast (~3 s)."""
from __future__ import annotations

import json
import re

from ..config import settings
from ..providers.gemini import EST_COST, GeminiError, gemini_provider, image_part
from . import composite, imaging

QC_PROMPT = """You are the quality-control reviewer of a real-estate photo editing service.
Image A = BEFORE the edit, Image B = AFTER, Image C = image B with every pixel that differs from A tinted magenta
(so you can see exactly where the edit touched the photo; small objects can be easy to miss without it).
The requested edit was:
---
{instruction}
---
First look at the magenta regions in C and compare those spots in A and B. Judge strictly, like a demanding
photographer. Reply in JSON:
{{"fulfilled": 0-5, "collateral": 0-5, "artifacts": 0-5, "verdict": "pass" | "needs_review", "reason": "<= 25 words"}}
fulfilled: 5 = the request is completely and convincingly done, 0 = nothing relevant changed.
collateral: 0 = nothing outside the request changed, 5 = large unrelated changes (layout, furniture, colours, crop).
artifacts: 0 = none, 5 = obvious smears, warped geometry, duplicated or ghost objects, wrong perspective.
verdict is "pass" only when fulfilled >= 4 and collateral <= 1 and artifacts <= 1."""


async def qc_check(instruction: str, before_path: str, after_path: str, long_side: int = 1024) -> dict:
    """Returns {"verdict", "fulfilled", "collateral", "artifacts", "reason", "cost_usd"}; never raises."""
    if settings.image_backend != "gemini" or not settings.qc_enabled:
        return {"verdict": "skipped", "reason": "QC disabled", "cost_usd": 0.0}
    try:
        a, b = imaging.load_rgb(before_path), imaging.load_rgb(after_path)
        parts = [{"text": "Image A (before):"}, image_part(a, long_side, 80),
                 {"text": "Image B (after):"}, image_part(b, long_side, 80),
                 {"text": "Image C (after, changed pixels in magenta):"}, image_part(composite.diff_overlay(a, b), long_side, 80),
                 {"text": QC_PROMPT.format(instruction=instruction.strip()[:1200])}]
        d = await gemini_provider.generate(settings.qc_model or settings.gemini_planner_model, [{"role": "user", "parts": parts}],
                                           generation_config={"responseMimeType": "application/json", "maxOutputTokens": 1200,
                                                              "thinkingConfig": {"thinkingLevel": "minimal"}}, timeout=90)
        txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
        try:
            out = json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, re.S)
            out = json.loads(m.group(0)) if m else {}
        verdict = out.get("verdict")
        if verdict not in ("pass", "needs_review"):
            f, c, a = out.get("fulfilled", 0), out.get("collateral", 0), out.get("artifacts", 0)
            verdict = "pass" if (f or 0) >= 4 and (c or 0) <= 1 and (a or 0) <= 1 else "needs_review"
        return {"verdict": verdict, "fulfilled": out.get("fulfilled"), "collateral": out.get("collateral"),
                "artifacts": out.get("artifacts"), "reason": str(out.get("reason", ""))[:200], "cost_usd": EST_COST["qc"]}
    except (GeminiError, Exception) as e:  # noqa: BLE001
        return {"verdict": "skipped", "reason": f"QC unavailable: {str(e)[:80]}", "cost_usd": 0.0}
