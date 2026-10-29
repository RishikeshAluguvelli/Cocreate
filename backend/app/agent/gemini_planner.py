"""Gemini planner: same tools and system prompt as the Anthropic path, driven through generateContent function calling."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .. import store
from ..config import settings
from ..models import Shoot
from ..providers.gemini import GeminiError, gemini_provider, image_part
from ..tools import imaging
from .executor import dispatch
from .system_prompt import SYSTEM_PROMPT
from .tools_schema import TOOLS

_ALLOWED = {"type", "description", "enum", "properties", "required", "items", "nullable"}


def _clean(schema: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k not in _ALLOWED:
            continue
        if k == "properties":
            out[k] = {pk: _clean(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _clean(v)
        else:
            out[k] = v
    return out


def gemini_tools() -> list[dict]:
    return [{"functionDeclarations": [{"name": t["name"], "description": t["description"], "parameters": _clean(t["input_schema"])} for t in TOOLS]}]


def _state_text(shoot: Shoot, ui: dict) -> str:
    from .loop import _shoot_state
    return _shoot_state(shoot, ui)


def _thumb_parts(shoot: Shoot) -> list[dict]:
    parts = []
    for i, p in enumerate(shoot.photos):
        parts.append({"text": f"Photo {i + 1}: {p.name} (photo_id {p.id}, version {p.head.index})"})
        im = imaging.load_pil(p.head.path)
        parts.append(image_part(im, settings.thumb_long_side, 78))
    return parts


def _strip_images(content: dict) -> dict:
    parts = [p for p in content.get("parts", []) if "inlineData" not in p]
    return {"role": content["role"], "parts": parts or [{"text": "[images omitted]"}]}


def _for_history(content: dict) -> dict:
    """Model turn as persisted: drop thought summaries (keep any part carrying a thoughtSignature)."""
    parts = [p for p in content.get("parts", []) if not (p.get("thought") and "thoughtSignature" not in p)]
    return {"role": content.get("role", "model"), "parts": parts or [{"text": ""}]}


async def gemini_turn(shoot: Shoot, text: str, ui: dict, emit, events_for_message: list[dict]) -> str:
    if shoot.llm_provider != "gemini":
        shoot.llm_messages, shoot.llm_provider = [], "gemini"
    user = {"role": "user", "parts": [{"text": "Current shoot state:\n" + _state_text(shoot, ui)}, *_thumb_parts(shoot),
                                      {"text": "Customer request: " + text}]}
    contents: list[dict] = [*shoot.llm_messages, user]
    new_history: list[dict] = [_strip_images(user)]
    final_text = ""
    gen_cfg = {"maxOutputTokens": 8000, "thinkingConfig": {"thinkingLevel": settings.gemini_planner_thinking, "includeThoughts": True}}
    nudges = 0
    n = len(shoot.photos)

    sep_pending = {"v": False}

    async def on_delta(kind: str, chunk: str) -> None:
        if kind == "text" and sep_pending["v"] and chunk.strip():
            sep_pending["v"] = False
            await emit("text", {"delta": "\n"})
        await emit(kind, {"delta": chunk})

    for _round in range(12):
        await emit("status", {"message": f"Looking at {n} photo{'s' if n != 1 else ''} and planning" if _round == 0
                              else "Reviewing results, deciding next step"})
        try:
            d = await gemini_provider.generate_stream(settings.gemini_planner_model, contents, system=SYSTEM_PROMPT,
                                                      tools=gemini_tools(), generation_config=gen_cfg, on_delta=on_delta)
        except GeminiError as e:
            await emit("error", {"message": str(e)[:400]})
            break
        cand = (d.get("candidates") or [{}])[0]
        content = cand.get("content") or {"role": "model", "parts": []}
        content.setdefault("role", "model")
        finish = cand.get("finishReason")
        parts = content.get("parts", [])
        if finish == "MALFORMED_FUNCTION_CALL" and nudges < 2:
            nudges += 1
            contents.append({"role": "user", "parts": [{"text": "Your last function call was malformed JSON. Call the tool again with valid arguments."}]})
            continue
        contents.append(content)
        new_history.append(_for_history(content))
        texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        if texts:
            chunk = "".join(texts)
            if final_text and chunk.strip() and not final_text.endswith(("\n", " ")):
                final_text += "\n"  # the separator was already streamed by on_delta
            final_text += chunk  # already streamed as deltas
        sep_pending["v"] = bool(final_text.strip())
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            break

        async def run_one(call: dict) -> list[dict]:
            name, args = call.get("name"), call.get("args") or {}
            await emit("tool_start", {"id": name, "name": name, "input": args})
            events_for_message.append({"type": "tool_start", "name": name, "input": args})
            try:
                result, image = await dispatch(shoot, emit, name, args)
                await emit("tool_result", {"id": name, "name": name, "result": result})
                events_for_message.append({"type": "tool_result", "name": name, "result": result})
                out = [{"functionResponse": {"name": name, "response": {"result": result}}}]
                if image:
                    out.append({"inlineData": {"mimeType": image["source"]["media_type"], "data": image["source"]["data"]}})
                return out
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:400]}"
                await emit("tool_result", {"id": name, "name": name, "error": msg})
                events_for_message.append({"type": "tool_result", "name": name, "error": msg})
                return [{"functionResponse": {"name": name, "response": {"error": msg}}}]

        results = await asyncio.gather(*(run_one(c) for c in calls))
        resp = {"role": "user", "parts": [p for group in results for p in group]}
        contents.append(resp)
        new_history.append(_strip_images(resp))

    shoot.llm_messages = [*shoot.llm_messages, *new_history]
    shoot.total_cost_usd += 0.01
    store.save(shoot)
    return final_text.strip()
