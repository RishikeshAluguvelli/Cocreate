"""Streaming planner loop: the planner sees the shoot, calls tools, and the UI receives SSE events as it goes.

History is append-only (never rewrite earlier turns) so preserved-thinking checks stay valid.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, AsyncIterator

from .. import store
from ..config import settings
from ..models import ChatMessage, Shoot
from ..tools import imaging
from .executor import CURRENT_UI, dispatch
from .system_prompt import SYSTEM_PROMPT
from .tools_schema import TOOLS

MOCK = os.environ.get("COCREATE_MOCK", "") not in ("", "0", "false")


def _edit_log(p) -> list[str]:
    """Chronological list of what has been done to a photo (head lineage only, so undos/reverts are reflected)."""
    chain = []
    v = p.head
    while v is not None and v.index > 0:
        chain.append(f"v{v.index}: {v.summary}")
        v = p.version(v.parent_id) if v.parent_id else None
    return list(reversed(chain))[-6:]


def _shoot_state(shoot: Shoot, ui: dict) -> str:
    rows = []
    for i, p in enumerate(shoot.photos):
        rows.append({"photo_id": p.id, "name": p.name, "size": f"{p.width}x{p.height}",
                     "version": p.head.index, "edit_log": _edit_log(p),
                     "brackets": len(p.bracket_paths), "references": len(p.reference_paths),
                     **({"customer_note": p.dataset_request[:160]} if p.dataset_request and ui.get("include_dataset_requests") else {})})
    ctx = {"shoot": shoot.name, "photos": rows, "logo_uploaded": bool(shoot.watermark_logo_path)}
    mode = ui.get("scope") or "shoot"
    sel = next((p for p in shoot.photos if p.id == ui.get("selected_photo_id")), None)
    if mode == "photo" and sel:
        ctx["scope"] = {"mode": "photo", "rule": f"edit ONLY '{sel.name}' (photo_id {sel.id}) unless the message names other photos"}
    else:
        ctx["scope"] = {"mode": "shoot", "rule": "whole shoot: decide per photo whether the request applies; 'this photo' means the selected one"}
    if ui.get("confirm_first"):
        ctx["confirm_first"] = ("ON: the user wants to approve the plan before anything runs. This turn, do NOT call editing "
                                "tools. Reply with a short plan: which photos, which tool, what stays unchanged, estimated cost; "
                                "then ask them to confirm. You may still call inspect_photo / sample_color. When the user confirms "
                                "('go ahead', 'yes', 'do it'), execute the plan exactly.")
    if shoot.memory_notes:
        ctx["conversation_memory"] = shoot.memory_notes
    if ui.get("selected_photo_id"):
        ctx["selected_photo_id"] = ui["selected_photo_id"]
    if ui.get("points"):
        ctx["points_placed_by_user"] = {"photo_id": ui.get("selected_photo_id"), "points": ui["points"],
                                        "note": "normalized x,y the user clicked on the selected photo"}
    m = ui.get("mask") or {}
    if m.get("png"):
        ctx["painted_area_by_user"] = {
            "photo_id": m.get("photo_id"), "area_pct": round(float(m.get("area") or 0) * 100, 1),
            "note": ("the user painted a region on this photo. For repaint / remove_object / generative_edit on that photo pass "
                     "painted_area=true: the painted region is added to what detection finds (or used alone when nothing is "
                     "detected). Use it when they say 'this area', 'the part I marked', 'also here', or to fix a spot a previous "
                     "edit missed. For repaint of a spot that was missed, reuse the same target and hex as the previous repaint.")}
    pc = ui.get("picked_color") or {}
    if pc.get("hex"):
        ctx["picked_color_by_user"] = {"hex": pc["hex"], "from": pc.get("from"),
                                       "note": "the user picked this colour with the eyedropper; 'the picked colour' / 'that colour' means this hex. Use it directly in repaint, no need to sample."}
    return json.dumps(ctx, ensure_ascii=False)


def _thumb_blocks(shoot: Shoot) -> list[dict]:
    blocks = []
    for i, p in enumerate(shoot.photos):
        im = imaging.load_pil(p.head.path)
        im.thumbnail((settings.thumb_long_side, settings.thumb_long_side))
        blocks.append({"type": "text", "text": f"Photo {i + 1}: {p.name} (photo_id {p.id}, version {p.head.index})"})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": imaging.pil_to_b64_jpeg(im, 78)}})
    return blocks


def _serialize(block: Any) -> dict:
    return block if isinstance(block, dict) else block.model_dump(exclude_none=True)


_shoot_locks: dict[str, asyncio.Lock] = {}


def _lock_for(shoot_id: str) -> asyncio.Lock:
    """Requests on different shoots run in parallel; requests on the same shoot are queued so two turns can never
    interleave their tool calls, transcript entries or chat messages."""
    lock = _shoot_locks.get(shoot_id)
    if lock is None:
        lock = _shoot_locks[shoot_id] = asyncio.Lock()
    return lock


async def run_turn(shoot: Shoot, text: str, ui: dict) -> AsyncIterator[dict]:
    """Yields SSE-ready events: {type, data}. Persists chat + planner transcript when done."""
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def emit(etype: str, data: dict) -> None:
        await queue.put({"type": etype, "data": data})

    thinking_acc: list[str] = []
    status_log: list[str] = []
    t_start = time.time()

    async def emit_tracked(etype: str, data: dict) -> None:
        if etype == "thinking":
            thinking_acc.append(data.get("delta", ""))
        elif etype in ("status", "progress"):
            m = data.get("message")
            if m and (not status_log or status_log[-1] != m):
                status_log.append(m)
        await emit(etype, data)

    async def worker() -> None:
        CURRENT_UI.set(ui)   # tools read the painted mask from here; the task's context keeps it per request
        lock = _lock_for(shoot.id)
        if lock.locked():
            await emit("status", {"message": "Waiting for the previous request on this shoot to finish"})
        async with lock:
            await _turn()

    async def _turn() -> None:
        events_for_message: list[dict] = []
        final_text = ""
        try:
            planner = settings.planner
            if planner == "anthropic":
                if shoot.llm_provider != "anthropic":
                    shoot.llm_messages, shoot.llm_provider = [], "anthropic"
                final_text = await _claude_turn(shoot, text, ui, emit_tracked, events_for_message)
            elif planner == "gemini":
                from .gemini_planner import gemini_turn
                final_text = await gemini_turn(shoot, text, ui, emit_tracked, events_for_message)
            else:
                final_text = await _mock_turn(shoot, text, ui, emit_tracked, events_for_message)
        except Exception as e:  # surface to the UI instead of dying silently
            await emit("error", {"message": f"{type(e).__name__}: {str(e)[:500]}"})
            final_text = final_text or "Something went wrong while running that request."
        sel = next((p for p in shoot.photos if p.id == ui.get("selected_photo_id")), None)
        scope = {"mode": ui.get("scope") or "shoot", "photo_name": sel.name if sel else None, "photos": len(shoot.photos)}
        shoot.chat.append(ChatMessage(role="user", text=text, scope=scope))
        shoot.chat.append(ChatMessage(role="assistant", text=final_text, events=events_for_message,
                                      thinking="".join(thinking_acc).strip(), status_log=status_log[-40:],
                                      seconds=round(time.time() - t_start, 1)))
        try:
            await _maintain_memory(shoot)
        except Exception:  # memory is best-effort
            pass
        store.save(shoot)
        await emit("done", {"text": final_text, "total_cost_usd": round(shoot.total_cost_usd, 3),
                            "seconds": round(time.time() - t_start, 1)})
        await queue.put(None)

    task = asyncio.create_task(worker())
    while True:
        ev = await queue.get()
        if ev is None:
            break
        if ev["type"] in ("version", "skipped", "tool_start", "tool_result", "error"):
            pass
        yield ev
    await task


async def _claude_turn(shoot: Shoot, text: str, ui: dict, emit, events_for_message: list[dict]) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    user_content = [{"type": "text", "text": "Current shoot state:\n" + _shoot_state(shoot, ui)}]
    user_content += _thumb_blocks(shoot)
    user_content.append({"type": "text", "text": "Customer request: " + text})
    messages: list[dict] = [*shoot.llm_messages, {"role": "user", "content": user_content}]
    new_history: list[dict] = [{"role": "user", "content": [b for b in user_content if b["type"] == "text"]}]
    # (we persist text-only user turns; the thumbnails are re-sent fresh each turn and are not part of the cached prefix)

    final_text = ""
    for _round in range(12):
        await emit("status", {"message": f"Looking at {len(shoot.photos)} photo{'s' if len(shoot.photos) != 1 else ''} and planning" if _round == 0 else "Reviewing results, deciding next step"})
        async with client.beta.messages.stream(
            model=settings.planner_model,
            max_tokens=settings.planner_max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=TOOLS,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": settings.planner_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        await emit("text", {"delta": event.delta.text})
                    elif event.delta.type == "thinking_delta" and event.delta.thinking:
                        await emit("thinking", {"delta": event.delta.thinking})
            response = await stream.get_final_message()

        assistant_blocks = [_serialize(b) for b in response.content]
        messages.append({"role": "assistant", "content": assistant_blocks})
        new_history.append({"role": "assistant", "content": assistant_blocks})
        chunk = "".join(b.text for b in response.content if b.type == "text")
        if final_text and chunk.strip() and not final_text.endswith(("\n", " ")):
            final_text += "\n"
        final_text += chunk

        if response.stop_reason == "refusal":
            await emit("error", {"message": "The planner declined this request."})
            break
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses or response.stop_reason == "max_tokens":
            break

        # run all tool calls of this turn concurrently, return all results in ONE user message
        async def run_one(tu):
            await emit("tool_start", {"id": tu.id, "name": tu.name, "input": tu.input})
            events_for_message.append({"type": "tool_start", "name": tu.name, "input": tu.input})
            try:
                result, image = await dispatch(shoot, emit, tu.name, tu.input)
                content: list[dict] = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
                if image:
                    content.append(image)
                await emit("tool_result", {"id": tu.id, "name": tu.name, "result": result})
                events_for_message.append({"type": "tool_result", "name": tu.name, "result": result})
                return {"type": "tool_result", "tool_use_id": tu.id, "content": content}
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:400]}"
                await emit("tool_result", {"id": tu.id, "name": tu.name, "error": msg})
                events_for_message.append({"type": "tool_result", "name": tu.name, "error": msg})
                return {"type": "tool_result", "tool_use_id": tu.id, "content": msg, "is_error": True}

        results = await asyncio.gather(*(run_one(tu) for tu in tool_uses))
        # strip image blocks from the persisted history to keep it small; the live request keeps them
        messages.append({"role": "user", "content": results})
        new_history.append({"role": "user", "content": [
            {**r, "content": [c for c in r["content"] if c.get("type") != "image"] if isinstance(r["content"], list) else r["content"]}
            for r in results]})

    shoot.llm_messages = [*shoot.llm_messages, *new_history]
    return final_text.strip()


# ---------------- long-horizon memory ----------------
MAX_HISTORY = 40          # planner transcript entries kept verbatim (each tool round is 2 entries)
MEMORY_CHARS = 1800


def _is_user_message(m: dict) -> bool:
    if m.get("role") != "user":
        return False
    content = m.get("content") if "content" in m else m.get("parts")
    if isinstance(content, str):
        return True
    for b in content or []:
        if isinstance(b, dict) and ("functionResponse" in b or b.get("type") == "tool_result"):
            return False
    return True


def _entry_text(m: dict) -> str:
    """Flatten one transcript entry (Anthropic or Gemini shape) into plain text."""
    out = []
    content = m.get("content") if "content" in m else m.get("parts")
    if isinstance(content, str):
        return content
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" or ("text" in b and not b.get("thought")):
            out.append(str(b.get("text", "")))
        elif b.get("type") == "tool_use":
            out.append(f"[called {b.get('name')} {json.dumps(b.get('input'))[:300]}]")
        elif "functionCall" in b:
            out.append(f"[called {b['functionCall'].get('name')} {json.dumps(b['functionCall'].get('args'))[:300]}]")
        elif b.get("type") == "tool_result":
            c = b.get("content")
            out.append("[result " + (c if isinstance(c, str) else json.dumps(c)[:300]) + "]")
        elif "functionResponse" in b:
            out.append("[result " + json.dumps(b["functionResponse"].get("response"))[:300] + "]")
    return "\n".join(out)


async def _maintain_memory(shoot: Shoot) -> None:
    """When the transcript grows past MAX_HISTORY entries, fold the oldest entries into `memory_notes` with a cheap
    summarisation call, so preferences and decisions survive even in very long chats."""
    if len(shoot.llm_messages) <= MAX_HISTORY:
        return
    cut = len(shoot.llm_messages) - MAX_HISTORY
    # cut on a REAL user message (not a tool-result turn, which also has role=user) so the kept transcript is valid
    while cut < len(shoot.llm_messages) and not _is_user_message(shoot.llm_messages[cut]):
        cut += 1
    if cut >= len(shoot.llm_messages):
        return
    old, shoot.llm_messages = shoot.llm_messages[:cut], shoot.llm_messages[cut:]
    transcript = "\n\n".join(f"{m.get('role')}: {_entry_text(m)}" for m in old)[:20000]
    prompt = ("Update the memory notes for an ongoing photo-editing chat. Keep ONLY facts that matter for future requests: "
              "customer preferences (colours as hex, styles, tone), decisions and approvals, things the customer rejected, "
              "pending follow-ups, and per-photo outcomes that are not obvious from the edit log. Max 12 short bullet lines.\n\n"
              f"EXISTING NOTES:\n{shoot.memory_notes or '(none)'}\n\nOLDER TRANSCRIPT TO FOLD IN:\n{transcript}\n\nNEW NOTES:")
    if settings.planner == "gemini":
        from ..providers.gemini import gemini_provider
        d = await gemini_provider.generate(settings.gemini_vision_model, [{"role": "user", "parts": [{"text": prompt}]}],
                                           generation_config={"maxOutputTokens": 800, "thinkingConfig": {"thinkingLevel": "minimal"}})
        notes = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
    elif settings.planner == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        r = await client.messages.create(model="claude-haiku-5-5", max_tokens=800, messages=[{"role": "user", "content": prompt}])
        notes = "".join(b.text for b in r.content if b.type == "text")
    else:
        notes = (shoot.memory_notes + "\n" + transcript[-600:])
    shoot.memory_notes = notes.strip()[-MEMORY_CHARS:]


def archive_chat(shoot: Shoot) -> None:
    """Start a fresh conversation on the same shoot: the old one stays readable, photo versions are untouched."""
    if shoot.chat:
        shoot.archived_chats.append({"id": f"c_{int(time.time())}", "created_at": shoot.chat[0].created_at,
                                     "ended_at": time.time(), "messages": [m.model_dump() for m in shoot.chat],
                                     "memory_notes": shoot.memory_notes})
        shoot.archived_chats = shoot.archived_chats[-10:]
    shoot.chat, shoot.llm_messages, shoot.memory_notes = [], [], ""
    store.save(shoot)


# ---------------- offline fallback planner (no Anthropic key) ----------------
async def _mock_turn(shoot: Shoot, text: str, ui: dict, emit, events_for_message: list[dict]) -> str:
    """Keyword planner so the UI and tools can be exercised without an API key."""
    t = text.lower()
    sel = ui.get("selected_photo_id") or (shoot.photos[0].id if shoot.photos else None)
    ids = ["all"] if ("all" in t or "every" in t or "shoot" in t) else [sel]

    async def call(name: str, inp: dict) -> dict:
        await emit("tool_start", {"id": name, "name": name, "input": inp})
        events_for_message.append({"type": "tool_start", "name": name, "input": inp})
        result, _ = await dispatch(shoot, emit, name, inp)
        await emit("tool_result", {"id": name, "name": name, "result": result})
        events_for_message.append({"type": "tool_result", "name": name, "result": result})
        return result

    if "watermark" in t or "logo" in t:
        m = re.search(r'["“](.+?)["”]', text)
        await call("add_watermark", {"photo_ids": ["all"], "text": m.group(1) if m else "© Preview", "use_logo": bool(shoot.watermark_logo_path) and "logo" in t})
        return "Added the watermark to every photo (offline planner)."
    if "undo" in t:
        await call("undo", {"photo_ids": ids})
        return "Stepped back one version."
    if "lot line" in t or "property line" in t or "boundary" in t:
        pts = ui.get("points") or [[0.25, 0.15], [0.62, 0.15], [0.62, 0.92], [0.22, 0.92]]
        await call("draw_lot_lines", {"photo_id": sel, "points": pts, "label": "LOT"})
        return "Drew lot lines; drag the corners to adjust (offline planner)."
    m = re.search(r"(bright|light|dark|exposure)", t)
    if m:
        amt = 40 if ("bright" in t or "light" in t) else -40
        if "little" in t or "slightly" in t or "bit" in t:
            amt = int(amt / 2)
        await call("adjust", {"photo_ids": ids, "brightness": amt})
        return f"Adjusted brightness {amt:+d} (offline planner)."
    m = re.search(r"remove (?:the |all |any )?(.+?)(?:\.|$| from| in )", t)
    if m:
        await call("remove_object", {"photo_ids": ids, "target": m.group(1).strip(), "points": ui.get("points")})
        return f"Removed '{m.group(1).strip()}' where found (offline planner, mock masks)."
    m = re.search(r"#([0-9a-f]{6})", t)
    if m and ("wall" in t or "floor" in t or "paint" in t):
        await call("repaint", {"photo_ids": ids, "target": "floor" if "floor" in t else "wall", "hex": "#" + m.group(1)})
        return "Repainted (offline planner)."
    await call("generative_edit", {"photo_ids": ids, "instruction": text, "protect": "auto"})
    return "Applied a generative edit (offline planner)."
