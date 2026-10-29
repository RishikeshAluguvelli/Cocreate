"""Run AutoHDR's 200 real customer requests through the production planner + tools and score the results.

Each record becomes a single-photo shoot (finished photo + its brackets + any customer references). The request text is
sent exactly as the customer wrote it. For every record we store: tools the planner called, versions produced, errors,
wall time, estimated cost, the fraction of pixels that changed, and (with --judge) a vision-model QC verdict comparing
before/after against the request.

Resumable: results live in data/eval/<run>/records/<idx>.json; already-finished records are skipped.

usage (from backend/):
  uv run python tests/eval_dataset.py --run baseline --concurrency 3 --judge
  uv run python tests/eval_dataset.py --run baseline --only 5,21,156 --force
  uv run python tests/eval_dataset.py --run baseline --summary-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import store  # noqa: E402
from app.agent.loop import run_turn  # noqa: E402
from app.config import settings  # noqa: E402
from app.demo_shoots import _fetch, _records  # noqa: E402
from app.models import Photo, Shoot, Version  # noqa: E402
from app.providers.gemini import gemini_provider, image_part  # noqa: E402
from app.tools import composite, imaging  # noqa: E402

EVAL_DIR = settings.data_dir / "eval"

# ---------------- request taxonomy (rule based; used to group results and to check tool choice) ----------------
REMOVAL_TOOLS = {"remove_object", "generative_edit"}
GEN_TOOLS = {"generative_edit"}
EXPECTED = {
    "remove": REMOVAL_TOOLS, "declutter": REMOVAL_TOOLS, "stains": REMOVAL_TOOLS, "reflection": REMOVAL_TOOLS,
    "shadows_rays": REMOVAL_TOOLS | {"adjust"},
    "tv_reference": GEN_TOOLS | {"tv_screen"}, "lights_on": GEN_TOOLS, "bedding": GEN_TOOLS, "lawn": GEN_TOOLS | {"repaint"},
    "add_generative": GEN_TOOLS,
    "colour": {"repaint", "sample_color", "generative_edit", "adjust"},
    "sharpen": {"adjust", "generative_edit"}, "exposure": {"adjust", "generative_edit"}, "other": set(),
}


def category(req: str) -> str:
    t = req.lower()
    if "tv screen" in t or "on the tv" in t or "to the tv" in t:
        return "tv_reference"
    if "turn on" in t or "add light" in t or "chandelier" in t or ("fire" in t and "fireplace" in t):
        return "lights_on"
    if "bedspread" in t or "wrinkl" in t:
        return "bedding"
    if "lawn" in t or "grass" in t or "ground a similar color green" in t:
        return "lawn"
    if any(w in t for w in ("declutter", "clear out", "clean out", "all of the items", "all things", "approved removals", "litter", "clutter")):
        return "declutter"
    if any(w in t for w in ("shadow", "sun ray", "flare", "the rays")):
        return "shadows_rays"
    if any(w in t for w in ("carpet", "stain", "dirt", "moss")):
        return "stains"
    if "reflection" in t:
        return "reflection"
    if any(w in t for w in ("colour", "color", "#", "bluer", "blue like", "darker green")):
        return "colour"
    if "sharpen" in t or "blurry" in t:
        return "sharpen"
    if "lighting" in t or "brighter" in t:
        return "exposure"
    if t.startswith("add") or " add " in t or "open the pool" in t or "window view" in t or "sky" in t or "workers" in t:
        return "add_generative"
    if "remove" in t:
        return "remove"
    return "other"


# ---------------- shoot construction ----------------
async def build_shoot(idx: int, rec: dict) -> Shoot:
    shoot = Shoot(name=f"eval {idx}: {rec['request'][:50]}")
    edited = await _fetch(rec["edited_photo"])
    brackets = await asyncio.gather(*(_fetch(u) for u in rec["inputs"]))
    refs = await asyncio.gather(*(_fetch(u) for u in rec.get("request_references", [])))
    with Image.open(edited) as im:
        w, h = im.size
    photo = Photo(shoot_id=shoot.id, name="Photo 1", width=w, height=h, original_path=str(edited),
                  bracket_paths=[str(b) for b in brackets], reference_paths=[str(x) for x in refs],
                  dataset_request=rec["request"].strip())
    pdir = store.photo_dir(photo)
    v0 = pdir / "v00_original.jpg"
    v0.write_bytes(edited.read_bytes())
    ver = Version(photo_id=photo.id, index=0, path=str(v0))
    photo.versions = [ver]
    photo.head_version_id = ver.id
    shoot.photos = [photo]
    store.save(shoot)
    return shoot


# ---------------- metrics ----------------
def changed_fraction(a_path: str, b_path: str, thresh: float = 0.06) -> float:
    a = imaging.load_pil(a_path); b = imaging.load_pil(b_path)
    a.thumbnail((512, 512)); b = b.resize(a.size)
    da = np.asarray(a).astype(np.float32) / 255; db = np.asarray(b).astype(np.float32) / 255
    d = np.abs(da - db).max(axis=2)
    return float((d > thresh).mean())


JUDGE_PROMPT = """You are the quality-control reviewer of a real-estate photo editing service.
Image A is the photo BEFORE the edit. Image B is AFTER. Image C is B with every pixel that differs from A tinted
magenta, so you can see exactly where the edit touched the photo (small objects are easy to miss without it).
The customer's request was:
---
{request}
---
First look at the magenta regions in C and compare those spots in A and B. Then judge strictly, as a demanding
photographer would. Answer in JSON with keys:
"fulfilled": 0-5 (5 = request completely and convincingly done; 0 = nothing relevant happened),
"collateral": 0-5 (0 = nothing outside the request changed; 5 = large unrelated changes, e.g. furniture, layout, colours or crop moved),
"artifacts": 0-5 (0 = no visible artifacts; 5 = obvious smears, warped geometry, blur, wrong perspective),
"verdict": "pass" | "partial" | "fail",
"notes": one sentence (max 30 words) on what is right or wrong.
A "pass" needs fulfilled >= 4, collateral <= 1, artifacts <= 1. If A and B are identical, fulfilled = 0."""


async def judge(request: str, before: str, after: str) -> dict:
    model = settings.gemini_planner_model
    a, b = imaging.load_rgb(before), imaging.load_rgb(after)
    parts = [{"text": "Image A (before):"}, image_part(a, 1024, 82),
             {"text": "Image B (after):"}, image_part(b, 1024, 82),
             {"text": "Image C (after, changed pixels in magenta):"}, image_part(composite.diff_overlay(a, b), 1024, 82),
             {"text": JUDGE_PROMPT.format(request=request.strip())}]
    d = await gemini_provider.generate(model, [{"role": "user", "parts": parts}],
                                       generation_config={"responseMimeType": "application/json", "maxOutputTokens": 2000,
                                                          "thinkingConfig": {"thinkingLevel": "low"}})
    txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
    try:
        out = json.loads(txt)
    except Exception:
        import re
        m = re.search(r"\{.*\}", txt, re.S)
        out = json.loads(m.group(0)) if m else {"verdict": "unparsed", "notes": txt[:200]}
    out["judge_model"] = model
    return out


# ---------------- one record ----------------
async def run_record(idx: int, rec: dict, out_path: Path, do_judge: bool) -> dict:
    t0 = time.time()
    res: dict = {"idx": idx, "id": rec["id"], "request": rec["request"].strip(), "category": category(rec["request"]),
                 "n_brackets": len(rec["inputs"]), "n_refs": len(rec.get("request_references", []))}
    try:
        shoot = await build_shoot(idx, rec)
        photo = shoot.photos[0]
        res["shoot_id"] = shoot.id
        events: list[dict] = []
        final_text = ""
        async for ev in run_turn(shoot, rec["request"].strip(), {"selected_photo_id": photo.id, "points": None, "scope": "shoot"}):
            if ev["type"] in ("tool_start", "tool_result", "skipped", "error", "version"):
                events.append(ev)
            if ev["type"] == "done":
                final_text = ev["data"].get("text", "")
        shoot = store.get(shoot.id) or shoot
        photo = shoot.photos[0]
        tools = [e["data"]["name"] for e in events if e["type"] == "tool_start"]
        tool_inputs = [{"name": e["data"]["name"], "input": e["data"].get("input")} for e in events if e["type"] == "tool_start"]
        tool_errors = [e["data"].get("error") for e in events if e["type"] == "tool_result" and e["data"].get("error")]
        skipped = [e["data"].get("reason") for e in events if e["type"] == "skipped"]
        errors = [e["data"].get("message") for e in events if e["type"] == "error"]
        head = photo.head
        res.update({
            "tools": tools, "tool_inputs": tool_inputs, "tool_errors": tool_errors, "skipped": skipped, "errors": errors,
            "edit_tools": [t for t in tools if t not in ("inspect_photo", "sample_color", "undo", "revert_to")],
            "versions": [{"index": v.index, "tool": v.tool, "summary": v.summary, "cost": v.cost_usd, "url": store.rel_url(v.path)} for v in photo.versions],
            "head_index": head.index, "edited": head.index > 0, "reply": final_text,
            "cost_usd": round(shoot.total_cost_usd, 4), "seconds": round(time.time() - t0, 1),
        })
        exp = EXPECTED.get(res["category"], set())
        res["tool_choice_ok"] = (not exp) or any(t in exp for t in res["edit_tools"])
        if head.index > 0:
            res["changed_fraction"] = round(changed_fraction(photo.original_path, head.path), 4)
            res["result_path"] = head.path
            if do_judge:
                try:
                    res["judge"] = await judge(rec["request"], photo.original_path, head.path)
                except Exception as e:  # noqa: BLE001
                    res["judge"] = {"verdict": "error", "notes": str(e)[:200]}
        elif do_judge:
            res["judge"] = {"verdict": "fail", "fulfilled": 0, "collateral": 0, "artifacts": 0, "notes": "no edit was produced"}
    except Exception as e:  # noqa: BLE001
        res["exception"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-1500:]
        res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


# ---------------- summary ----------------
def summarize(run_dir: Path) -> dict:
    recs = [json.loads(p.read_text()) for p in sorted((run_dir / "records").glob("*.json"), key=lambda p: int(p.stem))]
    n = len(recs)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cat[r["category"]].append(r)

    def rate(xs, f):
        xs = list(xs)
        return round(100 * sum(1 for x in xs if f(x)) / len(xs), 1) if xs else None

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 2) if xs else None

    judged = [r for r in recs if isinstance(r.get("judge"), dict) and r["judge"].get("verdict") in ("pass", "partial", "fail")]
    summary = {
        "run": run_dir.name, "records": n,
        "exceptions": sum(1 for r in recs if r.get("exception")),
        "edited_pct": rate(recs, lambda r: r.get("edited")),
        "tool_choice_ok_pct": rate(recs, lambda r: r.get("tool_choice_ok")),
        "planner_errors": sum(1 for r in recs if r.get("errors")),
        "tool_errors": sum(1 for r in recs if r.get("tool_errors")),
        "mean_seconds": mean(r.get("seconds") for r in recs),
        "total_cost_usd": round(sum(r.get("cost_usd", 0) or 0 for r in recs), 2),
        "tool_usage": dict(Counter(t for r in recs for t in r.get("tools", []))),
        "judged": len(judged),
        "judge_pass_pct": rate(judged, lambda r: r["judge"]["verdict"] == "pass"),
        "judge_partial_pct": rate(judged, lambda r: r["judge"]["verdict"] == "partial"),
        "judge_fail_pct": rate(judged, lambda r: r["judge"]["verdict"] == "fail"),
        "judge_mean_fulfilled": mean(r["judge"].get("fulfilled") for r in judged),
        "judge_mean_collateral": mean(r["judge"].get("collateral") for r in judged),
        "judge_mean_artifacts": mean(r["judge"].get("artifacts") for r in judged),
        "mean_changed_fraction": mean(r.get("changed_fraction") for r in recs),
        "by_category": {},
    }
    for cat, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        js = [r for r in rs if isinstance(r.get("judge"), dict) and r["judge"].get("verdict") in ("pass", "partial", "fail")]
        summary["by_category"][cat] = {
            "n": len(rs), "edited_pct": rate(rs, lambda r: r.get("edited")), "tool_choice_ok_pct": rate(rs, lambda r: r.get("tool_choice_ok")),
            "pass_pct": rate(js, lambda r: r["judge"]["verdict"] == "pass"), "fail_pct": rate(js, lambda r: r["judge"]["verdict"] == "fail"),
            "fulfilled": mean(r["judge"].get("fulfilled") for r in js), "collateral": mean(r["judge"].get("collateral") for r in js),
            "artifacts": mean(r["judge"].get("artifacts") for r in js), "mean_seconds": mean(r.get("seconds") for r in rs),
            "tools": dict(Counter(t for r in rs for t in r.get("edit_tools", []))),
        }
    summary["failures"] = [
        {"idx": r["idx"], "category": r["category"], "request": r["request"][:90], "tools": r.get("edit_tools"),
         "verdict": (r.get("judge") or {}).get("verdict"), "fulfilled": (r.get("judge") or {}).get("fulfilled"),
         "collateral": (r.get("judge") or {}).get("collateral"), "artifacts": (r.get("judge") or {}).get("artifacts"),
         "notes": (r.get("judge") or {}).get("notes") or r.get("exception") or (r.get("errors") or [None])[0] or (r.get("tool_errors") or [None])[0]}
        for r in recs if r.get("exception") or not r.get("edited") or (r.get("judge") or {}).get("verdict") in ("fail", "partial")
    ]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = [f"# Eval run `{run_dir.name}` — {n} records", ""]
    for k, v in summary.items():
        if k in ("by_category", "failures", "tool_usage"):
            continue
        lines.append(f"- **{k}**: {v}")
    lines.append(f"- **tool_usage**: {summary['tool_usage']}")
    lines += ["", "| category | n | edited% | tool ok% | pass% | fail% | fulfilled | collateral | artifacts | s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cat, c in summary["by_category"].items():
        lines.append(f"| {cat} | {c['n']} | {c['edited_pct']} | {c['tool_choice_ok_pct']} | {c['pass_pct']} | {c['fail_pct']} | {c['fulfilled']} | {c['collateral']} | {c['artifacts']} | {c['mean_seconds']} |")
    lines += ["", "## Not passed", ""]
    for f in summary["failures"]:
        lines.append(f"- #{f['idx']} [{f['category']}] {f['verdict']} F{f['fulfilled']}/C{f['collateral']}/A{f['artifacts']} {f['tools']} — {f['request']} — {f['notes']}")
    (run_dir / "summary.md").write_text("\n".join(lines))
    return summary


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated 0-based record indices")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run records that already have results")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--max-cost", type=float, default=25.0, help="stop launching new records past this estimated spend")
    args = ap.parse_args()

    run_dir = EVAL_DIR / args.run
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        s = summarize(run_dir)
        print(json.dumps({k: v for k, v in s.items() if k not in ("by_category", "failures")}, indent=1))
        return

    recs = _records()
    idxs = [int(x) for x in args.only.split(",") if x.strip()] if args.only else list(range(len(recs)))
    if args.limit:
        idxs = idxs[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    spent = 0.0
    done = 0
    t_start = time.time()

    async def one(i: int) -> None:
        nonlocal spent, done
        out = run_dir / "records" / f"{i}.json"
        if out.exists() and not args.force:
            return
        async with sem:
            if spent > args.max_cost:
                print(f"[{i}] skipped: cost cap {args.max_cost} reached", flush=True)
                return
            r = await run_record(i, recs[i], out, args.judge)
            spent += r.get("cost_usd", 0) or 0
            done += 1
            j = r.get("judge") or {}
            print(f"[{i:3d}] {r['category']:14s} {('EXC ' + r['exception'][:60]) if r.get('exception') else ''}"
                  f"edited={r.get('edited')} tools={r.get('edit_tools')} {r.get('seconds')}s ${r.get('cost_usd', 0)} "
                  f"judge={j.get('verdict')} F{j.get('fulfilled')}/C{j.get('collateral')}/A{j.get('artifacts')} | {j.get('notes', '')[:90]}",
                  flush=True)

    await asyncio.gather(*(one(i) for i in idxs))
    print(f"\nfinished {done} records in {(time.time() - t_start) / 60:.1f} min, est. spend ${spent:.2f}", flush=True)
    s = summarize(run_dir)
    print(json.dumps({k: v for k, v in s.items() if k not in ("by_category", "failures")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
