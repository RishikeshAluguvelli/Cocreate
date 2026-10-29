"""Smoke test on photos that are NOT in the AutoHDR dataset.

Builds one multi-photo shoot from a folder of images, sends a series of whole-shoot requests through the production
planner (exactly what the UI does), and checks two things per request:
  * which photos were edited vs. which ones we expected (the "apply the right requests, skip the rest" rubric), and
  * the quality of every edited photo with the same strict vision judge used by tests/eval_dataset.py.

The shoot is saved like any other, so it shows up on the home page and every version can be inspected in the UI.

    cd backend && uv run python tests/smoke_new_photos.py --photos /path/to/folder --judge --run unsplash1
    # custom requests: --requests requests.json  -> [{"text": "...", "expect": ["photo-name-stem", ...]}, ...]

Results: data/eval/newphotos/<run>/results.json and summary.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import store  # noqa: E402
from app.agent.loop import run_turn  # noqa: E402
from app.config import settings  # noqa: E402
from app.demo_shoots import create_shoot_from_files  # noqa: E402
from eval_dataset import changed_fraction, judge  # noqa: E402

# Default requests, written for the ten Unsplash photos used on 2026-10-29 (stems are the file names without extension).
# `expect` is the minimum set of photos that must change; photos edited beyond it are reported, not penalised.
DEFAULT_REQUESTS = [
    {"text": "Remove the dog from the living room photo.", "expect": ["02_living_dog"]},
    {"text": "Remove the people from the kitchen.", "expect": ["01_kitchen_people"]},
    {"text": "Remove the construction crane in the background of the pool photo.", "expect": ["03_pool_crane"]},
    {"text": "Turn on all the lamps and the fireplace in the interior photos.", "expect": ["04_living_lamp_tv", "05_bedroom", "06_living_fireplace"]},
    {"text": "Make the sky a vivid blue with a few light clouds on any exterior photo where it looks pale or washed out.", "expect": ["07_house_lawn_sky"]},
    {"text": "Repaint the beige walls in the small apartment living room to a light grey, #D8DBDF.", "expect": ["08_apartment_beige"]},
    {"text": "Brighten the dusk exterior photo by about 15%.", "expect": ["09_house_dusk"]},
    {"text": "Make the bed look neatly made with the bedspread smooth and ironed.", "expect": ["05_bedroom"]},
    {"text": "Add a small 'Digitally altered' disclosure watermark to every photo.", "expect": "all"},
]


def load_photos(folder: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not files:
        raise SystemExit(f"no images in {folder}")
    return [(p.name, p.read_bytes()) for p in files]


async def run_request(shoot_id: str, req: dict, do_judge: bool) -> dict:
    shoot = store.get(shoot_id)
    assert shoot is not None
    before = {p.id: p.head for p in shoot.photos}
    cost0 = shoot.total_cost_usd
    t0 = time.time()
    events: list[dict] = []
    reply = ""
    async for ev in run_turn(shoot, req["text"], {"selected_photo_id": None, "points": None, "scope": "shoot", "confirm_first": False}):
        if ev["type"] in ("tool_start", "tool_result", "skipped", "error", "version"):
            events.append(ev)
        elif ev["type"] == "done":
            reply = ev["data"].get("text", "")
    shoot = store.get(shoot_id) or shoot
    expect = req.get("expect", [])
    expected = {p.name for p in shoot.photos} if expect == "all" else set(expect)
    edited, per_photo = set(), []
    for p in shoot.photos:
        b = before[p.id]
        if p.head.id == b.id:
            continue
        edited.add(p.name)
        entry = {"photo": p.name, "tools": [v.tool for v in p.versions if v.index > b.index],
                 "summary": p.head.summary, "changed_fraction": round(changed_fraction(b.path, p.head.path), 4),
                 "qc": (p.head.params or {}).get("qc")}
        if do_judge:
            try:
                entry["judge"] = await judge(req["text"], b.path, p.head.path)
            except Exception as e:  # noqa: BLE001
                entry["judge"] = {"verdict": "error", "notes": str(e)[:200]}
        per_photo.append(entry)
    return {
        "request": req["text"], "expected": sorted(expected), "edited": sorted(edited),
        "expected_missed": sorted(expected - edited), "unexpected_edited": sorted(edited - expected),
        "tools": [e["data"]["name"] for e in events if e["type"] == "tool_start"],
        "skipped": [e["data"] for e in events if e["type"] == "skipped"],
        "errors": [e["data"].get("message") for e in events if e["type"] == "error"],
        "tool_errors": [e["data"].get("error") for e in events if e["type"] == "tool_result" and e["data"].get("error")],
        "reply": reply, "photos": per_photo,
        "cost_usd": round(shoot.total_cost_usd - cost0, 4), "seconds": round(time.time() - t0, 1),
    }


def summarize(run_dir: Path, results: list[dict], shoot_id: str, shoot_name: str) -> str:
    lines = [f"# New-photo smoke test · {shoot_name}", "", f"Shoot `{shoot_id}` · {len(results)} requests · "
             f"${sum(r['cost_usd'] for r in results):.2f} · {sum(r['seconds'] for r in results) / 60:.1f} min", ""]
    verdicts = [p.get("judge", {}).get("verdict") for r in results for p in r["photos"]]
    n = len(verdicts) or 1
    lines += [f"Edited photos judged: {len(verdicts)} · pass {verdicts.count('pass')} · partial {verdicts.count('partial')} · fail {verdicts.count('fail')}",
              f"Photo selection: {sum(1 for r in results if not r['expected_missed'] and not r['unexpected_edited'])}/{len(results)} requests "
              f"edited exactly the expected photos; {sum(len(r['expected_missed']) for r in results)} expected photos missed, "
              f"{sum(len(r['unexpected_edited']) for r in results)} unexpected photos edited.", ""]
    lines += ["| # | request | expected | edited | missed | extra | judge (per edited photo) | $ | s |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        j = ", ".join(f"{p['photo'].split('_', 1)[-1]}: {p.get('judge', {}).get('verdict', '-')}"
                      f"{' F' + str(p['judge']['fulfilled']) + '/C' + str(p['judge']['collateral']) + '/A' + str(p['judge']['artifacts']) if p.get('judge', {}).get('fulfilled') is not None else ''}"
                      for p in r["photos"]) or "—"
        exp = "all" if len(r["expected"]) > 5 else ", ".join(x.split('_', 1)[-1] for x in r["expected"])
        lines.append(f"| {i} | {r['request'][:70]} | {exp} | {len(r['edited'])} | {len(r['expected_missed'])} | {len(r['unexpected_edited'])} | {j} | {r['cost_usd']:.2f} | {r['seconds']:.0f} |")
    lines += ["", "## Notes per request", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"**{i}. {r['request']}**  ")
        lines.append(f"tools: {', '.join(r['tools']) or '—'} · reply: {r['reply'][:300].replace(chr(10), ' ')}  ")
        for p in r["photos"]:
            jd = p.get("judge", {})
            lines.append(f"- {p['photo']}: {p['summary'][:90]} · changed {p['changed_fraction'] * 100:.1f}% · QC {((p.get('qc') or {}).get('verdict') or '-')} · judge {jd.get('verdict', '-')} — {jd.get('notes', '')}")
        if r["expected_missed"]:
            lines.append(f"- MISSED: {', '.join(r['expected_missed'])}")
        if r["unexpected_edited"]:
            lines.append(f"- EXTRA: {', '.join(r['unexpected_edited'])}")
        if r["errors"] or r["tool_errors"]:
            lines.append(f"- errors: {r['errors'] + r['tool_errors']}")
        lines.append("")
    md = "\n".join(lines)
    (run_dir / "summary.md").write_text(md)
    return md


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, type=Path)
    ap.add_argument("--run", default="newphotos")
    ap.add_argument("--shoot-name", default=None)
    ap.add_argument("--requests", type=Path, default=None)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--max-cost", type=float, default=6.0)
    ap.add_argument("--only", type=str, default=None, help="comma-separated request numbers (1-based)")
    a = ap.parse_args()

    reqs = json.loads(a.requests.read_text()) if a.requests else DEFAULT_REQUESTS
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        reqs = [r for i, r in enumerate(reqs, 1) if i in keep]
    run_dir = settings.data_dir / "eval" / "newphotos" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    files = load_photos(a.photos)
    shoot = create_shoot_from_files(a.shoot_name or f"New photos test · {a.photos.name}", files)
    print(f"shoot {shoot.id} '{shoot.name}' with {len(shoot.photos)} photos: {[p.name for p in shoot.photos]}")
    results: list[dict] = []
    for i, req in enumerate(reqs, 1):
        print(f"\n[{i}/{len(reqs)}] {req['text']}")
        r = await run_request(shoot.id, req, a.judge)
        results.append(r)
        print(f"   edited={r['edited']} missed={r['expected_missed']} extra={r['unexpected_edited']} tools={r['tools']} ${r['cost_usd']:.2f} {r['seconds']:.0f}s")
        for p in r["photos"]:
            print(f"   - {p['photo']}: judge={p.get('judge', {}).get('verdict')} {p.get('judge', {}).get('notes', '')[:120]}")
        (run_dir / "results.json").write_text(json.dumps({"shoot_id": shoot.id, "shoot_name": shoot.name, "results": results}, indent=1, ensure_ascii=False))
        total = (store.get(shoot.id) or shoot).total_cost_usd
        if total > a.max_cost:
            print(f"stopping: shoot cost ${total:.2f} exceeded --max-cost {a.max_cost}")
            break
    print("\n" + summarize(run_dir, results, shoot.id, shoot.name))


if __name__ == "__main__":
    asyncio.run(main())
