"""Tool coverage suite: one request per tool, through the production planner, judged like the dataset eval.

Builds small shoots from cached dataset records (no downloads after the first demo run), sends a curated request per
tool with the exact UI payload (scope, selected photo, points), and checks for each case:
  * the planner chose one of the expected tools,
  * the expected photos got a new version (and which others did),
  * the vision judge's verdict on every edited photo (with --judge),
plus direct checks of the deterministic colour tool (eyedropper + apply) that bypass the planner.
Groups run on separate shoots in parallel (the per-shoot lock serialises cases inside a group).

    cd backend && uv run python tests/tool_coverage.py --run tools1 --judge
    uv run python tests/tool_coverage.py --run tools1 --only A,C,Q      # a few cases
Results: data/eval/tools/<run>/results.json and summary.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import store  # noqa: E402
from app.agent.loop import run_turn  # noqa: E402
from app.config import settings  # noqa: E402
from app.demo_shoots import _fetch, _records  # noqa: E402
from app.models import Photo, Shoot, Version  # noqa: E402
from app.tools import composite, imaging, pick  # noqa: E402
from eval_dataset import changed_fraction, judge  # noqa: E402

OUT_DIR = settings.data_dir / "eval" / "tools"

# dataset record numbers (1-based) per group; photo index inside the group's shoot = position in the list
GROUPS: dict[str, list[int]] = {
    "det": [182, 137, 7, 16, 151],          # adjust/sharpen, lot lines, blur, match look, crop, undo, watermark
    "colour": [22, 157],                     # sample + repaint from reference, repaint hex
    "remove": [137, 7, 198],                 # bins, cars, lockbox
    "gen": [151, 101, 120],                  # lamps (scope shoot), tv_screen
    "direct": [22, 157],                     # colour tool without the planner
}

# id, group, text, scope ('photo' = only the given photo index; 'shoot' = planner decides), photo index, points, expected tools, expected edited photo indices
CASES: list[dict] = [
    {"id": "A", "group": "det", "text": "Brighten this photo by 15%.", "scope": "photo", "photo": 4, "tools": {"adjust"}, "edited": {4}},
    {"id": "B", "group": "det", "text": "Please sharpen this photo, it looks blurry.", "scope": "photo", "photo": 0, "tools": {"adjust"}, "edited": {0}},
    {"id": "J", "group": "det", "text": "Draw the lot lines through the points I placed.", "scope": "photo", "photo": 1,
     "points": [[0.235, 0.14], [0.62, 0.14], [0.62, 0.93], [0.21, 0.93]], "tools": {"draw_lot_lines"}, "edited": {1}, "judge": False},
    {"id": "L", "group": "det", "text": "Blur the licence plates of the cars in this photo.", "scope": "photo", "photo": 2, "tools": {"blur"}, "edited": {2}, "judge": False, "optional_edit": True},
    {"id": "M", "group": "det", "text": "Make Photo 4 match the look and colour balance of Photo 3.", "scope": "shoot", "photo": 3, "tools": {"match_look"}, "edited": {3}, "judge": False},
    {"id": "N", "group": "det", "text": "Straighten this photo: rotate it about 1.5 degrees counter-clockwise.", "scope": "photo", "photo": 3, "tools": {"crop_rotate"}, "edited": {3}, "judge": False},
    {"id": "O", "group": "det", "text": "Undo that last change.", "scope": "photo", "photo": 3, "tools": {"undo", "revert_to"}, "edited": set(), "judge": False, "expect_head_drop": 3},
    {"id": "K", "group": "det", "text": "Add a small 'Digitally altered' disclosure watermark to every photo.", "scope": "shoot", "photo": None, "tools": {"add_watermark"}, "edited": {0, 1, 2, 3, 4}, "judge": False},
    {"id": "C", "group": "colour", "text": "Change the grey walls to the colour of the walls in the reference image.", "scope": "photo", "photo": 0, "tools": {"repaint"}, "also": {"sample_color", "inspect_photo"}, "edited": {0}},
    {"id": "D", "group": "colour", "text": "Please use #C8C2B7 color for the floors.", "scope": "photo", "photo": 1, "tools": {"repaint"}, "edited": {1}},
    {"id": "E", "group": "remove", "text": "Remove the garbage bins.", "scope": "photo", "photo": 0, "tools": {"remove_object"}, "edited": {0}},
    {"id": "F", "group": "remove", "text": "Remove all the cars.", "scope": "photo", "photo": 1, "tools": {"remove_object"}, "edited": {1}},
    {"id": "G", "group": "remove", "text": "Remove the lockbox hanging from the door handle. Do not change how the handle looks.", "scope": "photo", "photo": 2, "tools": {"remove_object", "generative_edit"}, "edited": {2}},
    {"id": "H", "group": "gen", "text": "Turn on the lamps in the photos that have lamps.", "scope": "shoot", "photo": None, "tools": {"generative_edit"}, "edited": {0}},
    {"id": "I", "group": "gen", "text": "Put the reference image on the TV screen, properly fitted.", "scope": "photo", "photo": 1, "tools": {"tv_screen"}, "edited": {1}},
]


async def build_shoot(name: str, nums: list[int]) -> Shoot:
    recs = _records()
    shoot = Shoot(name=f"eval tools: {name}")
    for i, n in enumerate(nums):
        r = recs[n - 1]
        edited = await _fetch(r["edited_photo"])
        brackets = await asyncio.gather(*(_fetch(u) for u in r["inputs"]))
        refs = await asyncio.gather(*(_fetch(u) for u in r.get("request_references", [])))
        with Image.open(edited) as im:
            w, h = im.size
        photo = Photo(shoot_id=shoot.id, name=f"Photo {i + 1}", width=w, height=h, original_path=str(edited),
                      bracket_paths=[str(b) for b in brackets], reference_paths=[str(x) for x in refs], dataset_request=r["request"].strip())
        pdir = store.photo_dir(photo)
        v0 = pdir / "v00_original.jpg"
        v0.write_bytes(edited.read_bytes())
        ver = Version(photo_id=photo.id, index=0, path=str(v0))
        photo.versions = [ver]
        photo.head_version_id = ver.id
        shoot.photos.append(photo)
    store.save(shoot)
    return shoot


async def run_case(shoot_id: str, case: dict, do_judge: bool) -> dict:
    shoot = store.get(shoot_id)
    assert shoot is not None
    before = {p.id: p.head for p in shoot.photos}
    sel = shoot.photos[case["photo"]] if case.get("photo") is not None else None
    ui = {"selected_photo_id": sel.id if sel else None, "points": case.get("points"), "scope": case["scope"], "confirm_first": False}
    t0 = time.time()
    cost0 = shoot.total_cost_usd
    events, reply = [], ""
    async for ev in run_turn(shoot, case["text"], ui):
        if ev["type"] in ("tool_start", "tool_result", "error"):
            events.append(ev)
        elif ev["type"] == "done":
            reply = ev["data"].get("text", "")
    shoot = store.get(shoot_id) or shoot
    tools = [e["data"]["name"] for e in events if e["type"] == "tool_start"]
    edited_idx = {i for i, p in enumerate(shoot.photos) if p.head.id != before[p.id].id}
    res = {"id": case["id"], "group": case["group"], "request": case["text"], "scope": case["scope"], "tools": tools,
           "tool_ok": any(t in case["tools"] for t in tools),
           "edited": sorted(edited_idx), "expected": sorted(case["edited"]),
           "expected_missed": sorted(case["edited"] - edited_idx), "unexpected": sorted(edited_idx - case["edited"]),
           "errors": [e["data"].get("message") for e in events if e["type"] == "error"],
           "tool_errors": [e["data"].get("error") for e in events if e["type"] == "tool_result" and e["data"].get("error")],
           "reply": reply[:400], "seconds": round(time.time() - t0, 1), "cost_usd": round(shoot.total_cost_usd - cost0, 4), "photos": []}
    if case.get("expect_head_drop") is not None:
        p = shoot.photos[case["expect_head_drop"]]
        res["head_dropped"] = p.head.index < before[p.id].index
    for i in sorted(edited_idx):
        p = shoot.photos[i]
        b = before[p.id]
        entry = {"photo": p.name, "tool": p.head.tool, "summary": p.head.summary[:120], "changed_fraction": round(changed_fraction(b.path, p.head.path), 4),
                 "qc": (p.head.params or {}).get("qc")}
        if do_judge and case.get("judge", True) and p.head.index > b.index:
            try:
                entry["judge"] = await judge(case["text"], b.path, p.head.path)
            except Exception as e:  # noqa: BLE001
                entry["judge"] = {"verdict": "error", "notes": str(e)[:200]}
        res["photos"].append(entry)
    # verdict for the case
    ok = res["tool_ok"] and not res["expected_missed"] and not res["errors"] and not res["tool_errors"]
    if case.get("optional_edit") and res["expected_missed"] and res["tool_ok"]:
        ok = True   # detection may legitimately find nothing (e.g. no readable plates)
    if case.get("expect_head_drop") is not None:
        ok = ok and bool(res.get("head_dropped"))
    judged = [p.get("judge", {}).get("verdict") for p in res["photos"] if p.get("judge")]
    if judged:
        ok = ok and all(v == "pass" for v in judged)
        res["judge_verdicts"] = judged
    res["ok"] = ok
    return res


async def run_direct(shoot_id: str, do_judge: bool) -> list[dict]:
    """Colour tool without the planner: pick on the reference, apply to the grey walls; apply a hex to the tile floor."""
    out = []
    shoot = store.get(shoot_id)
    assert shoot is not None
    p0, p1 = shoot.photos[0], shoot.photos[1]
    t0 = time.time()
    ref = imaging.load_rgb(p0.reference_paths[0])
    picked = pick.pick_color(ref, 0.30, 0.45)
    rgb = imaging.load_rgb(p0.head.path)
    r1, m1 = pick.region_for_apply(rgb, 0.08, 0.62)
    r2, m2 = pick.region_for_apply(rgb, 0.95, 0.45)
    reg = np.maximum(r1, r2)
    out_rgb = pick.apply_color(rgb, reg, picked["hex"])
    outside = composite.feather(reg, int(0.004 * max(rgb.shape[:2])) + 2) < 0.001
    leak = float(np.abs(out_rgb - rgb)[outside].max())
    pdir = store.photo_dir(p0)
    path = imaging.save_rgb(out_rgb, pdir / "v01_recolor.jpg")
    mask_path = imaging.save_mask(reg, pdir / "v01_mask.png")
    store.add_version(shoot, p0, path, "recolor", {"hex": picked["hex"], "mode": m1}, f"Recoloured walls to {picked['hex']}", mask_path=mask_path)
    case = {"id": "Q1", "group": "direct", "request": f"[colour tool] pick the teal wall in the reference ({picked['hex']}), apply to both grey walls",
            "picked": {k: picked[k] for k in ("hex", "hex_at_click", "area_pct", "region_mode")}, "region_pct": round(float(reg.mean()) * 100, 1),
            "modes": [m1, m2], "outside_mask_max_diff": round(leak, 5), "seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
    teal_ok = picked["hex"].lower() not in ("#000000", "#ffffff")
    lab = pick.color.rgb_to_lab(pick.color.hex_to_rgb01(picked["hex"]).reshape(1, 1, 3))[0, 0]
    teal_ok = teal_ok and lab[1] < 0 and lab[2] < 0        # teal: green-ish and blue-ish in LAB
    case["picked_is_teal"] = bool(teal_ok)
    case["ok"] = teal_ok and leak < 1e-4 and 8 < case["region_pct"] < 30
    if do_judge:
        case["judge"] = await judge(f"Repaint the two dark grey accent walls to teal {picked['hex']} keeping their texture and shading; everything else (TV, bed, ceiling, white walls, floor) must stay identical.", p0.original_path, str(path))
        case["ok"] = case["ok"] and case["judge"].get("verdict") == "pass"
    out.append(case)
    t0 = time.time()
    rgb = imaging.load_rgb(p1.head.path)
    reg, m = pick.region_for_apply(rgb, 0.55, 0.85)
    out_rgb = pick.apply_color(rgb, reg, "#C8C2B7")
    outside = composite.feather(reg, int(0.004 * max(rgb.shape[:2])) + 2) < 0.001
    leak = float(np.abs(out_rgb - rgb)[outside].max())
    pdir = store.photo_dir(p1)
    path = imaging.save_rgb(out_rgb, pdir / "v01_recolor.jpg")
    store.add_version(shoot, p1, path, "recolor", {"hex": "#C8C2B7", "mode": m}, "Recoloured floor to #C8C2B7", mask_path=imaging.save_mask(reg, pdir / "v01_mask.png"))
    case = {"id": "Q2", "group": "direct", "request": "[colour tool] apply #C8C2B7 to the tile floor with one click", "region_pct": round(float(reg.mean()) * 100, 1),
            "modes": [m], "outside_mask_max_diff": round(leak, 5), "seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
    case["ok"] = leak < 1e-4 and 5 < case["region_pct"] < 25
    if do_judge:
        case["judge"] = await judge("Make the tile floor colour #C8C2B7 (a warm light grey-beige), keeping the tiles and grout; walls, ceiling, stairs and door must stay identical.", p1.original_path, str(path))
        case["ok"] = case["ok"] and case["judge"].get("verdict") == "pass"
    out.append(case)
    return out


def summarize(run_dir: Path, results: list[dict], shoots: dict[str, str]) -> str:
    ok = sum(1 for r in results if r.get("ok"))
    lines = [f"# Tool coverage run `{run_dir.name}`", "", f"{ok}/{len(results)} cases passed · ${sum(r.get('cost_usd', 0) for r in results):.2f} · "
             f"{sum(r.get('seconds', 0) for r in results) / 60:.1f} min of work (groups ran in parallel)", "",
             "| case | request | tools called | edited (expected) | judge | ok | s | $ |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["group"] == "direct":
            j = (r.get("judge") or {})
            jd = f"{j.get('verdict', '-')} F{j.get('fulfilled')}/C{j.get('collateral')}/A{j.get('artifacts')}" if j else "-"
            lines.append(f"| {r['id']} | {r['request'][:60]} | eyedropper + apply ({', '.join(r['modes'])}) | region {r['region_pct']}% · leak {r['outside_mask_max_diff']} | {jd} | {'PASS' if r['ok'] else 'FAIL'} | {r['seconds']} | 0 |")
            continue
        jd = ", ".join(f"{p['photo']}: {p['judge']['verdict']} F{p['judge'].get('fulfilled')}/C{p['judge'].get('collateral')}/A{p['judge'].get('artifacts')}" for p in r["photos"] if p.get("judge")) or "-"
        lines.append(f"| {r['id']} | {r['request'][:60]} | {', '.join(r['tools']) or '—'} | {r['edited']} ({r['expected']}) | {jd} | {'PASS' if r['ok'] else 'FAIL'} | {r['seconds']} | {r['cost_usd']:.2f} |")
    lines += ["", "## Notes", ""]
    for r in results:
        if r["group"] == "direct":
            lines.append(f"- **{r['id']}** picked={r.get('picked')} region={r['region_pct']}% modes={r['modes']} outside-mask diff={r['outside_mask_max_diff']} judge={(r.get('judge') or {}).get('notes', '')}")
            continue
        lines.append(f"- **{r['id']}** [{r['group']}] tools={r['tools']} edited={r['edited']} missed={r['expected_missed']} extra={r['unexpected']} errors={r['errors'] + r['tool_errors']}")
        lines.append(f"  reply: {r['reply'][:220].replace(chr(10), ' ')}")
        for p in r["photos"]:
            j = p.get("judge") or {}
            lines.append(f"  - {p['photo']}: {p['tool']} · {p['summary'][:80]} · changed {p['changed_fraction'] * 100:.1f}% · QC {((p.get('qc') or {}).get('verdict') or '-')} · judge {j.get('verdict', '-')} {j.get('notes', '')}")
    lines += ["", "Shoots: " + ", ".join(f"{g}={sid}" for g, sid in shoots.items())]
    md = "\n".join(lines)
    (run_dir / "summary.md").write_text(md)
    return md


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tools")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated case ids (A,B,...,Q)")
    a = ap.parse_args()
    run_dir = OUT_DIR / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    only = {x.strip().upper() for x in a.only.split(",") if x.strip()}
    cases = [c for c in CASES if not only or c["id"] in only]
    do_direct = not only or any(x.startswith("Q") for x in only)
    groups = {g for g in (c["group"] for c in cases)} | ({"direct"} if do_direct else set())
    shoots = {g: (await build_shoot(f"{a.run} {g}", GROUPS[g])).id for g in groups}
    print("shoots:", shoots, flush=True)
    results: list[dict] = []
    lock = asyncio.Lock()

    async def run_group(g: str) -> None:
        for c in [c for c in cases if c["group"] == g]:
            print(f"[{c['id']}] {c['text']}", flush=True)
            try:
                r = await run_case(shoots[g], c, a.judge)
            except Exception as e:  # noqa: BLE001
                r = {"id": c["id"], "group": g, "request": c["text"], "tools": [], "edited": [], "expected": sorted(c["edited"]), "expected_missed": sorted(c["edited"]),
                     "unexpected": [], "errors": [f"{type(e).__name__}: {e}"], "tool_errors": [], "reply": "", "photos": [], "seconds": 0, "cost_usd": 0, "ok": False, "tool_ok": False}
            async with lock:
                results.append(r)
                (run_dir / "results.json").write_text(json.dumps({"shoots": shoots, "results": results}, indent=1, ensure_ascii=False))
            print(f"   -> {'PASS' if r['ok'] else 'FAIL'} tools={r['tools']} edited={r['edited']} judge={r.get('judge_verdicts')} {r['seconds']}s ${r['cost_usd']:.2f}", flush=True)

    async def run_direct_group() -> None:
        rs = await run_direct(shoots["direct"], a.judge)
        async with lock:
            results.extend(rs)
            (run_dir / "results.json").write_text(json.dumps({"shoots": shoots, "results": results}, indent=1, ensure_ascii=False))
        for r in rs:
            print(f"[{r['id']}] {r['request']}\n   -> {'PASS' if r['ok'] else 'FAIL'} region={r['region_pct']}% judge={(r.get('judge') or {}).get('verdict')}", flush=True)

    tasks = [run_group(g) for g in groups if g != "direct"]
    if do_direct:
        tasks.append(run_direct_group())
    await asyncio.gather(*tasks)
    order = {c["id"]: i for i, c in enumerate(CASES)}
    results.sort(key=lambda r: order.get(r["id"], 99))
    print("\n" + summarize(run_dir, results, shoots))


if __name__ == "__main__":
    asyncio.run(main())
