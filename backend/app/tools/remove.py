"""Remove an object with a precise mask: SAM-3 mask -> eraser (LaMa) or generative fill (FLUX Fill) -> composite."""
from __future__ import annotations

import asyncio
from pathlib import Path

from ..config import settings
from ..models import Photo, Shoot
from .. import store
from . import composite, editors, imaging, masks
from .qc import qc_check, qc_score


def _qc_score(q: dict) -> tuple:
    return qc_score(q)


async def remove_object(shoot: Shoot, photo: Photo, target: str, method: str = "auto",
                        points: list[list[float]] | None = None, box: list[float] | None = None,
                        fill_prompt: str | None = None, dilate_pct: float | None = None, progress=None, qc: bool = True,
                        user_mask=None) -> dict:
    head = photo.head
    rgb = imaging.load_rgb(head.path)
    H, W = rgb.shape[:2]
    if progress:
        await progress(f"finding {target} in {photo.name}")
    em = await masks.text_mask(head.path, (W, H), target, points=points, box=box, dilate_pct=dilate_pct)
    if user_mask is not None:   # the region the user painted is removed too (or alone, when detection finds nothing)
        em = masks.merge_user_mask(em, user_mask, (W, H), dilate_pct=dilate_pct)
    if em is None:
        return {"found": False, "photo_id": photo.id, "target": target,
                "message": f"No '{target}' detected in {photo.name}; left unchanged."}

    big = em.area > 0.006 and settings.image_backend != "fal"
    large = settings.image_backend == "gemini" and (method or "auto") == "auto" and em.area > settings.large_mask_area
    if progress:
        await progress(f"{photo.name}: mask covers {em.area*100:.1f}% · " + (
            "large area: two fill strategies in parallel (~40 s), QC keeps the better" if large else
            "filling with Nano Banana (~20 s)" if big else "inpainting"))
    strategies = ["semantic", "repair"] if large else ["repair"]
    results = await asyncio.gather(*(editors.fill(rgb, em.hard, target, method or "auto", em.area, fill_prompt, strategy=st)
                                     for st in strategies))
    cost = sum(float(r[2]) for r in results)
    cands = [{"strategy": st, "edited": e, "how": how} for st, (e, how, _c, changed) in zip(strategies, results) if changed]
    if not cands:
        shoot.total_cost_usd += cost
        store.save(shoot)
        return {"found": True, "changed": False, "photo_id": photo.id, "target": target, "method": results[0][1],
                "area_pct": round(em.area * 100, 2), "instances": em.instances, "cost_usd": cost,
                "message": f"Found '{target}' in {photo.name} but the fill model returned the photo unchanged, so no version was "
                           f"created. Options: method='erase' (works for thin objects), a fill_prompt describing the background, "
                           f"or a box/points to narrow the target."}
    pdir = store.photo_dir(photo)
    idx = len(photo.versions)
    verdict: dict | None = None
    for c in cands:
        c["final"] = await asyncio.to_thread(composite.composite, rgb, c["edited"], em.soft)
    if len(cands) > 1 and qc:
        # both candidates are composited and judged; the better one becomes the version, the other is discarded
        for c in cands:
            c["path"] = imaging.save_rgb(c["final"], pdir / f"v{idx:02d}_cand_{c['strategy']}.jpg")
        if progress:
            await progress(f"QC comparing {len(cands)} fills for {photo.name}")
        qcs = await asyncio.gather(*(qc_check(f"Remove the {target}; everything else must stay identical.", head.path, str(c["path"]))
                                     for c in cands))
        for c, q in zip(cands, qcs):
            c["qc"] = q
            cost += q.get("cost_usd", 0)
        cands.sort(key=lambda c: _qc_score(c["qc"]), reverse=True)
        best, losers = cands[0], cands[1:]
        for lo in losers:
            Path(lo["path"]).unlink(missing_ok=True)
        verdict = best["qc"]
        how = f"{best['how']} · QC preferred it over the {losers[0]['strategy']} fill"
        final = best["final"]
        out_path = pdir / f"v{idx:02d}_remove.jpg"
        Path(best["path"]).rename(out_path)
    else:
        best = cands[0]
        how, final = best["how"], best["final"]
        out_path = imaging.save_rgb(final, pdir / f"v{idx:02d}_remove.jpg")
    mask_path = imaging.save_mask(em.hard, pdir / f"v{idx:02d}_mask.png")
    imaging.save_rgb(composite.overlay_preview(rgb, em.hard), pdir / f"v{idx:02d}_maskpreview.jpg")
    v = store.add_version(shoot, photo, out_path, "remove_object",
                          {"target": target, "method": how, "area_pct": round(em.area * 100, 2), "instances": em.instances, "mask_backend": em.backend},
                          f"Removed {target} ({em.instances} region{'s' if em.instances != 1 else ''}, {em.area*100:.1f}% of frame) via {how}",
                          mask_path=mask_path, cost_usd=cost)
    out = {"found": True, "photo_id": photo.id, "version_id": v.id, "version_index": v.index, "target": target,
           "method": how, "instances": em.instances, "area_pct": round(em.area * 100, 2),
           "mask_scores": [round(s, 2) for s in em.scores][:12],
           "result_url": store.rel_url(out_path), "mask_preview_url": store.rel_url(pdir / f"v{idx:02d}_maskpreview.jpg"),
           "cost_usd": cost}
    if qc and "generative" in how:
        if verdict is None:
            if progress:
                await progress(f"QC check on {photo.name}")
            verdict = await qc_check(f"Remove the {target}; everything else must stay identical.", head.path, str(out_path))
            v.cost_usd += verdict.get("cost_usd", 0); shoot.total_cost_usd += verdict.get("cost_usd", 0)
        v.params["qc"] = verdict
        store.save(shoot)
        out["qc"] = {k: verdict.get(k) for k in ("verdict", "fulfilled", "collateral", "artifacts", "reason")}
    return out
