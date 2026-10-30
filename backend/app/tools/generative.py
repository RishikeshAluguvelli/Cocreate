"""Open-ended edits (turn on lamps, TV screen from a reference, smooth bedding, regenerate grass, add objects).

The prompt model repaints the whole frame; we then paste its pixels back into the original ONLY where the edit
belongs (a SAM mask for `protect_target`, or the detected change region), so unrelated parts stay byte-identical.
"""
from __future__ import annotations

import asyncio

from ..config import settings
from ..models import Photo, Shoot
from .. import store
from . import composite, editors, imaging, masks
from .qc import qc_check

EDIT_PREAMBLE = ("You are editing a finished real-estate photograph. Image 1 is the photo to edit. "
                 "Make exactly this change and nothing else: ")
EDIT_SUFFIX = (" Keep the camera angle, framing, perspective, lighting, colours, walls, floors, furniture and every "
               "other detail exactly as they are. Photorealistic, no added objects, no text. Return the full image.")


def _resolve_refs(photo: Photo, refs: list[str] | None) -> list[str]:
    """refs: ['reference:0', 'bracket:1', 'original'] -> file paths."""
    out: list[str] = []
    for r in refs or []:
        kind, _, idx = r.partition(":")
        if kind == "reference" and photo.reference_paths:
            out.append(photo.reference_paths[int(idx or 0) % len(photo.reference_paths)])
        elif kind == "bracket" and photo.bracket_paths:
            i = int(idx) if idx else len(photo.bracket_paths) // 2
            out.append(photo.bracket_paths[i % len(photo.bracket_paths)])
        elif kind == "original":
            out.append(photo.original_path)
    return out


async def generative_edit(shoot: Shoot, photo: Photo, instruction: str, references: list[str] | None = None,
                          protect: str = "auto", protect_dilate_pct: float = 0.02, resolution: str | None = None,
                          progress=None, qc: bool = True, user_mask=None, save: bool = True) -> dict:
    """save=False: compute the composited result and return it as arrays ("final", "hard", "note", "cost_usd", "changed")
    without creating a version; used by deterministic tools that try a generative version as a QC-gated fallback."""
    head = photo.head
    rgb = imaging.load_rgb(head.path)
    H, W = rgb.shape[:2]
    ref_paths = _resolve_refs(photo, references)

    # 1) protection mask first (so the model can work on a close-up window around it)
    if user_mask is not None and protect and protect.replace("_", " ").strip().lower() in ("painted area", "painted", "user mask", "the painted area", "marked area"):
        protect = "none"   # the painted region itself is the protection; nothing to detect
    em = None
    note = ""
    if protect not in ("auto", "none"):
        if progress:
            await progress(f"{photo.name}: locating '{protect}'")
        em = await masks.text_mask(head.path, (W, H), protect, dilate_pct=protect_dilate_pct, feather_pct=0.01)
        if em is None:
            note = f"'{protect}' not found; fell back to change-region protection"
    if user_mask is not None:   # painted by the user: the edit is confined to (detection ∪ painted area)
        em = masks.merge_user_mask(em, user_mask, (W, H), dilate_pct=protect_dilate_pct, feather_pct=0.01)
        note = (note + "; " if note else "") + "includes the area the user painted"
    win = editors.crop_window(em.hard, expand=0.8, min_side=1024, max_frac=0.35) if em is not None else None

    async def render(strong: bool) -> tuple:
        if win:
            sub = rgb[win.y0:win.y1, win.x0:win.x1]
            lead = "Image 1 is a close-up crop of a finished real-estate photograph. "
        else:
            sub = rgb
            lead = EDIT_PREAMBLE
        prompt = lead + "Make exactly this change and nothing else: " + instruction.strip() if win else lead + instruction.strip()
        if ref_paths:
            prompt += " Images 2 onwards are references supplied by the customer; use them for the content/colour requested."
        if strong:
            prompt += " The previous attempt returned the photo unchanged - the change must be clearly visible this time."
        prompt += EDIT_SUFFIX
        h, w = sub.shape[:2]
        out, model_name, cost = await editors.prompt_edit(prompt, [sub, *ref_paths], (w, h), resolution=resolution)
        if win:
            full = rgb.copy()
            full[win.y0:win.y1, win.x0:win.x1] = out
            out = full
        return out, model_name, cost

    if progress:
        await progress(f"{photo.name}: rendering with Nano Banana{' (close-up window)' if win else ''} (~20-40 s)")
    edited, model_name, cost = await render(False)
    check_mask = em.hard if em is not None else None
    tiny = (editors.change_inside(rgb, edited, check_mask) < 0.015) if check_mask is not None else \
        (composite.mask_area_fraction(composite.change_mask(rgb, edited, thresh=0.07)) < 0.0005)
    if tiny:
        if progress:
            await progress(f"{photo.name}: model returned no change, retrying once")
        edited2, model_name, cost2 = await render(True)
        cost += cost2
        tiny2 = (editors.change_inside(rgb, edited2, check_mask) < 0.015) if check_mask is not None else \
            (composite.mask_area_fraction(composite.change_mask(rgb, edited2, thresh=0.07)) < 0.0005)
        if tiny2:
            shoot.total_cost_usd += cost
            store.save(shoot)
            return {"photo_id": photo.id, "changed": False, "cost_usd": cost,
                    "message": f"The image model returned {photo.name} unchanged twice; no version created. Rephrase the "
                               f"instruction more concretely (what should it look like afterwards?) or change `protect`."}
        edited = edited2

    if progress:
        await progress(f"{photo.name}: compositing back into the original" + (f" (protecting '{protect}')" if protect not in ("auto", "none") else ""))

    if protect == "none":
        soft = None
        hard = composite.change_mask(rgb, edited)
        final = edited
        note = "whole frame accepted"
    elif protect == "auto" or em is None:
        hard = composite.change_mask(rgb, edited, thresh=0.07, min_blob_px=int(0.0004 * W * H))
        hard = composite.dilate(hard, int(max(W, H) * 0.008))
        soft = composite.feather(hard, int(max(W, H) * settings.mask_feather_pct))
        final = await asyncio.to_thread(composite.composite, rgb, edited, soft)
        note = note or "kept only the regions the model changed"
    else:
        cost += 0.01
        hard, soft = em.hard, em.soft
        note = f"edit confined to '{protect}' mask" + (" (close-up render)" if win else "")
        final = await asyncio.to_thread(composite.composite, rgb, edited, soft)

    area = composite.mask_area_fraction(hard)
    if not save:
        shoot.total_cost_usd += cost
        return {"photo_id": photo.id, "changed": True, "final": final, "hard": hard, "note": note, "cost_usd": cost, "area_pct": round(area * 100, 2)}
    pdir = store.photo_dir(photo)
    idx = len(photo.versions)
    out_path = imaging.save_rgb(final, pdir / f"v{idx:02d}_edit.jpg")
    mask_path = imaging.save_mask(hard, pdir / f"v{idx:02d}_mask.png")
    imaging.save_rgb(composite.overlay_preview(rgb, hard, color=(0.2, 0.6, 1.0)), pdir / f"v{idx:02d}_maskpreview.jpg")
    v = store.add_version(shoot, photo, out_path, "generative_edit",
                          {"instruction": instruction, "references": references or [], "protect": protect, "area_pct": round(area * 100, 2)},
                          f"{instruction.strip().rstrip('.')} ({note}, {area*100:.1f}% of frame changed)",
                          mask_path=mask_path, cost_usd=cost)
    out = {"photo_id": photo.id, "version_id": v.id, "version_index": v.index, "changed_area_pct": round(area * 100, 2),
           "protection": note, "result_url": store.rel_url(out_path),
           "mask_preview_url": store.rel_url(pdir / f"v{idx:02d}_maskpreview.jpg"), "cost_usd": cost}
    if qc:
        if progress:
            await progress(f"QC check on {photo.name}")
        verdict = await qc_check(instruction, head.path, str(out_path))
        v.params["qc"] = verdict
        v.cost_usd += verdict.get("cost_usd", 0); shoot.total_cost_usd += verdict.get("cost_usd", 0)
        store.save(shoot)
        out["qc"] = {k: verdict.get(k) for k in ("verdict", "fulfilled", "collateral", "artifacts", "reason")}
    return out
