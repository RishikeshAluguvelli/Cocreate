"""Text-prompted masks with the dilation/feather policy used by every masked edit.

Backends: fal SAM-3 (pixel-accurate) when FAL_KEY is set; otherwise Gemini detection boxes refined locally with
GrabCut (free). Points placed in the UI become boxes around the point."""
from __future__ import annotations

import asyncio

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import settings
from ..providers.fal import MaskResult, fal_provider
from ..providers.gemini import _parse_json_list, gemini_provider, image_part
from . import composite, imaging


@dataclass
class EditMask:
    raw: np.ndarray      # model output (union)
    hard: np.ndarray     # dilated, what the model is asked to repaint
    soft: np.ndarray     # feathered, what gets composited
    instances: int
    area: float          # fraction of the image covered by `hard`
    scores: list[float]
    backend: str = "sam3"
    instance_masks: list[np.ndarray] | None = None  # per-instance raw masks when the backend provides them


def _norm_points_to_px(points: list[list[float]] | None, W: int, H: int, label: int = 1) -> list[dict] | None:
    if not points:
        return None
    return [{"x": int(round(p[0] * W)), "y": int(round(p[1] * H)), "label": label} for p in points]


def grabcut_refine(rgb: np.ndarray, box: list[float], iters: int = 5, pad: float = 0.25, max_side: int = 640,
                   fallback_box: bool = True) -> np.ndarray:
    """Turn a normalized box into a tight object mask with GrabCut (CPU, ~0.3-1.5 s). Falls back to the box itself
    when GrabCut finds almost nothing (thin objects like cords); with fallback_box=False it returns empty instead."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * W, (y1 - y0) * H
    cx0, cy0 = int(max(0, x0 * W - pad * bw)), int(max(0, y0 * H - pad * bh))
    cx1, cy1 = int(min(W, x1 * W + pad * bw)), int(min(H, y1 * H + pad * bh))
    crop = (np.clip(rgb[cy0:cy1, cx0:cx1], 0, 1) * 255).astype(np.uint8)
    full = np.zeros((H, W), np.float32)
    if crop.size == 0:
        return full
    scale = 1.0
    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_side:
        scale = max_side / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    rect = (int((x0 * W - cx0) * scale), int((y0 * H - cy0) * scale),
            max(2, int(bw * scale)), max(2, int(bh * scale)))
    gc_mask = np.zeros(crop.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        m = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        m = np.zeros(crop.shape[:2], np.uint8)
    box_area = rect[2] * rect[3]
    if m.sum() < 0.12 * box_area:  # thin / low-contrast object: use the box
        m[:] = 0
        if fallback_box:
            m[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
    else:
        # keep the largest blobs, drop specks
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.02 * box_area:
                keep[labels == i] = 1
        m = keep if keep.any() else m
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if scale != 1.0:
        m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
    full[cy0:cy0 + m.shape[0], cx0:cx0 + m.shape[1]] = m
    return full


def _iou(a: list[float], b: list[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _merge_boxes(boxes: list[list[float]], thresh: float = 0.45) -> list[list[float]]:
    out: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(b, o) < thresh for o in out):
            out.append(b)
    return out


async def detect_tiled(rgb: np.ndarray, target: str, grid: int = 2, overlap: float = 0.12) -> list[dict]:
    """Detect on a grid of overlapping crops (each sent at full detector resolution) and merge. Catches the small
    instances a single 1024px pass misses on aerials and wide exteriors."""
    H, W = rgb.shape[:2]
    out: list[dict] = []
    tw, th = W / grid, H / grid

    async def tile(i: int, j: int) -> list[dict]:
        x0 = max(0, int(j * tw - overlap * tw)); y0 = max(0, int(i * th - overlap * th))
        x1 = min(W, int((j + 1) * tw + overlap * tw)); y1 = min(H, int((i + 1) * th + overlap * th))
        dets = await gemini_provider.detect(rgb[y0:y1, x0:x1], target, max_objects=40)
        res = []
        for d in dets:
            bx0, by0, bx1, by1 = d["box"]
            res.append({"box": [(x0 + bx0 * (x1 - x0)) / W, (y0 + by0 * (y1 - y0)) / H,
                                (x0 + bx1 * (x1 - x0)) / W, (y0 + by1 * (y1 - y0)) / H], "label": d.get("label", target)})
        return res

    import asyncio
    for group in await asyncio.gather(*(tile(i, j) for i in range(grid) for j in range(grid))):
        out.extend(group)
    return out


def _needs_tiling(dets: list[dict], W: int, H: int) -> bool:
    if len(dets) < 4:
        return False
    areas = sorted((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) for d in dets)
    return areas[len(areas) // 2] < 0.004 or (len(dets) >= 12 and max(W, H) >= 1800)


def extend_shadow(mask: np.ndarray, box: list[float], frac: float = 0.12) -> np.ndarray:
    """Ground-standing objects cast a contact shadow just below them; include it so the fill does not leave a dark
    'platform'. Shifts a copy of the mask down by `frac` of the box height and unions it."""
    H = mask.shape[0]
    dy = int((box[3] - box[1]) * H * frac)
    if dy <= 0:
        return mask
    shifted = np.zeros_like(mask)
    shifted[dy:] = mask[:-dy]
    return np.clip(mask + shifted, 0, 1)


async def _gemini_mask(rgb: np.ndarray, target: str | None, points: list[list[float]] | None, box: list[float] | None) -> MaskResult:
    H, W = rgb.shape[:2]
    boxes: list[list[float]] = []
    if points and is_surface(target):
        # a point on a wall / floor / lawn: grow the surface from exactly that spot (no detector, no box)
        inst = [await asyncio.to_thread(lambda p=p: region_at_point(rgb, p[0], p[1], 50.0, "surface")[0]) for p in points]
        inst = [m for m in inst if m.sum() > 16]
        union = np.clip(np.sum(inst, axis=0), 0, 1).astype(np.float32) if inst else np.zeros((H, W), np.float32)
        return MaskResult(mask=union, instances=inst, scores=[1.0] * len(inst), boxes=[])
    if points:
        r = 0.06
        for px, py in points:
            boxes.append([max(0, px - r), max(0, py - r * W / H), min(1, px + r), min(1, py + r * W / H)])
        if target:
            dets = await gemini_provider.detect(rgb, target)
            # keep detections that contain a point; else keep the point boxes
            hit = [d["box"] for d in dets if any(d["box"][0] <= px <= d["box"][2] and d["box"][1] <= py <= d["box"][3] for px, py in points)]
            if hit:
                boxes = hit
    elif target:
        dets = await gemini_provider.detect(rgb, target, restrict_box=box, max_objects=40)
        if not box and _needs_tiling(dets, W, H):
            dets = dets + await detect_tiled(rgb, target)
        boxes = _merge_boxes([d["box"] for d in dets])
        if box:
            boxes = [b for b in boxes if b[2] > box[0] and b[0] < box[2] and b[3] > box[1] and b[1] < box[3]]
    elif box:
        boxes = [box]
    surface = is_surface(target)
    refine = surface_refine if surface else grabcut_refine
    inst = []
    for b in boxes:
        m = await asyncio.to_thread(refine, rgb, b)   # GrabCut / colour growth: CPU work off the event loop
        if not surface:
            m = extend_shadow(m, b)
        inst.append(m)
    inst = [m for m in inst if m.sum() > 16]
    if surface and inst and target and (excl := exclusion_prompt(target)):
        carve = np.zeros((H, W), np.float32)
        try:
            exclusions = await gemini_provider.detect(rgb, excl, max_objects=30)
        except Exception:  # best effort: a failed exclusion pass must not fail the edit
            exclusions = []
        union0 = np.clip(np.sum(inst, axis=0), 0, 1)
        picked = []
        for d in exclusions:
            b = d["box"]
            if (b[2] - b[0]) * (b[3] - b[1]) > 0.5:      # a box this big is the surface itself, not something in front of it
                continue
            sub = union0[int(b[1] * H):int(b[3] * H) + 1, int(b[0] * W):int(b[2] * W) + 1]
            if sub.size == 0 or sub.mean() < 0.02:        # nothing of the surface mask inside this box: nothing to carve
                continue
            picked.append(b)
        picked.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        for b in picked[:14]:                             # GrabCut is ~0.2 s each at this size; cap the total
            carve = np.maximum(carve, await asyncio.to_thread(grabcut_refine, rgb, b, max_side=400, fallback_box=False))
        if carve.any():
            carve = cv2.dilate(carve.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
            inst = [m * (1 - carve) for m in inst]
            inst = [m for m in inst if m.sum() > 16]
    if surface and inst and target and settings.mask_review:
        union0 = np.clip(np.sum(inst, axis=0), 0, 1)
        try:
            wrong = await review_surface_mask(rgb, union0, target)
        except Exception as e:  # best effort
            print(f"[mask review] failed for {target!r}: {type(e).__name__}: {str(e)[:120]}")
            wrong = []
        if wrong:
            lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
            med = np.median(lab[union0 > 0.5].reshape(-1, 3), axis=0)
            differs = (np.abs(lab[..., 0] - med[0]) > 6) | (np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2]) > 4)
            carve = np.zeros((H, W), np.float32)
            for b in wrong[:8]:
                if (b[2] - b[0]) * (b[3] - b[1]) > 0.5:
                    continue
                m = await asyncio.to_thread(grabcut_refine, rgb, b, max_side=400, fallback_box=False)
                sl = (slice(int(b[1] * H), int(b[3] * H) + 1), slice(int(b[0] * W), int(b[2] * W) + 1))
                if not m.any():  # flat region (ceiling): carve the colour-different pixels inside the box
                    m = np.zeros((H, W), np.float32)
                    m[sl] = (differs[sl] & (union0[sl] > 0.5)).astype(np.float32)
                box_in_mask = float((union0[sl] > 0.5).sum())
                if box_in_mask and m.sum() < 0.1 * box_in_mask and (b[2] - b[0]) * (b[3] - b[1]) < 0.04:
                    # same colour as the surface and GrabCut found nothing (a table top in the wall's colour): for a small
                    # box the reviewer's rectangle is the best information we have
                    m = np.zeros((H, W), np.float32)
                    m[sl] = (union0[sl] > 0.5).astype(np.float32)
                carve = np.maximum(carve, m)
            frac = float((carve * union0).sum() / max(1.0, union0.sum()))
            print(f"[mask review] {target!r}: {len(wrong)} wrongly covered part(s), carving {frac*100:.1f}% of the mask")
            if carve.any() and frac < 0.6:   # never let the review delete the surface itself
                carve = cv2.dilate(carve.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
                inst = [m * (1 - carve) for m in inst]
                inst = [m for m in inst if m.sum() > 16]
        else:
            print(f"[mask review] {target!r}: nothing wrongly covered")
    union = np.clip(np.sum(inst, axis=0), 0, 1).astype(np.float32) if inst else np.zeros((H, W), np.float32)
    return MaskResult(mask=union, instances=inst, scores=[1.0] * len(inst), boxes=boxes)


def merge_user_mask(em: EditMask | None, user_mask: np.ndarray, size_wh: tuple[int, int],
                    dilate_pct: float | None = None, feather_pct: float | None = None) -> EditMask:
    """Union of what detection found (may be None) and a region the user painted by hand."""
    W, H = size_wh
    um = user_mask
    if um.shape != (H, W):
        um = cv2.resize(um.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    um = (um > 0.5).astype(np.float32)
    raw = um if em is None else np.maximum(em.raw, um)
    hard, soft = composite.prepare_edit_mask(raw, max(W, H),
                                             settings.mask_dilate_pct if dilate_pct is None else dilate_pct,
                                             settings.mask_feather_pct if feather_pct is None else feather_pct)
    return EditMask(raw=raw, hard=hard, soft=soft, instances=(em.instances if em else 0) + 1,
                    area=composite.mask_area_fraction(hard), scores=[*(em.scores if em else []), 1.0],
                    backend=((em.backend + "+") if em else "") + "painted", instance_masks=None)


async def text_mask(image_path: str | Path, size_wh: tuple[int, int], target: str | None,
                    points: list[list[float]] | None = None, box: list[float] | None = None,
                    dilate_pct: float | None = None, feather_pct: float | None = None,
                    max_masks: int = 12, min_score: float = 0.35) -> EditMask | None:
    """Return None when nothing matching `target` is found in the photo."""
    W, H = size_wh
    backend = settings.image_backend
    if backend == "fal":
        boxes = None
        if box:
            boxes = [{"x_min": int(box[0] * W), "y_min": int(box[1] * H), "x_max": int(box[2] * W), "y_max": int(box[3] * H)}]
        res: MaskResult = await fal_provider.segment(image_path, target, (W, H), points=_norm_points_to_px(points, W, H),
                                                     boxes=boxes, max_masks=max_masks, min_score=min_score)
        name = "sam3"
    elif backend == "gemini":
        rgb = imaging.load_rgb(image_path)
        res = await _gemini_mask(rgb, target, points, box)
        name = "gemini+grabcut"
    else:
        res = await fal_provider.segment(image_path, target, (W, H), points=_norm_points_to_px(points, W, H))
        name = "mock"
    if not res.found:
        return None
    long_side = max(W, H)
    hard, soft = composite.prepare_edit_mask(res.mask, long_side,
                                             settings.mask_dilate_pct if dilate_pct is None else dilate_pct,
                                             settings.mask_feather_pct if feather_pct is None else feather_pct)
    return EditMask(raw=res.mask, hard=hard, soft=soft, instances=len(res.instances),
                    area=composite.mask_area_fraction(hard), scores=res.scores, backend=name,
                    instance_masks=list(res.instances) if res.instances else None)


# ---------------- flat surfaces (walls, floors, lawns): colour-region refinement instead of GrabCut ----------------
SURFACE_WORDS = ("wall", "floor", "ceiling", "lawn", "grass", "roof", "sky", "water", "carpet", "counter", "tile",
                 "driveway", "road", "deck", "patio", "paint", "cabinet", "door", "siding", "fence")


def is_surface(target: str | None) -> bool:
    t = (target or "").lower()
    return any(w in t for w in SURFACE_WORDS)


# Things that sit in front of a surface and must never be repainted with it. Detected with one cheap box call and
# carved out of the colour-grown mask (a pale window view or a light curtain can match a beige wall colour-wise).
SURFACE_EXCLUSIONS: list[tuple[tuple[str, ...], str]] = [
    (("wall", "paint", "siding"), "window, door, doorway, picture frame, artwork, mirror, curtain, cabinet, shelf, table, counter, kitchen island, sofa, chair, bed, headboard, television, lamp, radiator, plant"),
    (("ceiling",), "light fixture, ceiling fan, beam, vent"),
    (("floor", "tile", "carpet"), "rug, furniture, plant, person, pet, box"),
    (("sky",), "roof, building, tree, pole, wire, sign"),
    (("lawn", "grass"), "tree, bush, car, person, path, driveway, patio, pool, fence"),
]


async def review_surface_mask(rgb: np.ndarray, mask: np.ndarray, target: str) -> list[list[float]]:
    """Show the vision model the mask as a red overlay and ask what it wrongly covers; returns boxes of those parts.
    Catches the cases colour growth cannot separate: a ceiling or white sideboard that touches a light wall."""
    overlay = composite.overlay_preview(rgb, mask)
    prompt = (f"The red overlay on this photo is meant to cover only the {target}. List every distinct thing the red overlay "
              f"wrongly covers that is NOT {target} (for example ceiling, floor, cabinet, table, sofa, curtain, window, artwork, "
              f"door, roof, trees). Return a JSON list where each item has \"box_2d\": [ymin, xmin, ymax, xmax] normalized to "
              f"0-1000 around that wrongly covered part only, and \"label\": what it is. Ignore slight overlaps at edges. "
              f"Return [] if the overlay covers only the {target}.")
    d = await gemini_provider.generate(settings.gemini_vision_model,
                                       [{"role": "user", "parts": [image_part(overlay, 1024, 86), {"text": prompt}]}],
                                       generation_config={"responseMimeType": "application/json", "maxOutputTokens": 2000,
                                                          "thinkingConfig": {"thinkingLevel": "minimal"}})
    txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    out = []
    for o in _parse_json_list(txt):
        b = o.get("box_2d") if isinstance(o, dict) else None
        if not b or len(b) != 4:
            continue
        y0, x0, y1, x1 = [max(0.0, min(1.0, float(v) / 1000.0)) for v in b]
        if x1 - x0 > 0.004 and y1 - y0 > 0.004:
            out.append([x0, y0, x1, y1])
    return out


def exclusion_prompt(target: str | None) -> str | None:
    t = (target or "").lower()
    for words, prompt in SURFACE_EXCLUSIONS:
        if any(w in t for w in words):
            return prompt
    return None


def _step_edges(lab: np.ndarray, canny: tuple[int, int] = (18, 45), g_grow: float = 26.0, min_edge_px_frac: float = 0.005,
                step_min: float = 8.0, step_chroma: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Barrier map for colour growth: lightness edges that separate two different colour levels (wall|cabinet, roof|sky).
    Thin lines between same-coloured areas (grout, cables, trim) are dropped, so growth crosses them. Returns
    (edge: bool HxW, grad: float HxW lightness gradient magnitude)."""
    H, W = lab.shape[:2]
    L = lab[..., 0]
    smooth = cv2.GaussianBlur(L, (0, 0), 1.2)
    grad = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    # ---- edge barrier: Canny on lightness catches low-contrast edges (cabinet/wall, roof/sky); speckles are dropped
    L8 = np.clip(cv2.GaussianBlur(L, (0, 0), 1.0) * 2.55, 0, 255).astype(np.uint8)
    edge = cv2.Canny(L8, canny[0], canny[1]) > 0
    edge |= grad >= g_grow
    n_e, lab_e, st_e, _ = cv2.connectedComponentsWithStats(edge.astype(np.uint8), 8)
    min_px = max(6, int(min_edge_px_frac * max(W, H)))
    small = np.zeros(n_e, bool); small[1:] = st_e[1:, cv2.CC_STAT_AREA] < min_px
    edge = edge & ~small[lab_e]
    # step test: sample the lightly blurred colour a few pixels to either side of the edge along its normal; a step
    # (wall|cabinet, roof|sky) shows two different levels, a thin line (cable, grout, trim) shows the same level twice
    Ls = cv2.GaussianBlur(L, (0, 0), 2.0)
    gx = cv2.Sobel(Ls, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(Ls, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) + 1e-6
    d = max(6.0, 0.004 * max(W, H))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (gx / mag).astype(np.float32), (gy / mag).astype(np.float32)
    labs = cv2.GaussianBlur(lab, (0, 0), 2.0); labs[..., 0] = Ls
    Pp = cv2.remap(labs, xs + d * nx, ys + d * ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Pm = cv2.remap(labs, xs - d * nx, ys - d * ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    step = (np.abs(Pp[..., 0] - Pm[..., 0]) > step_min) | (np.hypot(Pp[..., 1] - Pm[..., 1], Pp[..., 2] - Pm[..., 2]) > step_chroma)
    edge = edge & step
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return edge, grad


def _fill_holes(keep: np.ndarray, dchroma: np.ndarray, dlight: np.ndarray, chroma_tol: float, grow_light_tol: float) -> np.ndarray:
    """Fill tiny holes (sockets, switches) and holes that share the surface colour (soft shadows); artwork, windows and
    anything of another colour stay out. Holes touching the frame border are never filled."""
    H, W = keep.shape
    inv = (1 - keep).astype(np.uint8)
    nh, hl, hs, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, nh):
        x, y, w, h, a = hs[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        hole = hl == i
        tiny = a < 0.0008 * W * H
        similar = a < 0.005 * W * H and float(np.median(dchroma[hole])) < chroma_tol * 1.5 and float(np.median(dlight[hole])) < grow_light_tol
        if tiny or similar:
            keep[hole] = 1
    return keep


def grabcut_regions(rgb: np.ndarray, fg: np.ndarray, pr_fg: np.ndarray, roi: np.ndarray, iters: int = 5, max_side: int = 640) -> np.ndarray:
    """GrabCut from labelled regions: `fg` definite foreground, `pr_fg` probable foreground, `roi` the area that may
    still turn out to be foreground (probable background); everything else is background. Returns the foreground
    pieces connected to `fg` (stray matches elsewhere in the window are dropped)."""
    H, W = rgb.shape[:2]
    full = np.zeros((H, W), np.float32)
    bb = composite.mask_bbox(roi)
    if bb is None or not (fg > 0.5).any():
        return full
    x0, y0, x1, y1 = bb
    crop = (np.clip(rgb[y0:y1 + 1, x0:x1 + 1], 0, 1) * 255).astype(np.uint8)
    f, pf, r = (m[y0:y1 + 1, x0:x1 + 1] > 0.5 for m in (fg, pr_fg, roi))
    ch, cw = crop.shape[:2]
    scale = 1.0
    if max(ch, cw) > max_side:
        scale = max_side / max(ch, cw)
        size = (int(cw * scale), int(ch * scale))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        f, pf, r = (cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in (f, pf, r))
    gc = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    gc[r] = cv2.GC_PR_BGD
    gc[pf] = cv2.GC_PR_FGD
    gc[f] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return full
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(out, 8)
    keep = np.zeros_like(out)
    for i in range(1, n):
        comp = labels == i
        if (comp & f).any():
            keep[comp] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if scale != 1.0:
        keep = cv2.resize(keep, (cw, ch), interpolation=cv2.INTER_NEAREST)
    full[y0:y0 + keep.shape[0], x0:x0 + keep.shape[1]] = keep
    return full


def grabcut_from_mask(rgb: np.ndarray, mask: np.ndarray, iters: int = 5, max_side: int = 640, pad: float = 0.35) -> np.ndarray:
    """Refine a rough region into an object mask: its eroded core is definite foreground, the region probable
    foreground, a band around it probable background."""
    H, W = rgb.shape[:2]
    m = (mask > 0.5).astype(np.uint8)
    bb = composite.mask_bbox(mask)
    if bb is None:
        return np.zeros((H, W), np.float32)
    k = max(3, int(0.03 * max(bb[2] - bb[0], bb[3] - bb[1])) | 1)
    inner = cv2.erode(m, np.ones((k, k), np.uint8))
    outer = cv2.dilate(m, np.ones((3 * k, 3 * k), np.uint8))
    return grabcut_regions(rgb, (inner if inner.any() else m).astype(np.float32), m.astype(np.float32), outer.astype(np.float32), iters, max_side)


def _fit_shading_plane(L: np.ndarray, comp: np.ndarray) -> np.ndarray | None:
    """Least-squares plane L ~ a*x + b*y + c over the component: a wall is lit unevenly (window on one side, vignetting),
    so comparing lightness to a shading model instead of one global median lets growth follow the lighting gradient
    without opening the band wide enough to swallow the ceiling."""
    ys, xs = np.nonzero(comp)
    if len(xs) < 500:
        return None
    step = max(1, len(xs) // 20000)
    xs_, ys_ = xs[::step].astype(np.float32), ys[::step].astype(np.float32)
    A = np.stack([xs_, ys_, np.ones_like(xs_)], axis=1)
    coef, *_ = np.linalg.lstsq(A, L[ys_.astype(int), xs_.astype(int)], rcond=None)
    H, W = L.shape
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(coef[0] * gx + coef[1] * gy + coef[2], 0, 100)


def follow_lighting(rgb: np.ndarray, region: np.ndarray, tolerance: float = 50.0, scale_to: int = 900, iters: int = 80) -> np.ndarray:
    """Extend a surface region into parts that fade into strong light (the far end of a tiled hallway, a wall next to
    a window): pixels join when their colour is continuous with the already-accepted pixels *nearby* (local mean in a
    window), never because they match the clicked colour globally. Barriers are the finer step edges, the excursion is
    capped and the region may at most quadruple. Runs on a downscaled copy; the result is unioned with `region`."""
    H, W = rgb.shape[:2]
    t = float(np.clip(tolerance, 5.0, 100.0)) / 50.0
    s = min(1.0, scale_to / max(W, H))
    small = cv2.resize(rgb, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else rgb
    lab = cv2.cvtColor(np.clip(small, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    reg = cv2.resize((region > 0.5).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    if reg.sum() < 50:
        return region
    med = np.median(lab[reg > 0].reshape(-1, 3), axis=0)
    edge, _ = _step_edges(lab, step_min=5.5, step_chroma=4.0)
    dch = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2])
    # the far, sunlit end of a floor is nearly white, so the chroma cap must stay loose; what keeps a neighbouring white
    # wall out is local continuity (tight per-step tolerances) plus the step-edge barrier, not the global colour
    allowed = ~edge & (dch < 12.0 * t + 2) & (np.abs(lab[..., 0] - med[0]) < 32)
    radius = max(2, int(0.004 * max(small.shape[:2])))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    win = (6 * radius + 1, 6 * radius + 1)
    tol_l, tol_c, max_area = 3.5 * t, 2.0 * t, 4 * int(reg.sum())
    for _ in range(iters):
        frontier = cv2.dilate(reg, kernel).astype(bool) & (reg == 0) & allowed
        if not frontier.any() or reg.sum() > max_area:
            break
        w = cv2.boxFilter(reg.astype(np.float32), -1, win, normalize=False)
        ref = cv2.boxFilter(lab * reg[..., None].astype(np.float32), -1, win, normalize=False) / np.maximum(w, 1e-3)[..., None]
        ok = frontier & (w > 0) & (np.abs(lab[..., 0] - ref[..., 0]) < tol_l) & (np.hypot(lab[..., 1] - ref[..., 1], lab[..., 2] - ref[..., 2]) < tol_c)
        if not ok.any():
            break
        reg[ok] = 1
    grown = cv2.resize(reg, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32) if s < 1.0 else reg.astype(np.float32)
    grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32)
    return np.maximum(grown, (region > 0.5).astype(np.float32))


def region_at_point(rgb: np.ndarray, x: float, y: float, tolerance: float = 50.0, mode: str = "auto") -> tuple[np.ndarray, str]:
    """Mask of the surface or object under a clicked point (normalized x, y), for the eyedropper and one-click recolour.

    Colour growth from the clicked pixel with the same barrier logic as `surface_refine` (step edges stop it, thin lines
    are crossed), lightness compared against a fitted shading plane so uneven lighting is followed, then hole filling.
    Near-neutral surfaces (white, grey) get a finer step test and a narrower lightness band because the things around
    them (ceiling, trim, bedding) are usually near-neutral too. `tolerance` (5..100, 50 = default) widens the bands.
    `mode`: 'surface' = colour growth only (walls, floors, lawns); 'object' = plus a GrabCut refinement so a sofa or
    cabinet with shading and texture is taken as a whole, limited to a wide colour band around the clicked colour;
    'auto' = GrabCut only when the grown region is small (an object). Returns (mask, mode_used). No model calls."""
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    px, py = int(np.clip(x * W, 0, W - 1)), int(np.clip(y * H, 0, H - 1))
    t = float(np.clip(tolerance, 5.0, 100.0)) / 50.0
    r = max(3, int(0.006 * max(W, H)))
    sl = (slice(max(0, py - r), py + r + 1), slice(max(0, px - r), px + r + 1))
    win = lab[sl].reshape(-1, 3)
    med0 = np.median(win, axis=0)
    neutral = float(np.hypot(med0[1], med0[2])) < 8.0
    step_min, step_chroma = (5.5, 4.0) if neutral else (8.0, 6.0)
    chroma_tol = 6.0 * t
    grow_light = (20.0 if neutral else 26.0) * t
    # texture: grass, gravel, foliage, patterned fabric are edges everywhere at pixel scale. Measure the lightness
    # gradient around the click; on a textured surface work on a blurred image (texture averages out, real boundaries
    # survive), with coarser edge thresholds and a wider band
    g0 = np.hypot(cv2.Sobel(cv2.GaussianBlur(lab[..., 0], (0, 0), 1.2), cv2.CV_32F, 1, 0, ksize=3),
                  cv2.Sobel(cv2.GaussianBlur(lab[..., 0], (0, 0), 1.2), cv2.CV_32F, 0, 1, ksize=3))
    tex = float(np.median(g0[sl]))
    if tex > 9.0:
        sigma = max(1.5, 0.004 * max(W, H)) * min(2.0, tex / 9.0)
        lab = cv2.GaussianBlur(lab, (0, 0), sigma)
        # a coloured texture (grass, brick) can take a wider band; a neutral one (bedding, concrete) cannot, because
        # everything around it is neutral too
        chroma_tol *= 1.0 if neutral else 1.5
        grow_light *= 1.1 if neutral else 1.3
        edge, grad = _step_edges(lab, canny=(30, 75), g_grow=40.0, step_min=step_min + 4, step_chroma=step_chroma + 3)
    else:
        edge, grad = _step_edges(lab, step_min=step_min, step_chroma=step_chroma)
    L = lab[..., 0]
    win = lab[sl].reshape(-1, 3)
    smooth_px = (~edge[sl]).reshape(-1)
    med = np.median(win[smooth_px] if smooth_px.sum() >= 9 else win, axis=0)
    dchroma = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2])
    k = max(3, int(0.004 * max(W, H)) | 1)

    def grow(dlight: np.ndarray) -> np.ndarray | None:
        loose = ((dchroma < chroma_tol + 1.5) & (dlight < grow_light) & ~edge).astype(np.uint8)
        loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        loose[edge] = 0
        n, labels = cv2.connectedComponents(loose, connectivity=4)
        lbl = int(labels[py, px])
        if lbl == 0:   # clicked exactly on a barrier pixel or an outlier: nearest grown pixel within a few radii
            y0, y1, x0, x1 = max(0, py - 3 * r), min(H, py + 3 * r + 1), max(0, px - 3 * r), min(W, px + 3 * r + 1)
            sub = labels[y0:y1, x0:x1]
            ys, xs = np.nonzero(sub)
            if len(xs):
                i = int(np.argmin((ys + y0 - py) ** 2 + (xs + x0 - px) ** 2))
                lbl = int(sub[ys[i], xs[i]])
        return (labels == lbl) if lbl else None

    dlight = np.abs(L - med[0])
    comp = grow(dlight)
    if comp is None:   # nothing coherent around the click: a small disc so the caller still gets a colour
        disc = np.zeros((H, W), np.float32)
        cv2.circle(disc, (px, py), 2 * r, 1.0, -1)
        return disc, "point"
    for _ in range(2):   # refine the lightness band with the shading of what was found so far
        plane = _fit_shading_plane(L, comp)
        if plane is None:
            break
        dl2 = np.abs(L - plane)
        comp2 = grow(dl2)
        if comp2 is None or comp2.sum() < 0.8 * comp.sum() or comp2.sum() > 2.5 * comp.sum():
            break   # the plane made it worse (or exploded): keep the previous region
        comp, dlight = comp2, dl2
    keep = comp.astype(np.uint8)
    # the barrier strip belongs to whichever side it matches: grow back over adjacent pixels of the clicked colour so the
    # mask reaches the true boundary instead of stopping a few pixels short of it
    colour_ok = (dchroma < chroma_tol + 1.5) & (dlight < grow_light)
    reach = cv2.dilate(keep, np.ones((int(1.5 * k) | 1, int(1.5 * k) | 1), np.uint8)).astype(bool)
    keep[colour_ok & reach] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    keep = _fill_holes(keep, dchroma, dlight, chroma_tol, grow_light)
    used = "surface"
    area = float(keep.mean())
    if mode == "object" or (mode == "auto" and area < 0.02):
        # an object is what sits around the click, not every same-coloured thing in the room: GrabCut with the grown
        # pixels next to the click as definite foreground, the grown pixels a bit further out as probable foreground,
        # and a window around the click as the area that may still join. A wide colour band stops leaks into other hues.
        Rm = float(max(W, H))
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = np.hypot(gx - px, gy - py)
        kb = keep.astype(bool)
        core = kb & (dist < 0.05 * Rm)
        near = kb & (dist < 0.22 * Rm)
        roi = dist < 0.45 * Rm
        if not core.any():
            core = np.zeros((H, W), bool); cv2.circle(core.view(np.uint8), (px, py), r, 1, -1)
        gc = grabcut_regions(rgb, core.astype(np.float32), near.astype(np.float32), roi.astype(np.float32)) > 0.5
        band = (dchroma < chroma_tol * 2.5 + 4) & (dlight < min(60.0, grow_light * 2.2))
        obj = (gc & band) | core
        obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, labels, _, _ = cv2.connectedComponentsWithStats(obj, 8)
        lbl = int(labels[py, px])
        if lbl:
            keep = (labels == lbl).astype(np.uint8)
            keep = _fill_holes(keep, dchroma, dlight, chroma_tol * 2.5, 60.0)
            used = "object"
    return keep.astype(np.float32), used


def surface_refine(rgb: np.ndarray, box: list[float], chroma_tol: float = 9.0, light_tol: float = 18.0,
                   grow_light_tol: float = 24.0, g_seed: float = 18, g_grow: float = 26, canny: tuple[int, int] = (18, 45),
                   min_edge_px_frac: float = 0.005, sim_light: float = 10.0, sim_chroma: float = 5.0,
                   max_inner_edge: float = 0.04, step_min: float = 8.0, step_chroma: float = 6.0) -> np.ndarray:
    """Grow a mask over the painted surface inside (and slightly beyond) the box.

    Hysteresis colour growth from the box's median colour (seeds: strict band; growth: wider lightness band for shading),
    bounded by *step* edges: an edge counts as a barrier only when the colour differs on its two sides (wall|cabinet,
    roof|sky), so thin lines (cables, grout, trim) are crossed but neighbouring surfaces are not. The component holding
    most seeds is the surface; other components join only when they share its colour and are smooth. Small holes
    (sockets, soft shadows) are filled; artwork and window views stay out."""
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    edge, grad = _step_edges(lab, canny=canny, g_grow=g_grow, min_edge_px_frac=min_edge_px_frac, step_min=step_min, step_chroma=step_chroma)
    # ---- reference colour: median of the smooth (non-edge) pixels in the central half of the box
    cx0, cy0 = bx0 + (bx1 - bx0) // 4, by0 + (by1 - by0) // 4
    cx1, cy1 = max(bx1 - (bx1 - bx0) // 4, cx0 + 2), max(by1 - (by1 - by0) // 4, cy0 + 2)
    med = np.median(lab[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
    dchroma = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2]); dlight = np.abs(L - med[0])
    gx0, gy0 = max(0, int(bx0 - 0.06 * W)), max(0, int(by0 - 0.06 * H)); gx1, gy1 = min(W, int(bx1 + 0.06 * W)), min(H, int(by1 + 0.06 * H))
    lim = np.zeros((H, W), bool); lim[gy0:gy1, gx0:gx1] = True
    k = max(3, int(0.004 * max(W, H)) | 1)
    seed = ((dchroma < chroma_tol) & (dlight < light_tol) & (grad < g_seed) & ~edge & lim).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    # growth may run a little past the detection box: boxes are often a few percent short of the true edge of a wall
    lx0, ly0 = max(0, int(bx0 - 0.04 * W)), max(0, int(by0 - 0.04 * H)); lx1, ly1 = min(W, int(bx1 + 0.04 * W)), min(H, int(by1 + 0.04 * H))
    lim2 = np.zeros((H, W), bool); lim2[ly0:ly1, lx0:lx1] = True
    loose = ((dchroma < chroma_tol + 2) & (dlight < grow_light_tol) & lim2).astype(np.uint8)
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    loose[edge] = 0                      # closing must not bridge an edge
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.maximum(loose, seed), 4)
    inbox = np.zeros((H, W), bool); inbox[by0:by1, bx0:bx1] = True
    # ---- components: the one with most seed pixels inside the box is the surface; others join only if they look like it
    cands = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 0.002 * W * H:
            continue
        comp = labels == i
        seeds_in = int((seed.astype(bool) & comp & inbox).sum())
        cands.append((i, a, seeds_in, comp))
    keep = np.zeros((H, W), np.uint8)
    if cands:
        dom = max(cands, key=lambda c: c[2])
        if dom[2] > 0:
            keep[dom[3]] = 1
            dmed = np.median(lab[dom[3]].reshape(-1, 3), axis=0)
            er = np.ones((5, 5), np.uint8)
            near_dom = cv2.dilate(dom[3].astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
            for i, a, seeds_in, comp in cands:
                if i == dom[0]:
                    continue
                cmed = np.median(lab[comp].reshape(-1, 3), axis=0)
                if np.hypot(cmed[1] - dmed[1], cmed[2] - dmed[2]) > sim_chroma:
                    continue
                # a piece that touches the surface and is DARKER (the same wall under a soffit shadow) joins without
                # seeds; anything lighter (ceiling, window view) or detached must match closely and be seeded
                touching = bool((near_dom & comp).any())
                darker_by = dmed[0] - cmed[0]
                if touching and 0 <= darker_by <= grow_light_tol:
                    pass
                elif seeds_in == 0 or seeds_in < 0.5 * (seed.astype(bool) & comp).sum() or abs(darker_by) > (sim_light if not touching else 6.0):
                    continue
                inner = cv2.erode(comp.astype(np.uint8), er).astype(bool)
                if inner.sum() and (edge & inner).sum() / inner.sum() > max_inner_edge:
                    continue
                keep[comp] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    keep[edge & ~cv2.dilate(keep, np.ones((3, 3), np.uint8)).astype(bool)] = 0
    # the barrier strip itself belongs to whichever side it matches: grow back over adjacent pixels of the surface
    # colour so the mask reaches the true boundary (wall|ceiling) instead of stopping a few pixels short of it
    colour_ok = (dchroma < chroma_tol + 2) & (dlight < grow_light_tol) & lim2
    reach = cv2.dilate(keep, np.ones((int(1.5 * k) | 1, int(1.5 * k) | 1), np.uint8)).astype(bool)
    keep[colour_ok & reach] = 1
    # ---- holes: fill tiny ones (sockets) and ones that share the surface colour (soft shadows); keep artwork/windows out
    keep = _fill_holes(keep, dchroma, dlight, chroma_tol, grow_light_tol)
    if keep.sum() < 0.1 * max(1, (bx1 - bx0) * (by1 - by0)):
        keep[:] = 0; keep[by0:by1, bx0:bx1] = 1
    return keep.astype(np.float32)
