# Architecture

```
browser (React + Vite)                      FastAPI (uvicorn, asyncio)                          models (one Gemini key)
┌──────────────────────────┐   SSE   ┌──────────────────────────────────────────┐        ┌──────────────────────────┐
│ Home · Auth · Viewer     │◄───────►│ /api/shoots/{id}/chat  → agent.loop      │───────►│ gemini-3.8-flash         │
│ Gallery · Chat           │  fetch  │   per-shoot lock · status/thinking events │ plan   │   planner (function calls) │
│ per-shoot job state      │         │   gemini_planner → tools_schema (14 tools)│        │   QC verdict · judge       │
│ live LUT sliders         │         │   executor.dispatch → tools/*             │───────►│ gemini-3.5-flash-lite    │
│ paint / points overlays  │         │ store (JSON per shoot) · auth (SQLite)    │ detect │   boxes · mask review      │
└──────────────────────────┘         │ /files/* with ownership check            │───────►│ gemini-nano-banana-2.1   │
                                     └──────────────────────────────────────────┘ edit   │   masked generative edits  │
                                                                                          └──────────────────────────┘
```

## Request lifecycle

1. The browser posts the message, scope (whole shoot / one photo), clicked points and any painted mask to the chat
   endpoint and reads a Server-Sent-Events stream (`status`, `thinking`, `text`, `tool_start`, `progress`, `version`,
   `skipped`, `tool_result`, `error`, `done`). All UI state for the request lives under `jobs[shootId]`.
2. `agent/loop.run_turn` takes the shoot's lock (requests on other shoots run in parallel), builds the state the planner
   sees (`_shoot_state`: every photo with its edit log, scope rule, conversation memory, points, painted area), and
   runs the planner round-trip loop.
3. `agent/gemini_planner` streams the model's reply; each function call goes to `agent/executor.dispatch`, which runs
   the tool, emits events, and returns a JSON result (plus an image block for `inspect_photo`).
4. Tools write a new `Version` (JPEG + mask + mask preview) through `store.add_version`; the head moves; nothing is
   ever overwritten, so undo / revert / regenerate are free.
5. When the turn ends the two chat messages are appended, old transcript turns are folded into `memory_notes`, the
   shoot JSON is saved, and `done` is sent.

## Tool pipelines

| Tool | Pipeline |
|---|---|
| `remove_object` | detection boxes (tiled 2×2 on aerials) → when tiles were used or four or more instances were found, a vision review of the overlay drops instances that are not the target (a window or shed labelled as a bin) → GrabCut per box + contact-shadow extension → classical pre-erase (Telea) → Nano Banana repairs each close-up window with the mask as a second image → composite through a feathered mask → QC. Masks over 18 % of the frame also run a semantic full-frame removal; QC keeps the better. An explicit `erase` uses classical inpainting only below 0.8 % of the frame; larger ones take the repair path. |
| `repaint` | surface detection → colour-region growth bounded by step edges (directional line-vs-step test; on textured surfaces such as lawn or gravel the growth runs on a blurred copy with coarser edge thresholds so sod seams do not fragment it) → detected windows / curtains / furniture / artwork carved out → vision review of the overlay carves what it names → LAB chroma swap keeping texture and shading → QC. |
| `generative_edit` | protect mask (object detection or painted area) → close-up window around it → Nano Banana with the instruction and references → no-op detection and strong retry → composite inside the mask → QC. |
| `adjust` | linear-light exposure, S-curve contrast, chroma scale, white-balance shift, shadow/highlight lift, unsharp; optional region mask. The browser runs the same maths for the live sliders. |
| `tv_screen` | panel detection → screen quad (Otsu + polygon, accepted only when convex, landscape and of a plausible size for the TV box; otherwise the dark panel's bounding box, otherwise the inset box) → perspective-warp the reference, pixel exact. |
| `sample_color` | median LAB inside the detected surface, exposure-matched between brackets and the finished photo; distinct candidates per instance. |
| `draw_lot_lines`, `add_watermark`, `blur`, `match_look`, `crop_rotate` | deterministic OpenCV / PIL. |
| Colour tool (`/color-at`, `/region-at`, `/apply-color`, no planner) | `masks.region_at_point`: colour growth from the clicked pixel bounded by the same step-edge barrier as `surface_refine`, lightness compared to a least-squares shading plane fitted on what was found (follows window light and vignetting), finer edge test and narrower band for near-neutral surfaces, blurred working image for textured ones (lawn, gravel); `object` mode runs GrabCut with the grown pixels near the click as foreground inside a window around the click, limited to a wide colour band; several clicks are unioned, and hand-drawn Add / Erase strokes from the editor (PNG layers) are unioned with or cut out of the region before recolouring. `pick.pick_color` samples the median of the grown region's mid-lit pixels (plus lit / shade variants); `color.recolor` keeps a fraction of the surface's own chroma variation, desaturates specular highlights and compresses bright deviations for light targets instead of clipping. |

Every model output is composited back **only inside its mask**; the Changes view shows the exact pixels that differ.

Points the user places on a wall / floor / lawn seed the same `region_at_point` growth when the planner calls
`sample_color` or `repaint` with `points`, so a marked surface is grown from the exact spot instead of a detection box.

## Quality control

`tools/qc.py` sends before, after and a magenta changed-pixel overlay to the planner model and gets fulfilled / collateral
/ artifacts scores and a verdict. The planner retries a flagged photo once and keeps the better attempt. The same judge
powers `tests/eval_dataset.py` (200 real customer requests), `tests/smoke_new_photos.py` (any folder of photos) and
`tests/tool_coverage.py` (one request per tool, plus the direct colour tool).

## Isolation and accounts

- `auth.py`: SQLite users and sessions, scrypt hashes, HttpOnly cookie, throttled logins, single-use reset tokens, guest
  accounts upgradeable to real ones. `AUTH_MODE` = `open` (single user), `guest` (everyone gets a private workspace,
  sign-up keeps it), `required` (login wall).
- Every shoot has an `owner_id`; every route and every `/files/...` URL checks it. Other people's shoots answer 404.
- Per-shoot `asyncio.Lock` serialises requests on one shoot; different shoots run concurrently; CPU-bound steps run in
  `asyncio.to_thread`; the edit semaphore (`EDIT_CONCURRENCY`) caps concurrent model edits per process.
- Per-photo locks inside one turn: the planner runs independent tool calls in parallel, and two calls that edit the
  same photo (reference on the TV, lamps on) would otherwise both start from the current version and the second one
  saved would discard the first one's pixels. Every editing call holds a lock on each photo it touches, taken in a fixed
  order; calls on disjoint photos still run side by side (`tests/test_photo_lock.py`).

## Storage

`data/shoots/<id>/shoot.json` plus one folder per photo with `v00_original.jpg`, `vNN_<tool>.jpg`, `vNN_mask.png`,
`vNN_maskpreview.jpg`; `data/users.sqlite`; `data/demo_cache` for dataset downloads; `data/eval/<run>` for harness
results.
