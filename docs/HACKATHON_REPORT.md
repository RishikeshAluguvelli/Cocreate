# Cocreate · AutoHDR hackathon, Track 3 — requirements check and demo script

## What the brief asks for (quoted from the Track 3 page)

> Build a chat interface where they can describe what they want and an agent makes the right edits across a photo shoot.

Tools the agent should choose from: brightness adjustments "like a customer would make with a slider"; checking the
original photo for a wall's paint colour, extracting its hex code and repainting the wall; drawing lot lines; adding a
watermark; removing an object with a precise mask.

Rules: "Edits apply to the finished photo" (brackets are references for colours, details and scene understanding);
"Keep unrelated parts of each photo intact"; customers can also make manual adjustments in the editor.

Judging: "the quality of the edits", "how well your agent understands the request", "how useful and easy the interface
feels". Deliverable: a deployed demo that accepts chat requests, runs the edits and shows the results, submitted through
the Google Form with instructions for trying it.

## Requirement by requirement

| Brief | Status | Where / evidence |
|---|---|---|
| Chat interface, agent edits across a shoot | Done | Whole-shoot scope: the planner sees every photo, edits the ones the request applies to and reports what it skipped; "Only this photo" scope for single edits |
| Brightness like a slider | Done | `adjust` tool (brightness, contrast, saturation, warmth, shadows, highlights, sharpen); the same maths runs live in the browser sliders; a chat adjustment moves the sliders and can be fine-tuned by hand |
| Wall colour from the original, hex, repaint | Done | `inspect_photo` + `sample_color` (exposure-matched samples from brackets or references, candidate list) + `repaint` (LAB chroma swap keeping texture and shading, edge-aware mask, detected windows/furniture carved out, vision mask review, QC verdict) |
| Lot lines | Done | `draw_lot_lines` from points the user clicks on the photo |
| Watermark | Done | `add_watermark` with text or uploaded logo, plus the "Digitally altered" disclosure preset |
| Remove an object with a precise mask | Done | Detection boxes (tiled on aerials) → GrabCut + contact-shadow extension → classical pre-erase → Nano Banana repair per close-up window; large areas run two strategies and QC keeps the better |
| Edits apply to the finished photo, brackets as references | Done | Every version derives from the finished photo; brackets and references are only read for colours and scene checks |
| Keep unrelated parts intact | Done | Every model result is pasted back only inside a feathered mask; the **Changes** view shows every pixel that differs; QC scores collateral change after each generative edit |
| Manual adjustments in the editor | Done | Live sliders with before/after compare, Apply as a new version, Undo, revert to any version |
| Deployed demo with try-it instructions | Ready to deploy | Dockerfile + `render.yaml`; accounts with a one-click guest workspace so judges need no sign-up; README has run and try instructions |

Beyond the brief: lamps/fire/TV on, bedding, lawn, sky and twilight through a protected generative edit with no-op
detection and retry; pixel-exact TV screen replacement from the reference; blur faces/plates; match look across a
shoot; crop/straighten; Plan-first mode; streamed status and intent; QC badges; per-photo edit log and long-chat memory;
archived chats; tenant isolation with accounts, guests and password reset.

## Judging criteria (as published)

| Criterion | How Cocreate answers it | Show it |
|---|---|---|
| **Completeness**: core workflow without crashing | Every model call has retries and fallbacks; a refused or unchanged edit is reported instead of failing; per-shoot queueing; a React error boundary; 41 + 12 + 25 automated checks (auth, concurrency, guest mode) plus the 200-request eval | Run a full request live; mention the test suites |
| **Technical depth**: a pipeline, not a wrapper | Planner with 14 tools, detection → GrabCut / edge-aware colour growth → classical pre-erase → masked generative repair → QC verdict; two fill strategies for large areas; vision mask review; pixel-exact TV warp; brackets exposure-matched for colour sampling | Open the Mask and Changes views; `docs/ARCHITECTURE.md` |
| **Insight quality**: non-obvious, useful output | QC scores per edit, the changed-pixel map that proves nothing else moved, candidates when a colour is ambiguous, the per-photo edit log, the HTML report | Export the Report for a finished shoot |
| **Usability**: act on it tomorrow | Guest workspace with no sign-up, the sign-in card right on the home page (sign-up keeps the guest's work), download one photo or the whole shoot, the report to forward to the customer, undo / revert / regenerate, live sliders, Paint to fix a missed spot, the Colour tool (eyedropper from a reference, one click to recolour, instant) | Pick the teal from the customer reference and click the grey wall: recoloured in under a second, then "Ask the assistant instead" for the QC-checked version |
| **Creativity**: novel combination, look and feel | Classical CV + VLM detection + generative repair + VLM QC in one loop; sliders the assistant moves; dark, focused editor UI | The sliders moving by themselves after a chat request |
| **Performance**: speed or scale | Parallel per-photo edits, close-up windows at native resolution instead of full-frame regenerations, CPU work on worker threads, per-shoot locks with cross-shoot parallelism, 1K/2K choice by window size, median 77 s per real request | Two shoots working at once; the elapsed time on every reply |

## Measured quality

- AutoHDR's 200 real customer requests, strict vision judge (before, after, changed-pixel map): **70.5% pass, 10%
  partial, 19.5% fail**; on the 85 records the first version completed: pass 42% → 74%, fail 41% → 18%.
- Ten stock photos not in the dataset, nine whole-shoot requests: 15 pass / 2 partial / 4 fail across 21 edited
  photos; the right photos were picked on 7 of 9 requests. The failures drove the mask redesign and the two-strategy
  removal; the kitchen removal now passes.
- Recolour regression check (2026-10-29, after the texture-preserving recolour and the new surface masks): the 31
  colour and lawn records of the dataset re-run against the v3 baseline: colour 3 → 4 pass (records that used
  `repaint`: 2 → 3), lawn 11 → 9 pass with every change in generative or removal records whose code did not change
  (model variance); overall 14 → 13 pass, 11 → 11 fail. Results: `data/eval/v6recolor/summary.md`.
- Weak spots to say out loud if asked: water colour nudges, subtle lights-on, sun rays, repainting a wall whose ceiling
  or table has the identical colour, the detector's box varying between runs, and the colour tool on a near-white
  surface next to other near-white things (ceiling, bedding): it is a colour method, so it needs the band lowered,
  Object mode, or Paint.
- Tool coverage suite (`backend/tests/tool_coverage.py`, one request per tool plus the direct colour tool, judged):
  **14 of 17 pass** on 2026-10-29 for $0.87. The three misses: the chat repaint and the one-click recolour of a tile
  floor whose far end fades into sunlight (the judge wants the lit end repainted too; Follow lighting in the Colour
  tool reaches it), and a lamp the judge scored "partial" (glow looked artificial). Every deterministic tool, the
  removals, the TV screen, the reference-colour repaint and the eyedropper + apply case pass.

## Deliverables the judges score
Live demo at the table, a **demo video**, and the **codebase**. The video is not generated here: record the demo
script below (OBS or QuickTime, 1080p, 3 to 5 minutes, narrated). Keep the codebase link ready with
`docs/ARCHITECTURE.md` and the README open.

## Demo script (about 6 minutes)

1. **Open the home page** (30 s). You are already in a private guest workspace ("every judge gets one, nothing to
   sign up for"); point at the workspace card: sign up keeps the work, log in opens an account. Then the three steps
   and the two ways to start.
2. **Removal across a shoot** (90 s). Open *Aerials · remove the garbage bins*, click the request chip. While it runs,
   show the streamed status, the per-photo "working" badges and the tool cards filling in. When done: the compare
   slider, then the **Mask** view ("precise mask, contact shadow included"), then **Changes** ("magenta is every pixel
   that moved: the rest is untouched").
3. **Colour from a reference** (90 s). Open the *Sampler*, pick the wall-colour request. Show `sample_color` returning
   the hex and candidates, the repaint with texture and shading kept, and the QC badge. Then type "a bit lighter" to
   show the follow-up understanding and the edit log.
4. **Sliders** (45 s). Type "brighten this photo by 15%" with *Only this photo* selected: the panel opens with the
   values the assistant applied; drag brightness further, live preview under the compare handle, Apply as a new
   version, Undo.
5. **Understanding and skipping** (60 s). In *Living rooms · TV reference + lamps on*, ask "turn on all the lamps and
   put the reference on the TV". Show photos without lamps being skipped with a reason, the perspective-fit TV
   screen, and the turn summary "edited N, unchanged K".
6. **Colour tool and Paint to fix** (75 s). Open **Colour**, click *Reference 1*, click the teal wall: the hex and
   lit / shade variants appear. Switch to *Apply*, click the grey wall: the highlight shows exactly what will change
   (nudge the band or add a second click), Apply: a new version in under a second, Mask view shows the region. Then
   if the highlight took too much or too little, **Add** or **Erase** with the brush before applying. Then open
   Paint, brush the spot the last edit missed, send "also repaint the area I painted": the planner reuses the same
   colour and the mask shows the painted band included. Then **Report** in a new tab.
7. **Plan first and lot lines** (45 s). Tick *Plan first*, click two points on an aerial, ask for lot lines: the plan
   is described, approve, lines drawn; mention the watermark preset and the download of the whole shoot.
8. **Close** (30 s). The numbers above, the eval harness, and the accounts layer: "every shoot belongs to a user, two
   requests on the same shoot are queued, requests on different shoots run in parallel".

## Submission checklist

- Deploy (Render blueprint, `GEMINI_API_KEY` set, `AUTH_MODE=required`, persistent disk if accounts must survive redeploys).
- Open the deployed URL once, create a guest workspace, run one demo request to warm the model cache.
- Submit the link via the Google Form (Track 3) with: "Click *Continue as guest*, open *Aerials · remove the garbage
  bins*, click the request chip above the chat box."
