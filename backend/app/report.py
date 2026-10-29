"""Self-contained HTML report for a shoot: what was asked, what was done to every photo, QC verdicts, before/after.

Something a photographer can forward to the customer tomorrow: one file, images embedded, no server needed to view it.
"""
from __future__ import annotations

import base64
import html
import io
import time
from pathlib import Path

from PIL import Image

from .models import Shoot

TOOL_LABEL = {"inspect_photo": "Inspected", "sample_color": "Sampled colour", "repaint": "Re-painted", "adjust": "Adjusted",
              "draw_lot_lines": "Lot lines", "add_watermark": "Watermarked", "remove_object": "Removed", "generative_edit": "AI edit",
              "tv_screen": "TV screen", "blur": "Blurred", "match_look": "Matched look", "crop_rotate": "Crop / straighten",
              "manual_adjust": "Sliders", "recolor": "Recoloured", "original": "Original"}


def _data_uri(path: str | Path, long_side: int = 900, quality: int = 78) -> str:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return ""
    im.thumbnail((long_side, long_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _lineage(photo) -> list:
    """Head version and its ancestors, oldest first (what actually made the current image)."""
    by_id = {v.id: v for v in photo.versions}
    out, v = [], by_id.get(photo.head_version_id)
    while v is not None:
        out.append(v)
        v = by_id.get(v.parent_id) if v.parent_id else None
    return list(reversed(out))


def build_report_html(shoot: Shoot) -> str:
    e = html.escape
    edits = sum(max(0, len(p.versions) - 1) for p in shoot.photos)
    requests = [m for m in shoot.chat if m.role == "user"]
    seconds = sum((m.seconds or 0) for m in shoot.chat if m.role == "assistant")
    qc_pass = sum(1 for p in shoot.photos for v in p.versions if (v.params or {}).get("qc", {}).get("verdict") == "pass")
    qc_review = sum(1 for p in shoot.photos for v in p.versions if (v.params or {}).get("qc", {}).get("verdict") == "needs_review")

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(shoot.name)} · edit report</title>
<style>
:root{{color-scheme:light dark}}body{{font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;background:#0c0c0e;color:#e4e4e7}}
main{{max-width:1100px;margin:0 auto;padding:32px 20px}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:17px;margin:36px 0 10px;border-top:1px solid #27272a;padding-top:18px}}
.muted{{color:#a1a1aa}}.stats{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}.stat{{border:1px solid #27272a;border-radius:10px;padding:8px 12px;background:#18181b}}
.stat b{{display:block;font-size:18px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}}.pair img{{width:100%;border-radius:8px;display:block}}
.cap{{font-size:12px;color:#a1a1aa;margin-top:4px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid #27272a;vertical-align:top}}
.pass{{color:#6ee7b7}}.review{{color:#fcd34d}}.chat{{border:1px solid #27272a;border-radius:10px;padding:10px 14px;margin:8px 0;background:#18181b}}
.chat.user{{border-color:#f59e0b55}}.who{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#a1a1aa}}@media print{{body{{background:#fff;color:#111}}.stat,.chat{{background:#fff}}}}
</style></head><body><main>
<div class="muted" style="letter-spacing:.1em;text-transform:uppercase;font-size:11px">Cocreate · edit report</div>
<h1>{e(shoot.name)}</h1>
<div class="muted">Generated {time.strftime('%Y-%m-%d %H:%M')} · shoot created {time.strftime('%Y-%m-%d', time.localtime(shoot.created_at))}</div>
<div class="stats">
<div class="stat"><b>{len(shoot.photos)}</b>photos</div><div class="stat"><b>{len(requests)}</b>requests</div><div class="stat"><b>{edits}</b>edits</div>
<div class="stat"><b>{qc_pass}</b>QC pass</div><div class="stat"><b>{qc_review}</b>QC flagged</div>
<div class="stat"><b>{seconds/60:.1f} min</b>processing</div><div class="stat"><b>${shoot.total_cost_usd:.2f}</b>model cost</div></div>
<p class="muted">Every edit was pasted back only inside its mask; everything outside the request is pixel-identical to the finished photo.
QC is an automated vision check of each generative edit (fulfilled / collateral / artifacts).</p>"""]

    for i, p in enumerate(shoot.photos, 1):
        head = p.head
        lineage = _lineage(p)
        parts.append(f'<h2>{i}. {e(p.name)} <span class="muted" style="font-weight:normal">· {p.width}×{p.height} · v{head.index}</span></h2>')
        if head.index > 0:
            parts.append(f'<div class="pair"><div><img src="{_data_uri(p.original_path)}" alt="before"><div class="cap">Before (finished photo)</div></div>'
                         f'<div><img src="{_data_uri(head.path)}" alt="after"><div class="cap">After (v{head.index})</div></div></div>')
            parts.append('<table><tr><th>Step</th><th>Tool</th><th>What changed</th><th>QC</th></tr>')
            for v in lineage[1:]:
                qc = (v.params or {}).get("qc") or {}
                verdict = qc.get("verdict")
                qcell = (f'<span class="{"pass" if verdict == "pass" else "review"}">{e(verdict.replace("_", " "))}</span>'
                         + (f'<div class="cap">{e(str(qc.get("reason", ""))[:160])}</div>' if qc.get("reason") else "")) if verdict in ("pass", "needs_review") else '<span class="muted">—</span>'
                parts.append(f'<tr><td>v{v.index}</td><td>{e(TOOL_LABEL.get(v.tool, v.tool))}</td><td>{e(v.summary)}</td><td>{qcell}</td></tr>')
            parts.append('</table>')
        else:
            parts.append(f'<div class="pair"><div><img src="{_data_uri(p.original_path)}" alt="photo"><div class="cap">Unchanged</div></div></div>')

    if shoot.chat:
        parts.append('<h2>Conversation</h2>')
        for m in shoot.chat:
            who = "Customer" if m.role == "user" else "Assistant"
            extra = f' · {m.seconds:.0f} s' if m.role == "assistant" and m.seconds else ""
            tools = [ev.get("name") for ev in (m.events or []) if ev.get("type") == "tool_start"]
            tl = f'<div class="cap">tools: {e(", ".join(TOOL_LABEL.get(t, t) for t in tools))}</div>' if tools else ""
            parts.append(f'<div class="chat {m.role}"><div class="who">{who}{extra}</div><div>{e(m.text or "(no reply)").replace(chr(10), "<br>")}</div>{tl}</div>')
    parts.append('</main></body></html>')
    return "".join(parts)
