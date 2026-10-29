"""Run the four deterministic tools on real sample photos and write previews for visual inspection.
usage: uv run python tests/smoke_deterministic.py <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.tools import brightness, color, composite, imaging, lot_lines, watermark  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
S = ROOT / "data" / "samples"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "smoke_out")
OUT.mkdir(parents=True, exist_ok=True)


def t(label, fn):
    t0 = time.time()
    r = fn()
    print(f"{label:28s} {time.time() - t0:5.2f}s")
    return r


# 1) brightness slider on a dim lamp photo (r151) + shadows lift
img = imaging.load_rgb(S / "r151_edited.jpg")
out = t("brightness +40", lambda: brightness.adjust(img, brightness=40))
imaging.save_rgb(out, OUT / "01_brightness_plus40.jpg")
out = t("brightness -40", lambda: brightness.adjust(img, brightness=-40))
imaging.save_rgb(out, OUT / "01_brightness_minus40.jpg")
out = t("shadows +60 / warm +30", lambda: brightness.adjust(img, shadows=60, temperature=30))
imaging.save_rgb(out, OUT / "01_shadows_warm.jpg")
# regional: brighten only the left half (stand-in for a SAM region)
m = np.zeros(img.shape[:2], np.float32); m[:, : img.shape[1] // 2] = 1
out = t("regional brightness", lambda: brightness.adjust(img, brightness=60, region_mask=m))
imaging.save_rgb(out, OUT / "01_regional_left_half.jpg")

# 2) watermark text + tile on an exterior (r007)
img = imaging.load_rgb(S / "r007_edited.jpg")
out = t("watermark text br", lambda: watermark.add_watermark(img, text="© Skyline Realty Photo", position="bottom_right", opacity=0.7, scale=0.28))
imaging.save_rgb(out, OUT / "02_watermark_text.jpg")
out = t("watermark tile", lambda: watermark.add_watermark(img, text="PREVIEW", position="tile", opacity=0.18, scale=0.18))
imaging.save_rgb(out, OUT / "02_watermark_tile.jpg")

# 3) lot lines on the aerial (r137): rough parcel around the white 3-storey house
img = imaging.load_rgb(S / "r137_edited.jpg")
pts = [[0.235, 0.14], [0.62, 0.14], [0.62, 0.93], [0.21, 0.93]]
out = t("lot lines dashed", lambda: lot_lines.draw_lot_lines(img, pts, style="dashed", label="LOT 12 · 0.18 ac"))
imaging.save_rgb(out, OUT / "03_lot_lines_dashed.jpg")
out = t("lot lines solid red", lambda: lot_lines.draw_lot_lines(img, pts, color="#FF3B30", style="solid", fill_opacity=0.0, corner_markers=False))
imaging.save_rgb(out, OUT / "03_lot_lines_solid.jpg")

# 4) colour: sample the teal wall from the customer reference (r022_ref0), repaint the grey accent wall (r022_edited)
ref = imaging.load_rgb(S / "r022_ref0.jpg")
lab = color.rgb_to_lab(ref)
# teal wall in the reference: moderate L, negative a (greenish), negative b (bluish) -> crude stand-in for a SAM 'wall' mask
ref_mask = ((lab[..., 1] < -5) & (lab[..., 2] < -5) & (lab[..., 0] > 20) & (lab[..., 0] < 60)).astype(np.float32)
ref_mask = composite.erode(ref_mask, 4)
sampled = t("sample colour (ref)", lambda: color.sample_color(ref, ref_mask))
print("   sampled:", sampled)
imaging.save_rgb(composite.overlay_preview(ref, ref_mask), OUT / "04a_ref_mask_preview.jpg")

img = imaging.load_rgb(S / "r022_edited.jpg")
lab = color.rgb_to_lab(img)
# dark grey accent wall in the finished photo: low L, low chroma, left 30% and right 10% of frame
chroma = np.hypot(lab[..., 1], lab[..., 2])
wall = ((lab[..., 0] < 45) & (chroma < 8)).astype(np.float32)
W = img.shape[1]
region = np.zeros_like(wall); region[:, : int(0.30 * W)] = 1; region[:, int(0.89 * W):] = 1
wall = composite.erode(wall * region, 3)
# remove the TV (very dark) from the wall mask
wall[lab[..., 0] < 12] = 0
wall = composite.dilate(composite.erode(wall, 2), 2)
print(f"   wall mask area: {composite.mask_area_fraction(wall):.3f}")
imaging.save_rgb(composite.overlay_preview(img, wall), OUT / "04b_wall_mask_preview.jpg")
out = t("recolor wall -> sampled", lambda: color.recolor(img, wall, sampled["hex"]))
imaging.save_rgb(out, OUT / "04c_wall_repainted.jpg")
out2 = t("recolor wall -> #C8C2B7", lambda: color.recolor(img, wall, "#C8C2B7"))
imaging.save_rgb(out2, OUT / "04d_wall_repainted_beige.jpg")

# 5) composite sanity: edited == original outside the mask
soft = composite.feather(wall, 6)
comp = composite.composite(img, out, soft)
outside = (soft < 0.001)
print("   max diff outside mask:", float(np.abs(comp - img)[outside].max()))
print("done ->", OUT)
