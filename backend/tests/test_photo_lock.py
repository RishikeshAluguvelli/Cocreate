"""Two planner tool calls that edit the same photo at the same time must not lose each other's pixels.

Runs fully offline (COCREATE_MOCK=1 is set by the test itself): `adjust` and `add_watermark` are dispatched concurrently on
one photo; afterwards the photo must have three versions in a chain (v2's parent is v1) and the final image must carry
BOTH edits. Without the per-photo lock the second tool to finish starts from the original and the first edit vanishes.

    cd backend && uv run python tests/test_photo_lock.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ["COCREATE_MOCK"] = "1"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="cc-lock-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app.agent.executor import dispatch  # noqa: E402
from app.demo_shoots import create_shoot_from_files  # noqa: E402
from app.tools import imaging  # noqa: E402

FAILS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILS
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' · ' + detail) if detail else ''}")
    FAILS += 0 if ok else 1


async def noop(_etype: str, _data: dict) -> None:
    return None


def make_photo() -> bytes:
    import io
    im = Image.new("RGB", (900, 600), (120, 110, 100))
    d = ImageDraw.Draw(im)
    d.rectangle([100, 100, 500, 400], fill=(60, 90, 140))
    d.ellipse([600, 150, 850, 450], fill=(200, 180, 60))
    b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return b.getvalue()


async def main() -> None:
    shoot = create_shoot_from_files("lock test", [("room.jpg", make_photo())])
    photo = shoot.photos[0]
    original = imaging.load_rgb(photo.original_path)

    print("concurrent adjust + watermark on one photo")
    r1, r2 = await asyncio.gather(
        dispatch(shoot, noop, "adjust", {"photo_ids": [photo.id], "brightness": 40}),
        dispatch(shoot, noop, "add_watermark", {"photo_ids": [photo.id], "text": "LOCK TEST", "position": "bottom_right", "opacity": 1.0, "scale": 0.35}),
    )
    check("both tools returned results", bool(r1[0]["results"]) and bool(r2[0]["results"]))
    check("three versions exist", len(photo.versions) == 3, f"{len(photo.versions)}")
    v1, v2 = photo.versions[1], photo.versions[2]
    check("versions form a chain (v2's parent is v1)", v2.parent_id == v1.id and v1.parent_id == photo.versions[0].id)
    check("head is the last version", photo.head.id == v2.id)
    final = imaging.load_rgb(photo.head.path)
    brighter = float(final[:300, :300].mean()) > float(original[:300, :300].mean()) + 0.05
    check("final image carries the brightness change (top-left region brighter)", brighter, f"{original[:300, :300].mean():.3f} -> {final[:300, :300].mean():.3f}")
    # the watermark sits bottom-right: that corner must differ from a brightness-only render of the original
    from app.tools import brightness
    bright_only = brightness.adjust(original, brightness=40)
    corner = (slice(450, 600), slice(550, 900))
    wm_diff = float(np.abs(final[corner] - bright_only[corner]).max(axis=2).mean())
    check("final image carries the watermark too (bottom-right differs from brightness-only)", wm_diff > 0.02, f"mean diff {wm_diff:.3f}")

    print("concurrent calls on DIFFERENT photos still run (no deadlock, both edited)")
    shoot2 = create_shoot_from_files("lock test 2", [("a.jpg", make_photo()), ("b.jpg", make_photo())])
    pa, pb = shoot2.photos
    await asyncio.wait_for(asyncio.gather(
        dispatch(shoot2, noop, "adjust", {"photo_ids": [pa.id], "brightness": 30}),
        dispatch(shoot2, noop, "add_watermark", {"photo_ids": [pb.id], "text": "B"}),
        dispatch(shoot2, noop, "adjust", {"photo_ids": ["all"], "contrast": 20}),
    ), timeout=60)
    check("photo A: 3 versions (adjust + all-photos contrast)", len(pa.versions) == 3, f"{len(pa.versions)}")
    check("photo B: 3 versions (watermark + all-photos contrast)", len(pb.versions) == 3, f"{len(pb.versions)}")
    check("every chain is linear", all(v.parent_id == p.versions[i - 1].id for p in (pa, pb) for i, v in enumerate(p.versions) if i))

    print(f"\n{'ALL PASSED' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    asyncio.run(main())
