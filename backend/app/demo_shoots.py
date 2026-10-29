"""Demo shoots assembled from AutoHDR's real customer-request dataset (customer-requests.json).
Images are downloaded on first use and cached under data/demo_cache."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image

from . import store
from .config import settings
from .models import Photo, Shoot, Version
from .tools import imaging

DATASET = settings.data_dir / "customer-requests.json"
CACHE = settings.data_dir / "demo_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# (template id, title, blurb, 1-based record numbers from the dataset)
TEMPLATES: list[tuple[str, str, str, list[int]]] = [
    ("sampler", "Sampler · one of everything",
     "Wall colour from a reference, a hex floor request, aerials for lot lines, cars, a lockbox, a lamp, a TV reference, a blurry photo.",
     [22, 157, 137, 7, 198, 151, 101, 2, 182]),
    ("vehicles", "Exteriors · remove all the vehicles", "Six exteriors; the customer asked to remove every vehicle.", [16, 17, 18, 19, 20, 21]),
    ("bins", "Aerials · remove the garbage bins", "Five drone shots with wheelie bins to remove.", [137, 138, 139, 140, 141]),
    ("bedding", "Bedrooms · iron the bedspread", "Seven bedrooms; make the bedspread look ironed without changing the sheets.", [32, 33, 34, 35, 36, 37, 38]),
    ("tv_lamps", "Living rooms · TV reference + lamps on", "Four rooms with a customer reference to put on the TV, lamps to turn on.", [120, 121, 122, 123]),
    ("lawns", "Exteriors · regenerate the lawn", "Six exteriors with a reference lawn; replace patchy grass.", [86, 87, 88, 89, 90, 91]),
    ("fans", "Interiors · turn on the ceiling fan lights", "Three interiors.", [111, 112, 113]),
    ("cords", "Interiors · remove loose cords", "Lamp cords and plugs to remove with a precise mask.", [72, 73, 74]),
]


def _records() -> list[dict]:
    return json.loads(DATASET.read_text())["records"]


def list_templates() -> list[dict]:
    recs = _records()
    out = []
    for tid, title, blurb, nums in TEMPLATES:
        out.append({"id": tid, "title": title, "blurb": blurb, "count": len(nums),
                    "cover": recs[nums[0] - 1]["edited_photo"],
                    "requests": sorted({recs[n - 1]["request"][:140] for n in nums})[:3]})
    return out


async def _fetch(url: str) -> Path:
    name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg"
    p = CACHE / name
    if p.exists() and p.stat().st_size > 1000:
        return p
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


async def create_demo_shoot(template_id: str, owner_id: str | None = None) -> Shoot:
    tpl = next((t for t in TEMPLATES if t[0] == template_id), None)
    if not tpl:
        raise ValueError("unknown template")
    _, title, _, nums = tpl
    recs = _records()
    shoot = Shoot(name=title, owner_id=owner_id)
    sem = asyncio.Semaphore(6)

    async def build(i: int, n: int) -> Photo:
        r = recs[n - 1]
        async with sem:
            edited = await _fetch(r["edited_photo"])
            brackets = await asyncio.gather(*(_fetch(u) for u in r["inputs"]))
            refs = await asyncio.gather(*(_fetch(u) for u in r.get("request_references", [])))
        with Image.open(edited) as im:
            w, h = im.size
        photo = Photo(shoot_id=shoot.id, name=f"Photo {i + 1}", width=w, height=h, original_path=str(edited),
                      bracket_paths=[str(b) for b in brackets], reference_paths=[str(x) for x in refs],
                      dataset_request=r["request"].strip())
        pdir = store.photo_dir(photo)
        v0 = pdir / "v00_original.jpg"
        v0.write_bytes(edited.read_bytes())
        ver = Version(photo_id=photo.id, index=0, path=str(v0))
        photo.versions = [ver]
        photo.head_version_id = ver.id
        return photo

    shoot.photos = list(await asyncio.gather(*(build(i, n) for i, n in enumerate(nums))))
    store.save(shoot)
    return shoot


def create_shoot_from_files(name: str, files: list[tuple[str, bytes]], brackets: dict[str, list[bytes]] | None = None,
                            references: list[bytes] | None = None, owner_id: str | None = None) -> Shoot:
    shoot = Shoot(name=name or "Uploaded shoot", owner_id=owner_id)
    add_photos_to_shoot(shoot, files, brackets=brackets, references=references)
    return shoot


def add_photos_to_shoot(shoot: Shoot, files: list[tuple[str, bytes]], brackets: dict[str, list[bytes]] | None = None,
                        references: list[bytes] | None = None) -> Shoot:
    base = len(shoot.photos)
    for k, (fname, data) in enumerate(files):
        i = base + k
        photo = Photo(shoot_id=shoot.id, name=Path(fname).stem[:40] or f"Photo {i + 1}", width=1, height=1, original_path="")
        pdir = store.photo_dir(photo)
        src = pdir / "source" / Path(fname).name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(data)
        im = imaging.load_pil(src)
        # cap very large uploads for speed (long side 2400)
        if max(im.size) > 2400:
            im.thumbnail((2400, 2400), Image.LANCZOS)
        v0 = pdir / "v00_original.jpg"
        im.save(v0, "JPEG", quality=94, subsampling=0)
        photo.width, photo.height = im.size
        photo.original_path = str(v0)
        for j, b in enumerate((brackets or {}).get(fname, [])):
            bp = pdir / "source" / f"bracket_{j}.jpg"
            bp.write_bytes(b)
            photo.bracket_paths.append(str(bp))
        for j, b in enumerate(references or []):
            rp = pdir / "source" / f"reference_{j}.jpg"
            rp.write_bytes(b)
            photo.reference_paths.append(str(rp))
        ver = Version(photo_id=photo.id, index=0, path=str(v0))
        photo.versions = [ver]
        photo.head_version_id = ver.id
        shoot.photos.append(photo)
    store.save(shoot)
    return shoot
