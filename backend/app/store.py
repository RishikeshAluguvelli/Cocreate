"""Tiny persistent store: shoots are JSON documents, images live on disk under data/shoots/<shoot>/."""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

from .config import settings
from .models import Photo, Shoot, Version

_lock = threading.RLock()
_shoots: dict[str, Shoot] = {}
SHOOTS_DIR = settings.data_dir / "shoots"
SHOOTS_DIR.mkdir(parents=True, exist_ok=True)


def shoot_dir(shoot_id: str) -> Path:
    d = SHOOTS_DIR / shoot_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def photo_dir(photo: Photo) -> Path:
    d = shoot_dir(photo.shoot_id) / photo.id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _doc_path(shoot_id: str) -> Path:
    return shoot_dir(shoot_id) / "shoot.json"


def save(shoot: Shoot) -> None:
    with _lock:
        _shoots[shoot.id] = shoot
        _doc_path(shoot.id).write_text(shoot.model_dump_json(indent=1))


def get(shoot_id: str) -> Shoot | None:
    with _lock:
        if shoot_id in _shoots:
            return _shoots[shoot_id]
        p = _doc_path(shoot_id)
        if p.exists():
            s = Shoot.model_validate_json(p.read_text())
            _shoots[s.id] = s
            return s
    return None


def list_shoots() -> list[Shoot]:
    with _lock:
        for p in SHOOTS_DIR.glob("*/shoot.json"):
            sid = p.parent.name
            if sid not in _shoots:
                try:
                    _shoots[sid] = Shoot.model_validate_json(p.read_text())
                except Exception:
                    continue
        return sorted(_shoots.values(), key=lambda s: s.created_at, reverse=True)


def delete(shoot_id: str) -> None:
    with _lock:
        _shoots.pop(shoot_id, None)
        shutil.rmtree(SHOOTS_DIR / shoot_id, ignore_errors=True)


def add_version(shoot: Shoot, photo: Photo, path: Path, tool: str, params: dict, summary: str,
                mask_path: Path | None = None, cost_usd: float = 0.0, parent: Version | None = None) -> Version:
    """Register a new edited image as the photo's head version."""
    parent = parent or photo.head
    v = Version(photo_id=photo.id, parent_id=parent.id, index=len(photo.versions), path=str(path),
                mask_path=str(mask_path) if mask_path else None, tool=tool, params=params, summary=summary,
                cost_usd=cost_usd)
    photo.versions.append(v)
    photo.head_version_id = v.id
    shoot.total_cost_usd += cost_usd
    save(shoot)
    return v


def rel_url(path: str | Path) -> str:
    """Public URL for a file under data/ (served by FastAPI at /files/...)."""
    p = Path(path).resolve()
    rel = p.relative_to(settings.data_dir.resolve())
    base = settings.public_base_url.rstrip("/")
    return f"{base}/files/{rel.as_posix()}"
