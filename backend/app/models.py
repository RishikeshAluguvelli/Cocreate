"""Pydantic models shared by the store, the agent and the HTTP API."""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Version(BaseModel):
    id: str = Field(default_factory=lambda: new_id("v"))
    photo_id: str
    parent_id: str | None = None
    index: int = 0                       # 0 = original finished photo
    path: str                            # absolute path on disk (JPEG)
    mask_path: str | None = None         # mask used for this edit, if any
    tool: str = "original"
    params: dict[str, Any] = Field(default_factory=dict)
    summary: str = "Original finished photo"
    created_at: float = Field(default_factory=time.time)
    cost_usd: float = 0.0


class Photo(BaseModel):
    id: str = Field(default_factory=lambda: new_id("p"))
    shoot_id: str
    name: str
    width: int
    height: int
    original_path: str                   # the finished photo the request applies to
    bracket_paths: list[str] = Field(default_factory=list)    # original exposures (references)
    reference_paths: list[str] = Field(default_factory=list)  # customer-supplied references
    versions: list[Version] = Field(default_factory=list)
    head_version_id: str | None = None
    dataset_request: str | None = None   # the real customer request this photo came with (demo only)
    caption: str | None = None

    @property
    def head(self) -> Version:
        for v in self.versions:
            if v.id == self.head_version_id:
                return v
        return self.versions[0]

    def version(self, vid: str) -> Version | None:
        return next((v for v in self.versions if v.id == vid), None)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    created_at: float = Field(default_factory=time.time)
    events: list[dict[str, Any]] = Field(default_factory=list)  # tool cards rendered with the message
    thinking: str = ""                                          # planner's thought summary (assistant turns)
    scope: dict[str, Any] | None = None                          # user turns: {"mode": "shoot"|"photo", "photo_name": ...}
    status_log: list[str] = Field(default_factory=list)         # progress lines shown while the turn ran
    seconds: float | None = None


class Shoot(BaseModel):
    id: str = Field(default_factory=lambda: new_id("s"))
    name: str
    photos: list[Photo] = Field(default_factory=list)
    chat: list[ChatMessage] = Field(default_factory=list)
    llm_messages: list[dict[str, Any]] = Field(default_factory=list)  # planner transcript (text only)
    llm_provider: str = ""  # anthropic | gemini ; transcript is reset when the provider changes
    memory_notes: str = ""  # rolling summary of older conversation turns (kept when the transcript is trimmed)
    archived_chats: list[dict[str, Any]] = Field(default_factory=list)  # earlier chat sessions on this shoot
    watermark_logo_path: str | None = None
    owner_id: str | None = None  # account that owns this shoot (None = created in open/single-user mode)
    created_at: float = Field(default_factory=time.time)
    total_cost_usd: float = 0.0

    def photo(self, pid: str) -> Photo | None:
        return next((p for p in self.photos if p.id == pid), None)


# ---- API payloads ----
class ToolEvent(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
