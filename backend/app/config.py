"""Runtime settings. Everything comes from environment variables (or a .env file)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
APP_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM planner (Anthropic) ---
    anthropic_api_key: str | None = None
    planner_model: str = "claude-opus-5-5"
    planner_effort: str = "medium"  # low | medium | high
    planner_max_tokens: int = 16000

    # --- Google Gemini (planner + detection + Nano Banana edits; one key) ---
    gemini_api_key: str | None = None
    gemini_planner_model: str = "gemini-3.8-flash"
    gemini_planner_thinking: str = "low"          # minimal | low | medium | high
    gemini_vision_model: str = "gemini-3.5-flash-lite"  # detection / colour questions
    gemini_edit_model: str = "gemini-nano-banana-2.1"
    gemini_edit_resolution: str = "2K"
    planner_provider: str = "auto"                 # auto | anthropic | gemini | mock
    qc_enabled: bool = True                        # vision QC verdict after generative edits
    qc_model: str | None = None                    # default: planner model

    # --- Optional: fal.ai (SAM-3, LaMa, FLUX Fill, Nano Banana) ---
    fal_key: str | None = None
    fal_sam_endpoint: str = "fal-ai/sam-3/image"
    fal_lama_endpoint: str = "fal-ai/lama"
    fal_fill_endpoint: str = "fal-ai/flux-pro/v1/fill"
    fal_edit_endpoint: str = "google/nano-banana-2.1/edit"
    fal_edit_resolution: str = "2K"

    # --- Accounts / tenant isolation ---
    # open = single-user local mode, no accounts; guest = everyone gets a private guest workspace automatically and can sign up
    # to keep it (deploy default); required = login wall (guest button still offered)
    auth_mode: str = "open"
    cookie_secure: bool = False      # set true behind https (Render sets X-Forwarded-Proto, which is also honoured)
    auth_show_reset_link: bool = False   # dev only: return the password-reset link in the API response when no SMTP is configured
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True

    # --- Storage ---
    data_dir: Path = ROOT / "data"
    public_base_url: str = ""  # e.g. https://demo.example.com ; empty = relative URLs

    # --- Behaviour ---
    thumb_long_side: int = 768      # thumbnails shown to the planner
    inspect_long_side: int = 1400   # zoomed inspections shown to the planner
    mask_dilate_pct: float = 0.012  # dilation radius as fraction of long side
    mask_feather_pct: float = 0.006
    small_mask_area: float = 0.03   # <= this fraction of image -> LaMa, else generative fill
    large_mask_area: float = 0.18   # > this fraction -> two fill strategies run and the QC verdict picks the better
    mask_review: bool = True        # vision check of surface masks: carve out parts that are not the target
    edit_concurrency: int = 3       # model edits running at once per process; 1 on a 512 MB host (each 2K edit peaks ~300 MB)
    generative_fallback: bool = True  # when a deterministic tool's result is flagged by QC, also try a Nano Banana version and keep the better

    @property
    def has_planner(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_fal(self) -> bool:
        return bool(self.fal_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def planner(self) -> str:
        """Which planner actually runs."""
        import os
        if os.environ.get("COCREATE_MOCK", "") not in ("", "0", "false"):
            return "mock"
        if self.planner_provider != "auto":
            return self.planner_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.gemini_api_key:
            return "gemini"
        return "mock"

    @property
    def planner_model_name(self) -> str:
        return {"anthropic": self.planner_model, "gemini": self.gemini_planner_model}.get(self.planner, "keyword planner")

    @property
    def image_backend(self) -> str:
        import os
        if os.environ.get("COCREATE_MOCK", "") not in ("", "0", "false"):
            return "mock"
        if self.fal_key:
            return "fal"
        if self.gemini_api_key:
            return "gemini"
        return "mock"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
