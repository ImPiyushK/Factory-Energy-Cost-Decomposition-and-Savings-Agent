"""Central configuration. All knobs come from environment / .env (keys never in Git)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    llm_provider: str = "mock"          # mock | openai | anthropic | google | ollama
    llm_model: str = "mock-template"
    mock_llm: bool = True
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_max_retries: int = 1            # brief regeneration retries on faithfulness failure

    # --- Paths ---
    db_path: str = "data/energy_agent.db"
    data_dir: str = "data"

    # --- Analytical settings (documented defaults; overridable per env/judges) ---
    recon_tolerance: float = 0.05       # |sum(sub) - main| / main tolerance (spec 4.5)
    cdd_base: float = 18.0
    night_start: int = 22
    night_end: int = 6
    emission_factor_kg_per_kwh: float | None = None  # None => must come from data pack or be flagged
    decomp_method: str = "shapley"      # shapley (Config A) | sequential (Config B) - eval comparisons
    n_bootstrap: int = 60               # weekly block bootstrap draws

    # --- API ---
    app_version: str = "1.0.0"

    def resolve(self, raw: str | None) -> str:
        """Resolve a possibly-relative path against the project root."""
        if not raw:
            return raw
        p = Path(raw)
        return str(p if p.is_absolute() else ROOT / p)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    os.environ.setdefault("TZ", "UTC")
    return s


settings = get_settings()
