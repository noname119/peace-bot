from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    api_id: int
    api_hash: str
    session_name: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: float
    llm_thinking: bool
    llm_reasoning_effort: str
    reply_enabled: bool
    chime_enabled: bool
    chime_min_gap: int
    mandatory_reply: bool


def load_config() -> Config:
    effort = os.getenv("LLM_REASONING_EFFORT", "high").strip().lower()
    if effort not in {"low", "high", "max"}:
        effort = "high"
    return Config(
        api_id=_env_int("TG_API_ID", 0),
        api_hash=os.getenv("TG_API_HASH", "").strip(),
        session_name=os.getenv("SESSION_NAME", "userbot").strip() or "userbot",
        llm_base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        llm_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "deepseek-flash").strip() or "deepseek-flash",
        llm_timeout=max(5.0, _env_float("LLM_TIMEOUT", 60.0)),
        llm_thinking=_env_bool("LLM_THINKING", True),
        llm_reasoning_effort=effort,
        reply_enabled=_env_bool("REPLY_ENABLED", True),
        chime_enabled=_env_bool("CHIME_ENABLED", True),
        chime_min_gap=max(60, _env_int("CHIME_MIN_GAP", 900)),
        mandatory_reply=_env_bool("MANDATORY_REPLY", True),
    )
