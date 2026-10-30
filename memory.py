from __future__ import annotations

import json
from pathlib import Path

MEMORY_PATH = Path(__file__).with_name("memory.json")
RECENT_LIMIT = 100
WAKE_MIN = 1
WAKE_MAX = 15
DEFAULT_WAKE = 1


def _load_all() -> dict:
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_all(data: dict) -> None:
    temporary = MEMORY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MEMORY_PATH)


def get_chat(chat_id: int) -> dict:
    data = _load_all().get(str(chat_id)) or {}

    facts = data.get("facts")
    if not isinstance(facts, str):
        facts = ""

    recent = data.get("recent")
    if not isinstance(recent, list):
        recent = []
    clean_recent: list[list] = []
    for item in recent[-RECENT_LIMIT:]:
        if (
            isinstance(item, list)
            and len(item) in (2, 3)
            and isinstance(item[0], str)
            and isinstance(item[1], str)
        ):
            ts = None
            if len(item) >= 3 and item[2] is not None:
                try:
                    ts = int(item[2])
                except (TypeError, ValueError):
                    ts = None
            clean_recent.append([item[0], item[1], ts])

    norms = data.get("norms")
    if not isinstance(norms, str):
        norms = ""

    try:
        next_wake = int(data.get("next_wake_in", DEFAULT_WAKE))
    except (TypeError, ValueError):
        next_wake = DEFAULT_WAKE
    next_wake = max(WAKE_MIN, min(WAKE_MAX, next_wake))

    return {
        "facts": facts,
        "norms": norms,
        "recent": clean_recent,
        "next_wake_in": next_wake,
    }


def save_chat(
    chat_id: int,
    facts: str,
    recent: list[list[str]],
    next_wake: int,
    norms: str = "",
) -> None:
    all_data = _load_all()
    all_data[str(chat_id)] = {
        "facts": facts,
        "norms": norms,
        "recent": recent[-RECENT_LIMIT:],
        "next_wake_in": max(WAKE_MIN, min(WAKE_MAX, next_wake)),
    }
    _save_all(all_data)
