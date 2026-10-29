from __future__ import annotations

import json
from pathlib import Path

STATE_PATH = Path(__file__).with_name("state.json")


def load() -> dict | None:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "chat_id" not in data:
        return None
    return data


def save(chat_id: int, access_hash: int | None, title: str) -> None:
    payload = {"chat_id": chat_id, "access_hash": access_hash, "title": title}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear() -> None:
    try:
        STATE_PATH.unlink()
    except FileNotFoundError:
        pass
