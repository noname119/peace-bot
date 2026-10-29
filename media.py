from __future__ import annotations

from telethon.tl import types as tl


def _attributes(doc) -> list:
    return list(getattr(doc, "attributes", None) or [])


def media_kind(message) -> tuple[str | None, dict]:
    """Определяет тип медиа только по локальным данным.

    Возвращает (kind, info), где kind — photo, sticker, gif, video или None.
    """
    media = getattr(message, "media", None)
    if media is None:
        return None, {}
    if isinstance(media, tl.MessageMediaPhoto):
        return "photo", {}
    if isinstance(media, tl.MessageMediaDocument):
        doc = getattr(media, "document", None)
        if doc is None or isinstance(doc, tl.DocumentEmpty):
            return None, {}
        mime = (getattr(doc, "mime_type", "") or "").lower()
        attrs = _attributes(doc)
        sticker = next(
            (a for a in attrs if isinstance(a, tl.DocumentAttributeSticker)), None
        )
        if sticker is not None:
            return "sticker", {
                "alt": getattr(sticker, "alt", "") or "",
                "animated": "tgsticker" in mime,
                "video": mime == "video/webm",
                "mime": mime,
            }
        if any(isinstance(a, tl.DocumentAttributeAnimated) for a in attrs):
            return "gif", {"mime": mime}
        for attr in attrs:
            if isinstance(attr, tl.DocumentAttributeVideo):
                try:
                    duration = int(getattr(attr, "duration", 0) or 0)
                except (TypeError, ValueError):
                    duration = 0
                return "video", {"duration": duration, "mime": mime}
        if mime.startswith("image/"):
            return "photo", {"mime": mime}
        return None, {}
    return None, {}


def stub_for_message(message) -> str | None:
    """Короткая текстовая заглушка для медиа или None."""
    kind, info = media_kind(message)
    if kind is None:
        return None
    if kind == "photo":
        return "[фото]"
    if kind == "sticker":
        parts = ["стикер"]
        alt = info.get("alt", "")
        if alt:
            parts.append(alt)
        if info.get("video"):
            parts.append("(видео)")
        elif info.get("animated"):
            parts.append("(анимированный)")
        return "[" + " ".join(parts) + "]"
    if kind == "gif":
        return "[гифка]"
    if kind == "video":
        duration = info.get("duration", 0)
        if duration:
            minutes, seconds = divmod(int(duration), 60)
            length = f"{minutes}:{seconds:02d}" if minutes else f"0:{seconds:02d}"
            return f"[видео {length}]"
        return "[видео]"
    return None


def render_message(message, description: str | None = None) -> str:
    """Текст сообщения для контекста: заглушка + описание + подпись."""
    raw = (getattr(message, "raw_text", None) or "").strip()
    stub = stub_for_message(message)
    if not stub and not raw and not description:
        return ""
    parts = []
    if stub:
        parts.append(stub)
    if description:
        parts.append(description.strip())
    if raw:
        parts.append(raw)
    return "\n".join(parts)


def vision_mime(message) -> str | None:
    """MIME для отправки в vision-модель или None, если пиксели недоступны."""
    kind, info = media_kind(message)
    if kind == "photo":
        return info.get("mime") or "image/jpeg"
    if kind == "sticker" and not info.get("animated") and not info.get("video"):
        return info.get("mime") or "image/webp"
    return None
