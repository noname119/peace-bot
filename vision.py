from __future__ import annotations

import base64
import logging

from config import Config

log = logging.getLogger("vision")

MAX_IMAGE_BYTES = 8_000_000

_vision_unsupported = False


def _downscale(data: bytes, max_side: int = 1568) -> tuple[bytes, str] | None:
    try:
        from PIL import Image
        import io as _io
    except ImportError:
        return None
    try:
        with _io.BytesIO(data) as buf:
            with Image.open(buf) as img:
                img = img.convert("RGB")
                width, height = img.size
                biggest = max(width, height)
                if biggest > max_side:
                    scale = max_side / biggest
                    img = img.resize(
                        (round(width * scale), round(height * scale)),
                        Image.LANCZOS,
                    )
                out = _io.BytesIO()
                img.save(out, format="JPEG", quality=85)
                return out.getvalue(), "image/jpeg"
    except Exception:
        log.debug("Не удалось уменьшить изображение", exc_info=True)
        return None
    return None


async def describe_image(cfg: Config, data: bytes, mime: str) -> str | None:
    """Описывает картинку через vision-модель. None = недоступно."""
    global _vision_unsupported
    if _vision_unsupported:
        return None
    if not cfg.vision_enabled:
        return None
    if not data or len(data) > MAX_IMAGE_BYTES:
        return None

    prepared = _downscale(data)
    payload_data, payload_mime = prepared if prepared else (data, mime)
    content = base64.b64encode(payload_data).decode()

    payload = {
        "model": cfg.vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Опиши, что на изображении, коротко (1–2 предложения, "
                            "по-русски): главные объекты, действие, настроение. "
                            "Если это мем — передай суть шутки. "
                            "Только описание, без лишних слов."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{payload_mime};base64,{content}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.4,
    }
    headers = {"Authorization": f"Bearer {cfg.vision_api_key}"}

    try:
        import httpx
    except ImportError:
        return None

    try:
        async with httpx.AsyncClient(timeout=cfg.llm_timeout) as http:
            response = await http.post(
                f"{cfg.vision_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
        text = result["choices"][0]["message"]["content"]
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status == 400:
            _vision_unsupported = True
            log.warning("Vision недоступен на этом endpoint, описания отключены")
        else:
            log.warning("Vision-запрос не удался: %s", error)
        return None

    text = " ".join(str(text).split()).strip()
    return text or None
