from __future__ import annotations

import asyncio
import logging
import sys
import time
import warnings
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from telethon import TelegramClient, events
from telethon.tl.types import InputPeerChannel, InputPeerChat, InputPeerUser

import brain
import media
import memory
import state
import vision
from config import load_config

log = logging.getLogger("userbot")

ROBOT_PREFIX = "🤖 "
PLACEHOLDER_TEXT = "думаю…"
TEXT_LIMIT = 400
RECENT_LIMIT = 100
BOT_ID_LIMIT = 10
INITIAL_WAKE = 1
LLM_BACKOFF = 60
BOT_MIN_GAP = 30
RATE_WINDOW = 600
RATE_MAX = 6
SEND_TIMEOUT = 30
SEED_TIMEOUT = 120
DELETE_TIMEOUT = 20
DOWNLOAD_TIMEOUT = 60
FETCH_TIMEOUT = 15
FALLBACK_ANSWER = "Слушаю тебя. Что случилось?"
RESET_FLAG = Path(__file__).with_name("reset.flag")
LOG_PATH = Path(__file__).with_name("bot.log")

BOT_WORD_STRIP = ".,!?;:()[]{}\"'«»…—–-"


async def tg_call(coro, timeout: float, what: str):
    """Вызов Telegram API с таймаутом: вместо вечного зависания — исключение."""
    try:
        return await asyncio.wait_for(asyncio.ensure_future(coro), timeout)
    except asyncio.TimeoutError:
        log.warning("%s: таймаут %s c", what, timeout)
        raise


def truncate(text: str, limit: int = TEXT_LIMIT) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def display_name(chat) -> str:
    title = getattr(chat, "title", None)
    if title:
        return title
    first = getattr(chat, "first_name", "") or ""
    last = getattr(chat, "last_name", "") or ""
    name = (first + " " + last).strip()
    if name:
        return name
    chat_id = getattr(chat, "id", None)
    return f"id{chat_id}" if chat_id else "без названия"


def sender_label(entity) -> str:
    if entity is None:
        return "Собеседник"
    name = display_name(entity)
    username = getattr(entity, "username", None)
    if username:
        return f"{name} (@{username})"
    return name


def build_peer(active: dict):
    chat_id = active["chat_id"]
    access_hash = active.get("access_hash")
    if chat_id > 0 and access_hash is not None:
        return InputPeerUser(user_id=chat_id, access_hash=access_hash)
    if chat_id <= -1000000000000 and access_hash is not None:
        channel_id = -chat_id - 1000000000000
        return InputPeerChannel(channel_id=channel_id, access_hash=access_hash)
    if chat_id < 0:
        return InputPeerChat(chat_id=-chat_id)
    return None


def parse_command(text: str) -> str | None:
    command = (text or "").strip().lower()
    if command == "!activate":
        return "activate"
    if command == "!deactivate":
        return "deactivate"
    if command == "!reset":
        return "reset"
    return None


def parse_service_command(text: str) -> str | None:
    """Возвращает текст после '!служебно' (или '!служебное') или None."""
    stripped = (text or "").strip()
    low = stripped.lower()
    for prefix in ("!служебно", "!служебное"):
        if low == prefix:
            return ""
        if low.startswith(prefix) and len(low) > len(prefix) and not low[len(prefix)].isalnum():
            return stripped[len(prefix):].lstrip(" \t\n:,-—–").strip()
    return None


_SHOW_VERBS = (
    "покажи", "показать", "показывай", "выдай", "выдать", "выдавай",
    "выведи", "вывести", "выводи", "прочитай", "прочитать",
    "расскажи", "рассказать", "напомни", "напомнить",
    "выгрузи", "выгрузить", "скинь", "скинуть",
    "продублируй", "продублировать",
)

_MEMORY_NOUNS = {
    "память": "memory", "памяти": "memory", "памятью": "memory",
    "нормы": "norms", "норм": "norms", "нормах": "norms", "норму": "norms",
    "правила": "norms", "правилах": "norms", "правило": "norms",
    "факты": "facts", "фактах": "facts", "факт": "facts", "фактов": "facts",
    "контекст": "recent", "контексте": "recent",
    "переписку": "recent", "переписки": "recent", "переписке": "recent",
    "историю": "recent", "истории": "recent",
    "сообщения": "recent", "сообщений": "recent",
    "помнишь": "memory", "запомнил": "memory",
}


def detect_show_order(text: str) -> str | None:
    """Определяет приказ показать память. Возвращает target или None."""
    low = " " + (text or "").lower() + " "
    if "что ты помнишь" in low or "что помнишь" in low:
        return "memory"
    if not any(verb in low for verb in _SHOW_VERBS):
        return None
    for noun, target in _MEMORY_NOUNS.items():
        if noun in low:
            return target
    return None


def contains_bot_word(text: str) -> bool:
    for token in (text or "").lower().split():
        cleaned = token.strip(BOT_WORD_STRIP)
        if cleaned == "бот" or cleaned.startswith("бот-"):
            return True
    return False


def replace_recent_placeholder(recent: list, old_text: str, new_text: str) -> bool:
    for i in range(len(recent) - 1, -1, -1):
        entry = recent[i]
        role = entry[0]
        text = entry[1]
        ts = entry[2] if len(entry) > 2 else None
        if role == "Бот" and text == old_text:
            recent[i] = ("Бот", new_text, ts)
            return True
    return False


def remove_recent_placeholder(recent: list, old_text: str) -> bool:
    for i in range(len(recent) - 1, -1, -1):
        entry = recent[i]
        role = entry[0]
        text = entry[1] if len(entry) > 1 else ""
        if role == "Бот" and text == old_text:
            del recent[i]
            return True
    return False


async def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )
    cfg = load_config()
    if not cfg.api_id or not cfg.api_hash:
        raise SystemExit(
            "Заполните TG_API_ID и TG_API_HASH в .env. "
            "Ключи выдаются на my.telegram.org, инструкция в README.md."
        )
    if not cfg.llm_api_key:
        raise SystemExit("Заполните OPENAI_API_KEY в .env: без LLM бот не работает.")

    client = TelegramClient(cfg.session_name, cfg.api_id, cfg.api_hash)
    await client.start()
    me = await client.get_me()
    own_id = me.id

    active = state.load()
    facts = ""
    norms = ""
    recent: list[tuple[str, str]] = []
    next_wake = INITIAL_WAKE
    pending_count = 0
    bot_message_ids: deque[int] = deque(maxlen=BOT_ID_LIMIT)
    last_bot_at: dict[int, float] = {}
    last_conflict_at: dict[int, float] = {}
    sent_times: dict[int, deque[float]] = {}
    waking = False
    wake_queued = False
    wake_queued_addressed = False
    llm_backoff_until = 0.0
    silence_until = 0.0

    log.info("Аккаунт: %s (id=%s)", me.username or me.first_name or own_id, own_id)
    log.info("LLM: %s (%s)", cfg.llm_model, cfg.llm_base_url)

    def is_active_chat(event) -> bool:
        return active is not None and event.chat_id == active["chat_id"]

    def can_send(chat_id: int, reason: str, now: float, mandatory: bool = False) -> tuple[bool, str]:
        if not mandatory and now - last_bot_at.get(chat_id, 0) < BOT_MIN_GAP:
            return False, "слишком часто"
        if reason == "conflict" and now - last_conflict_at.get(chat_id, 0) < cfg.conflict_min_gap:
            return False, "недавно уже вмешивались в конфликт"
        if reason == "chime" and now - last_bot_at.get(chat_id, 0) < cfg.chime_min_gap:
            return False, "недавно писали"
        times = sent_times.setdefault(chat_id, deque())
        while times and now - times[0] > RATE_WINDOW:
            times.popleft()
        if not mandatory and len(times) >= RATE_MAX:
            return False, "лимит 6 сообщений за 10 минут"
        return True, ""

    def load_chat(chat_id: int) -> None:
        nonlocal facts, norms, recent, next_wake, pending_count
        saved = memory.get_chat(chat_id)
        facts = saved["facts"]
        norms = saved["norms"]
        recent = [tuple(item) if len(item) == 3 else (item[0], item[1], None) for item in saved["recent"]]
        next_wake = saved["next_wake_in"]
        pending_count = 0
        bot_message_ids.clear()

    def persist() -> None:
        if active is None:
            return
        memory.save_chat(
            active["chat_id"],
            facts,
            [list(item) for item in recent],
            next_wake,
            norms,
        )

    def append_recent(role: str, text: str, ts: float | None = None) -> None:
        if ts is None:
            ts = time.time()
        recent.append((role, truncate(text), int(ts)))
        if len(recent) > RECENT_LIMIT:
            del recent[:-RECENT_LIMIT]

    async def describe_photo(message) -> str | None:
        """Скачивает и описывает фото/статичный стикер. None — не вышло."""
        mime = media.vision_mime(message)
        if not mime:
            return None
        try:
            data = await tg_call(
                client.download_media(message, file=bytes),
                DOWNLOAD_TIMEOUT,
                "Скачивание медиа",
            )
        except Exception:
            data = None
            log.debug("Не удалось скачать медиа", exc_info=True)
        if not data:
            log.info("Фото: не удалось скачать")
            return None
        desc = await vision.describe_image(cfg, data, mime)
        if desc:
            log.info("Фото описано: %s", desc[:150])
            return desc
        log.info("Фото: vision не вернул описание")
        return None

    async def seed_context() -> None:
        nonlocal recent
        if active is None:
            return
        peer = build_peer(active) or active["chat_id"]
        try:
            messages = await tg_call(
                client.get_messages(peer, limit=RECENT_LIMIT),
                SEED_TIMEOUT,
                "Загрузка истории чата",
            )
        except Exception:
            log.warning("Не удалось загрузить историю чата", exc_info=True)
            return

        items: list[tuple[str, str, int | None]] = []
        labels: dict[int, str] = {}
        vision_targets: list[tuple[int, object, str]] = []
        for message in reversed(messages):
            raw = (message.raw_text or "").strip()
            if raw.startswith(ROBOT_PREFIX):
                role = "Бот"
                if message.id not in bot_message_ids:
                    bot_message_ids.append(message.id)
                if message.date:
                    last_bot_at[active["chat_id"]] = message.date.timestamp()
            elif message.out:
                role = "Я"
            else:
                sender_id = message.sender_id or 0
                if sender_id not in labels:
                    try:
                        sender = await message.get_sender()
                    except Exception:
                        sender = None
                    labels[sender_id] = sender_label(sender)
                role = labels[sender_id]
            try:
                ts = int(message.date.timestamp()) if message.date else None
            except (TypeError, ValueError, OSError, OverflowError):
                ts = None
            parts = media.split_message(message, role)
            if not parts:
                continue
            for entry_role, entry_text in parts:
                items.append((entry_role, truncate(entry_text), ts))
            mime = media.vision_mime(message)
            if mime:
                vision_targets.append((len(items) - 1, message, mime))

        for index, message, mime in reversed(vision_targets[-2:]):
            desc = await describe_photo(message)
            if desc:
                try:
                    vts = int(message.date.timestamp()) if message.date else None
                except (TypeError, ValueError, OSError, OverflowError):
                    vts = None
                items.insert(index + 1, ("Система", truncate(f"{media.SYSTEM_DESC_PREFIX}: {desc}"), vts))

        if items:
            recent = items[-RECENT_LIMIT:]
            persist()
            log.info("Контекст: загружено %s сообщений", len(recent))

    async def notify_owner(text: str) -> None:
        try:
            await tg_call(
                client.send_message("me", ROBOT_PREFIX + text),
                SEND_TIMEOUT,
                "Уведомление владельцу",
            )
        except Exception:
            log.exception("Не удалось отправить подтверждение в Избранное")

    async def delete_command(event) -> None:
        try:
            await tg_call(event.delete(), DELETE_TIMEOUT, "Удаление команды")
        except Exception:
            log.debug("Не удалось удалить команду", exc_info=True)

    async def activate(event) -> None:
        nonlocal active, silence_until, wake_queued, wake_queued_addressed
        if active is not None and active["chat_id"] == event.chat_id:
            await notify_owner("Бот уже активен в этом чате.")
            return

        chat = await event.get_chat()
        title = display_name(chat)
        active = {
            "chat_id": event.chat_id,
            "access_hash": getattr(chat, "access_hash", None),
            "title": title,
        }
        state.save(active["chat_id"], active["access_hash"], title)
        load_chat(active["chat_id"])
        silence_until = 0.0
        wake_queued = False
        wake_queued_addressed = False
        await seed_context()
        log.info("Бот активирован в чате %s (%s)", title, event.chat_id)
        await notify_owner(
            f"Бот активирован в чате «{title}». Он будет просыпаться по контексту."
        )

    async def deactivate(event) -> None:
        nonlocal active, silence_until, wake_queued, wake_queued_addressed
        if active is None:
            await notify_owner("Бот уже остановлен.")
            return
        title = active.get("title") or str(active["chat_id"])
        active = None
        state.clear()
        silence_until = 0.0
        wake_queued = False
        wake_queued_addressed = False
        log.info("Бот остановлен")
        await notify_owner(f"Бот остановлен. В чате «{title}» он больше не проснётся.")

    async def reset_memory(event) -> None:
        nonlocal facts, norms, recent, next_wake, pending_count
        nonlocal wake_queued, wake_queued_addressed
        if active is None:
            await notify_owner("Нет активного чата: память очищать нечего.")
            return
        title = active.get("title") or str(active["chat_id"])
        facts = ""
        norms = ""
        recent = []
        next_wake = INITIAL_WAKE
        pending_count = 0
        wake_queued = False
        wake_queued_addressed = False
        persist()
        log.info("Память и контекст очищены в чате %s", title)
        await notify_owner(
            f"Память и контекст в чате «{title}» очищены. Начинаю с чистого листа."
        )

    async def execute_service_order(event, order_text: str) -> None:
        nonlocal facts, norms, next_wake, silence_until
        if active is None:
            await notify_owner(
                "Нет активного чата: некуда выполнять приказ. Сначала !activate."
            )
            return
        chat_id = active["chat_id"]
        title = active.get("title") or str(chat_id)
        if not order_text:
            await notify_owner("Пустой приказ. Пример: !служебно скажи, что я скоро буду.")
            return

        show_target = detect_show_order(order_text)
        if show_target is not None:
            order = brain.Order(action="show", target=show_target)
            log.info("Приказ %r: прямое распознавание, target=%s", order_text[:150], show_target)
        else:
            try:
                order = await brain.execute_order(cfg, facts, norms, list(recent), order_text)
            except Exception:
                log.exception("Не удалось разобрать приказ")
                await notify_owner("Не удалось разобрать приказ: ошибка LLM.")
                return
            if order is None:
                await notify_owner("Не удалось разобрать приказ: LLM недоступен.")
                return
            log.info(
                "Приказ %r: action=%s target=%s minutes=%s wake_in=%s",
                order_text[:150],
                order.action,
                order.target,
                order.minutes,
                order.wake_in,
            )

        if order.action == "say":
            if not order.text:
                await notify_owner("Не понял, что именно сказать.")
                return
            allowed, why = can_send(chat_id, "", time.time(), mandatory=True)
            if not allowed:
                await notify_owner(f"Не могу отправить сейчас: {why}.")
                return
            try:
                await send_bot_message(chat_id, order.text, "")
            except Exception:
                log.exception("Не удалось выполнить приказ say")
                await notify_owner("Не удалось отправить сообщение.")
                return
            await notify_owner(f"Сказал в чате «{title}».")
        elif order.action == "remember":
            if not order.text:
                await notify_owner("Не понял, что именно запомнить.")
                return
            entry = order.text.strip()
            if order.target == "norms":
                norms = (norms + "\n" + entry).strip() if norms else entry
                where = "нормы"
            else:
                facts = (facts + "\n" + entry).strip() if facts else entry
                where = "память"
            persist()
            await notify_owner(f"Записал в {where}: {order.text}")
        elif order.action == "set_wake":
            if not 1 <= order.wake_in <= 15:
                await notify_owner("Не понял интервал: укажите число от 1 до 15.")
                return
            next_wake = order.wake_in
            persist()
            await notify_owner(f"Интервал пробуждений: {next_wake} сообщений.")
        elif order.action == "silence":
            if order.minutes <= 0:
                silence_until = 0.0
                await notify_owner("Режим молчания выключен.")
            else:
                silence_until = time.monotonic() + order.minutes * 60
                end = datetime.fromtimestamp(time.time() + order.minutes * 60).strftime("%H:%M")
                await notify_owner(f"Молчу {order.minutes} мин (до {end}).")
        elif order.action == "status":
            last_ts = last_bot_at.get(chat_id)
            if last_ts:
                last_text = f"{max(0, int((time.time() - last_ts) / 60))} мин. назад"
            else:
                last_text = "ещё не писал"
            if silence_until > time.monotonic():
                left = int((silence_until - time.monotonic()) / 60)
                silence_text = f"до конца осталось ~{left} мин"
            else:
                silence_text = "выкл"
            fact_lines = [line for line in facts.splitlines() if line.strip()]
            lines = [
                f"Активный чат: «{title}» ({chat_id})",
                f"Пробуждение через: {next_wake} сообщений (накоплено: {pending_count})",
                f"Последнее сообщение бота: {last_text}",
                f"Молчание: {silence_text}",
                f"Память: фактов: {len(fact_lines)}, нормы: {'есть' if norms.strip() else 'нет'}, "
                f"сообщений в контексте: {len(recent)}",
                f"LLM: {cfg.llm_model} (thinking {'вкл' if cfg.llm_thinking else 'выкл'})",
            ]
            await notify_owner("\n".join(lines))
        elif order.action == "show":
            if order.target == "facts":
                body = facts.strip() or "(память пуста)"
                header = "Факты из памяти:"
            elif order.target == "norms":
                body = norms.strip() or "(нормы не заданы)"
                header = "Нормы чата:"
            elif order.target == "recent":
                convo = [f"{entry[0]}: {entry[1]}" for entry in recent[-15:]]
                body = "\n".join(convo) or "(контекст пуст)"
                header = "Последние сообщения:"
            else:
                parts = [
                    "Факты:\n" + (facts.strip() or "(пусто)"),
                    "Нормы:\n" + (norms.strip() or "(не заданы)"),
                ]
                body = "\n\n".join(parts)
                header = "Память бота:"
            await notify_owner(f"{header}\n{body}")
        else:
            await notify_owner("Не понял приказ.")

    async def send_bot_message(chat_id: int, text: str, reason: str = "") -> None:
        snapshot = active
        if snapshot is None:
            return
        peer = build_peer(snapshot) or chat_id
        sent = await tg_call(
            client.send_message(peer, ROBOT_PREFIX + text),
            SEND_TIMEOUT,
            "Отправка сообщения",
        )
        if sent and sent.id not in bot_message_ids:
            bot_message_ids.append(sent.id)
        now = time.time()
        last_bot_at[chat_id] = now
        sent_times.setdefault(chat_id, deque()).append(now)
        if reason == "conflict":
            last_conflict_at[chat_id] = now
        log.info("Сообщение отправлено в чат %s (reason=%s)", chat_id, reason or "-")

    async def run_wake(force: bool = False, addressed: bool = False, proactive: bool = False) -> None:
        nonlocal pending_count, next_wake, facts, norms, waking, llm_backoff_until
        nonlocal wake_queued, wake_queued_addressed
        if active is None:
            return
        if waking:
            if force and not proactive:
                wake_queued = True
                wake_queued_addressed = wake_queued_addressed or addressed
                log.info("Пробуждение поставлено в очередь (бот занят)")
            elif proactive:
                log.info("Плановое пробуждение пропущено: бот занят")
            return
        if pending_count <= 0 and not force and not proactive:
            return
        if not force and not proactive and pending_count < next_wake:
            return
        if time.monotonic() < llm_backoff_until:
            return
        if proactive:
            log.info("Плановое пробуждение по таймеру")

        waking = True
        first = True
        placeholder = None

        async def drop_placeholder() -> None:
            nonlocal placeholder
            if placeholder is None:
                return
            msg, placeholder = placeholder, None
            try:
                await tg_call(
                    client.delete_messages(chat_id, msg.id),
                    DELETE_TIMEOUT,
                    "Удаление заглушки",
                )
            except Exception:
                log.debug("Не удалось удалить заглушку", exc_info=True)
            if remove_recent_placeholder(recent, PLACEHOLDER_TEXT):
                persist()

        try:
            while True:
                if wake_queued:
                    wake_queued = False
                    force = True
                    addressed = wake_queued_addressed
                    wake_queued_addressed = False
                    first = True
                    log.info("Пробуждение из очереди")
                snapshot = active
                if snapshot is None:
                    return
                run_now = force or proactive
                if pending_count <= 0 and not run_now:
                    return
                if not run_now and pending_count < next_wake:
                    return
                force = False
                cur_proactive = proactive
                proactive = False

                hint = addressed if first else False
                must_answer = hint and cfg.mandatory_reply
                first = False
                analyzed = pending_count

                chat_id = snapshot["chat_id"]
                if hint and placeholder is None:
                    if time.monotonic() < silence_until:
                        log.info("Заглушка пропущена: режим молчания")
                    else:
                        log.info("Отправляю заглушку (думаю…)")
                        try:
                            placeholder = await tg_call(
                                client.send_message(
                                    build_peer(snapshot) or chat_id,
                                    ROBOT_PREFIX + PLACEHOLDER_TEXT,
                                ),
                                SEND_TIMEOUT,
                                "Отправка заглушки",
                            )
                        except Exception:
                            log.warning("Не удалось отправить заглушку", exc_info=True)
                            placeholder = None
                        if placeholder is not None and placeholder.id not in bot_message_ids:
                            bot_message_ids.append(placeholder.id)
                last_ts = last_bot_at.get(chat_id)
                last_minutes = (
                    max(0, int((time.time() - last_ts) / 60)) if last_ts else None
                )

                decision = await brain.decide(
                    cfg,
                    facts,
                    norms,
                    list(recent),
                    analyzed,
                    last_minutes,
                    hint,
                    cfg.reply_enabled,
                    cfg.chime_enabled,
                    must_answer,
                    brain.format_now(),
                    cur_proactive,
                )
                if decision is None:
                    llm_backoff_until = time.monotonic() + LLM_BACKOFF
                    log.warning("Пробуждение отложено на %s секунд", LLM_BACKOFF)
                    await drop_placeholder()
                    return

                if must_answer and (not decision.speak or not decision.message):
                    forced = await brain.answer_forced(cfg, facts, norms, list(recent))
                    if forced is None:
                        llm_backoff_until = time.monotonic() + LLM_BACKOFF
                        log.warning("Пробуждение отложено на %s секунд", LLM_BACKOFF)
                        await drop_placeholder()
                        return
                    decision = brain.Decision(
                        True,
                        forced or FALLBACK_ANSWER,
                        decision.next_wake,
                        decision.memory,
                        decision.norms,
                        "answer",
                    )
                    log.info("Обязательный ответ на обращение")

                pending_count = max(0, pending_count - analyzed)
                next_wake = decision.next_wake
                if decision.memory:
                    facts = decision.memory
                if decision.norms:
                    norms = decision.norms
                persist()

                log.info(
                    "Пробуждение: speak=%s reason=%s next_wake=%s pending=%s",
                    decision.speak,
                    decision.reason or "-",
                    next_wake,
                    pending_count,
                )

                should_send = decision.speak
                if (
                    should_send
                    and decision.reason == "answer"
                    and not cfg.reply_enabled
                    and not must_answer
                ):
                    should_send = False
                    log.info("Сообщение пропущено: ответы отключены (REPLY_ENABLED=0)")
                if should_send and decision.reason == "chime" and not cfg.chime_enabled:
                    should_send = False
                    log.info("Сообщение пропущено: вступления отключены (CHIME_ENABLED=0)")
                if should_send and time.monotonic() < silence_until:
                    should_send = False
                    log.info("Сообщение пропущено: режим молчания (служебный приказ)")
                if should_send:
                    allowed, why = can_send(
                        chat_id,
                        decision.reason,
                        time.time(),
                        mandatory=(
                            decision.reason == "answer" and hint
                        ),
                    )
                    if not allowed:
                        should_send = False
                        log.info("Сообщение пропущено: %s", why)

                if should_send:
                    if active is not snapshot:
                        await drop_placeholder()
                        return
                    if placeholder is not None:
                        try:
                            await tg_call(
                                client.edit_message(
                                    chat_id,
                                    placeholder.id,
                                    ROBOT_PREFIX + decision.message,
                                ),
                                SEND_TIMEOUT,
                                "Правка заглушки",
                            )
                        except Exception:
                            log.warning(
                                "Не удалось изменить заглушку, отправляю новым сообщением",
                                exc_info=True,
                            )
                            await drop_placeholder()
                            try:
                                await send_bot_message(
                                    snapshot["chat_id"],
                                    decision.message,
                                    decision.reason,
                                )
                            except Exception:
                                log.exception("Не удалось отправить сообщение")
                        else:
                            if replace_recent_placeholder(
                                recent, PLACEHOLDER_TEXT, truncate(decision.message)
                            ):
                                persist()
                            now = time.time()
                            last_bot_at[chat_id] = now
                            sent_times.setdefault(chat_id, deque()).append(now)
                            if decision.reason == "conflict":
                                last_conflict_at[chat_id] = now
                            log.info("Сообщение отправлено (правка заглушки)")
                        placeholder = None
                    else:
                        try:
                            await send_bot_message(
                                snapshot["chat_id"],
                                decision.message,
                                decision.reason,
                            )
                        except Exception:
                            log.exception("Не удалось отправить сообщение")
                elif placeholder is not None:
                    await drop_placeholder()

                if pending_count < next_wake:
                    return
                if time.monotonic() < llm_backoff_until:
                    return
        finally:
            waking = False

    async def proactive_loop() -> None:
        minutes = max(0, cfg.proactive_wake_minutes)
        if minutes <= 0:
            return
        interval = max(300, minutes * 60)
        log.info("Плановые пробуждения: каждые %s мин", minutes)
        while True:
            await asyncio.sleep(interval)
            if active is None:
                continue
            if time.monotonic() < llm_backoff_until:
                continue
            if time.monotonic() < silence_until:
                continue
            try:
                await run_wake(force=True, addressed=False, proactive=True)
            except Exception:
                log.exception("Ошибка планового пробуждения")

    @client.on(events.NewMessage(outgoing=True))
    async def on_outgoing(event):
        nonlocal pending_count
        text = (event.raw_text or "").strip()

        if text.startswith(ROBOT_PREFIX):
            if is_active_chat(event):
                append_recent("Бот", text.removeprefix(ROBOT_PREFIX).strip())
                if event.id not in bot_message_ids:
                    bot_message_ids.append(event.id)
                last_bot_at[event.chat_id] = time.time()
                persist()
            return

        command = parse_command(text)
        if command == "activate":
            if event.chat_id != own_id:
                await delete_command(event)
                await activate(event)
            return
        if command == "deactivate":
            if event.chat_id != own_id:
                await delete_command(event)
            await deactivate(event)
            return
        if command == "reset":
            if event.chat_id != own_id:
                await delete_command(event)
            await reset_memory(event)
            return

        service = parse_service_command(text)
        if service is not None:
            if event.chat_id != own_id:
                await delete_command(event)
            await execute_service_order(event, service)
            return

        if not is_active_chat(event):
            return

        desc = await describe_photo(event.message)
        parts = media.split_message(event.message, "Я", desc)
        if not parts:
            return
        for entry_role, entry_text in parts:
            append_recent(entry_role, entry_text)
        persist()
        pending_count += 1

        addressed_hint = bool(
            (event.reply_to_msg_id and event.reply_to_msg_id in bot_message_ids)
            or contains_bot_word(text)
        )
        await run_wake(force=addressed_hint, addressed=addressed_hint)

    @client.on(events.NewMessage(incoming=True))
    async def on_incoming(event):
        nonlocal pending_count
        if not is_active_chat(event):
            return
        text = (event.raw_text or "").strip()
        desc = await describe_photo(event.message)

        try:
            sender = await tg_call(event.get_sender(), FETCH_TIMEOUT, "Получение отправителя")
        except Exception:
            sender = None
        parts = media.split_message(event.message, sender_label(sender), desc)
        if not parts:
            return
        for entry_role, entry_text in parts:
            append_recent(entry_role, entry_text)
        persist()
        pending_count += 1

        addressed_hint = bool(
            (event.reply_to_msg_id and event.reply_to_msg_id in bot_message_ids)
            or contains_bot_word(text)
        )
        await run_wake(force=addressed_hint, addressed=addressed_hint)

    skip_seed = RESET_FLAG.exists()
    if skip_seed:
        try:
            RESET_FLAG.unlink()
        except OSError:
            pass

    if active:
        load_chat(active["chat_id"])
        log.info("Активный чат: %s (%s)", active.get("title"), active["chat_id"])
        if skip_seed:
            log.info("Память очищена: контекст начнётся с новых сообщений")
        else:
            await seed_context()
    else:
        log.info("Бот не активирован. Отправьте !activate в нужном чате")

    log.info("Слушаю сообщения")
    asyncio.create_task(proactive_loop())
    await client.run_until_disconnected()


if __name__ == "__main__":
    if sys.platform == "win32":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    failures = 0
    while True:
        started = time.monotonic()
        try:
            asyncio.run(main())
        except (KeyboardInterrupt, SystemExit):
            break
        except BaseException:
            log.exception("Критическая ошибка, перезапуск")
        else:
            log.warning("Основной цикл завершён, перезапуск")
        uptime = time.monotonic() - started
        failures = failures + 1 if uptime < 60 else 0
        delay = min(300, 10 * (2 ** min(failures, 4)))
        log.info("Перезапуск через %s c", delay)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            break
