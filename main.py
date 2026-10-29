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
import memory
import state
from config import load_config

log = logging.getLogger("userbot")

ROBOT_PREFIX = "🤖 "
TEXT_LIMIT = 400
RECENT_LIMIT = 100
BOT_ID_LIMIT = 10
INITIAL_WAKE = 1
LLM_BACKOFF = 60
BOT_MIN_GAP = 30
RATE_WINDOW = 600
RATE_MAX = 6
FALLBACK_ANSWER = "Слушаю тебя. Что случилось?"
RESET_FLAG = Path(__file__).with_name("reset.flag")
LOG_PATH = Path(__file__).with_name("bot.log")

BOT_WORD_STRIP = ".,!?;:()[]{}\"'«»…—–-"


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
    """Возвращает текст после '!служебно' или None, если это не такая команда."""
    stripped = (text or "").strip()
    low = stripped.lower()
    prefix = "!служебно"
    if low == prefix:
        return ""
    if low.startswith(prefix) and len(low) > len(prefix) and not low[len(prefix)].isalnum():
        return stripped[len(prefix):].lstrip(" \t\n:,-—–").strip()
    return None


def contains_bot_word(text: str) -> bool:
    for token in (text or "").lower().split():
        cleaned = token.strip(BOT_WORD_STRIP)
        if cleaned == "бот" or cleaned.startswith("бот-"):
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
        if len(times) >= RATE_MAX:
            return False, "лимит 6 сообщений за 10 минут"
        return True, ""

    def load_chat(chat_id: int) -> None:
        nonlocal facts, norms, recent, next_wake, pending_count
        saved = memory.get_chat(chat_id)
        facts = saved["facts"]
        norms = saved["norms"]
        recent = [tuple(item) for item in saved["recent"]]
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

    def append_recent(role: str, text: str) -> None:
        recent.append((role, truncate(text)))
        if len(recent) > RECENT_LIMIT:
            del recent[:-RECENT_LIMIT]

    async def seed_context() -> None:
        nonlocal recent
        if active is None:
            return
        peer = build_peer(active) or active["chat_id"]
        try:
            messages = await client.get_messages(peer, limit=RECENT_LIMIT)
        except Exception:
            log.warning("Не удалось загрузить историю чата", exc_info=True)
            return

        items: list[tuple[str, str]] = []
        labels: dict[int, str] = {}
        for message in reversed(messages):
            text = (message.raw_text or "").strip()
            if not text:
                continue
            if text.startswith(ROBOT_PREFIX):
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
            items.append((role, truncate(text)))

        if items:
            recent = items[-RECENT_LIMIT:]
            persist()
            log.info("Контекст: загружено %s сообщений", len(recent))

    async def notify_owner(text: str) -> None:
        try:
            await client.send_message("me", ROBOT_PREFIX + text)
        except Exception:
            log.exception("Не удалось отправить подтверждение в Избранное")

    async def delete_command(event) -> None:
        try:
            await event.delete()
        except Exception:
            log.debug("Не удалось удалить команду", exc_info=True)

    async def activate(event) -> None:
        nonlocal active, silence_until
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
        await seed_context()
        log.info("Бот активирован в чате %s (%s)", title, event.chat_id)
        await notify_owner(
            f"Бот активирован в чате «{title}». Он будет просыпаться по контексту."
        )

    async def deactivate(event) -> None:
        nonlocal active, silence_until
        if active is None:
            await notify_owner("Бот уже остановлен.")
            return
        title = active.get("title") or str(active["chat_id"])
        active = None
        state.clear()
        silence_until = 0.0
        log.info("Бот остановлен")
        await notify_owner(f"Бот остановлен. В чате «{title}» он больше не проснётся.")

    async def reset_memory(event) -> None:
        nonlocal facts, norms, recent, next_wake, pending_count
        if active is None:
            await notify_owner("Нет активного чата: память очищать нечего.")
            return
        title = active.get("title") or str(active["chat_id"])
        facts = ""
        norms = ""
        recent = []
        next_wake = INITIAL_WAKE
        pending_count = 0
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
            "Приказ: action=%s target=%s minutes=%s wake_in=%s",
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
        else:
            await notify_owner("Не понял приказ.")

    async def send_bot_message(chat_id: int, text: str, reason: str = "") -> None:
        snapshot = active
        if snapshot is None:
            return
        peer = build_peer(snapshot) or chat_id
        sent = await client.send_message(peer, ROBOT_PREFIX + text)
        if sent and sent.id not in bot_message_ids:
            bot_message_ids.append(sent.id)
        now = time.time()
        last_bot_at[chat_id] = now
        sent_times.setdefault(chat_id, deque()).append(now)
        if reason == "conflict":
            last_conflict_at[chat_id] = now

    async def run_wake(force: bool = False, addressed: bool = False) -> None:
        nonlocal pending_count, next_wake, facts, norms, waking, llm_backoff_until
        if waking or active is None:
            return
        if not force and pending_count < next_wake:
            return
        if time.monotonic() < llm_backoff_until:
            return

        waking = True
        first = True
        try:
            while True:
                snapshot = active
                if snapshot is None or pending_count <= 0:
                    return
                if not force and pending_count < next_wake:
                    return
                force = False

                hint = addressed if first else False
                must_answer = hint and cfg.mandatory_reply
                first = False
                analyzed = pending_count

                chat_id = snapshot["chat_id"]
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
                )
                if decision is None:
                    llm_backoff_until = time.monotonic() + LLM_BACKOFF
                    log.warning("Пробуждение отложено на %s секунд", LLM_BACKOFF)
                    return

                if must_answer and (not decision.speak or not decision.message):
                    forced = await brain.answer_forced(cfg, facts, norms, list(recent))
                    if forced is None:
                        llm_backoff_until = time.monotonic() + LLM_BACKOFF
                        log.warning("Пробуждение отложено на %s секунд", LLM_BACKOFF)
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
                        mandatory=(must_answer and decision.reason == "answer"),
                    )
                    if not allowed:
                        should_send = False
                        log.info("Сообщение пропущено: %s", why)

                if should_send:
                    if active is not snapshot:
                        return
                    try:
                        await send_bot_message(
                            snapshot["chat_id"],
                            decision.message,
                            decision.reason,
                        )
                    except Exception:
                        log.exception("Не удалось отправить сообщение")

                if pending_count < next_wake:
                    return
                if time.monotonic() < llm_backoff_until:
                    return
        finally:
            waking = False

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

        if not is_active_chat(event) or not text:
            return

        append_recent("Я", text)
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
        if not text:
            return

        try:
            sender = await event.get_sender()
        except Exception:
            sender = None
        append_recent(sender_label(sender), text)
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
    await client.run_until_disconnected()


if __name__ == "__main__":
    if sys.platform == "win32":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
