from __future__ import annotations

import asyncio
import logging
import sys
import time
import warnings
from collections import deque
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
RESET_FLAG = Path(__file__).with_name("reset.flag")


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


async def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    waking = False
    llm_backoff_until = 0.0

    log.info("Аккаунт: %s (id=%s)", me.username or me.first_name or own_id, own_id)
    log.info("LLM: %s (%s)", cfg.llm_model, cfg.llm_base_url)

    def is_active_chat(event) -> bool:
        return active is not None and event.chat_id == active["chat_id"]

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
        nonlocal active
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
        await seed_context()
        log.info("Бот активирован в чате %s (%s)", title, event.chat_id)
        await notify_owner(
            f"Бот активирован в чате «{title}». Он будет просыпаться по контексту."
        )

    async def deactivate(event) -> None:
        nonlocal active
        if active is None:
            await notify_owner("Бот уже остановлен.")
            return
        title = active.get("title") or str(active["chat_id"])
        active = None
        state.clear()
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

    async def send_bot_message(chat_id: int, text: str) -> None:
        snapshot = active
        if snapshot is None:
            return
        peer = build_peer(snapshot) or chat_id
        sent = await client.send_message(peer, ROBOT_PREFIX + text)
        if sent and sent.id not in bot_message_ids:
            bot_message_ids.append(sent.id)

    async def run_wake(force: bool = False) -> None:
        nonlocal pending_count, next_wake, facts, norms, waking, llm_backoff_until
        if waking or active is None:
            return
        if not force and pending_count < next_wake:
            return
        if time.monotonic() < llm_backoff_until:
            return

        waking = True
        try:
            while True:
                snapshot = active
                if snapshot is None or pending_count <= 0:
                    return
                if not force and pending_count < next_wake:
                    return
                force = False

                analyzed = pending_count
                decision = await brain.decide(cfg, facts, norms, list(recent), analyzed)
                if decision is None:
                    llm_backoff_until = time.monotonic() + LLM_BACKOFF
                    log.warning("Пробуждение отложено на %s секунд", LLM_BACKOFF)
                    return

                pending_count = max(0, pending_count - analyzed)
                next_wake = decision.next_wake
                if decision.memory:
                    facts = decision.memory
                if decision.norms:
                    norms = decision.norms
                persist()

                log.info(
                    "Пробуждение: speak=%s next_wake=%s pending=%s",
                    decision.speak,
                    next_wake,
                    pending_count,
                )

                if decision.speak:
                    if active is not snapshot:
                        return
                    try:
                        await send_bot_message(snapshot["chat_id"], decision.message)
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

        if not is_active_chat(event) or not text:
            return

        append_recent("Я", text)
        persist()
        pending_count += 1

        reply_to_bot = bool(event.reply_to_msg_id and event.reply_to_msg_id in bot_message_ids)
        await run_wake(force=reply_to_bot)

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

        reply_to_bot = bool(event.reply_to_msg_id and event.reply_to_msg_id in bot_message_ids)
        await run_wake(force=reply_to_bot)

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
