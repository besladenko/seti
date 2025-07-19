"""
SetiNews – Automated City News Network for Telegram
Version 1.0.0  (complete single‑file implementation)

Key features
-------------
• Parses posts from multiple donor channels (Telethon).
• Cleans, deduplicates, paraphrases, classifies ads (GigaChat stub).
• Publishes to per‑city channels; auto‑mode or manual via admin‑bot (aiogram 3.7+).
• Uses PostgreSQL (async SQLAlchemy) with Channel‑level settings.
• Fully async; runs parser + two bots concurrently; uvloop & loguru.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Optional, Tuple, Set

import httpx
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telethon import TelegramClient, events, errors
from telethon.tl.functions.channels import JoinChannelRequest

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, select, Float
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------- AIogram (bots) ------------------------------------------
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BotCommand, BotCommandScopeDefault

# ---------------- Optional speed‑ups --------------------------------------
try:
    import uvloop  # type: ignore

    uvloop.install()
except ImportError:
    pass

# -------------------------------------------------------------------------
# 1. Configuration (.env)
# -------------------------------------------------------------------------
load_dotenv()

API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
NEWS_BOT_TOKEN = os.getenv("NEWS_BOT_TOKEN")
ADMIN_BOT_TOKEN = os.getenv("ADMIN_BOT_TOKEN")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "media"))
DONOR_CACHE_TTL_MIN = int(os.getenv("DONOR_CACHE_TTL_MIN", "10"))
CUSTOM_SIGNATURE = os.getenv("CUSTOM_SIGNATURE", "❤️ Подпишись на {city}")

MEDIA_ROOT.mkdir(exist_ok=True, parents=True)

# -------------------------------------------------------------------------
# 2. Database models (SQLAlchemy async)
# -------------------------------------------------------------------------
Base = declarative_base()
_engine = create_async_engine(POSTGRES_DSN, echo=False, future=True)
SessionLocal = sessionmaker(
    bind=_engine, expire_on_commit=False, class_=AsyncSession
)


class City(Base):
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(Integer, unique=True, nullable=False)
    link = Column(String)
    auto_mode = Column(Boolean, default=True)

    donors = relationship("DonorChannel", back_populates="city")
    settings = relationship("ChannelSetting", back_populates="city", uselist=False)


class DonorChannel(Base):
    __tablename__ = "donor_channels"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(Integer, unique=True, nullable=False)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    mask_pattern = Column(Text)

    city = relationship("City", back_populates="donors")
    posts = relationship("Post", back_populates="donor")


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    donor_id = Column(Integer, ForeignKey("donor_channels.id"), nullable=False)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)

    original_text = Column(Text)
    processed_text = Column(Text)
    media_path = Column(String)
    source_link = Column(String)

    is_ad = Column(Boolean, default=False)
    is_duplicate = Column(Boolean, default=False)
    status = Column(String, default="pending")  # pending | published | rejected

    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime)

    donor = relationship("DonorChannel", back_populates="posts")


class Admin(Base):
    __tablename__ = "admins"

    tg_id = Column(Integer, primary_key=True)
    username = Column(String)
    is_super = Column(Boolean, default=False)


class ChannelSetting(Base):
    __tablename__ = "channel_settings"

    city_id = Column(Integer, ForeignKey("cities.id"), primary_key=True)
    min_words_for_dedup = Column(Integer, default=4)
    dedup_threshold = Column(Float, default=SIMILARITY_THRESHOLD)

    city = relationship("City", back_populates="settings")


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("DB schema ready")


async def is_admin(tg_id: int) -> bool:
    async with SessionLocal() as s:
        row = await s.execute(select(Admin.tg_id).where(Admin.tg_id == tg_id))
        return row.scalar() is not None


# -------------------------------------------------------------------------
# 3. LLM stub (to be replaced by GigaChat client)
# -------------------------------------------------------------------------


class DummyLLM:
    async def detect_ads(self, text: str) -> bool:
        return False

    async def paraphrase(self, text: str) -> str:
        return text


gigachat = DummyLLM()


# -------------------------------------------------------------------------
# 4. Text utilities
# -------------------------------------------------------------------------
LINK_RE = re.compile(r"https?://\S+|t\.me/\S+|@\w+|#[\wА-Яа-я_]+")
AD_PHRASES = ["подпишись", "жми", "переходи", "смотри канал"]
DANGER = ["бпла", "ракет", "тревог"]


def clean_text(text: str) -> str:
    return LINK_RE.sub("", text).strip()


def contains_ad(text: str) -> bool:
    return any(p in text.lower() for p in AD_PHRASES)


def contains_danger(text: str) -> bool:
    return any(k in text.lower() for k in DANGER)


async def is_duplicate(session: AsyncSession, city_id: int, text: str, threshold: float) -> bool:
    stmt = (
        select(Post.processed_text)
        .where(Post.city_id == city_id, Post.status == "published")
        .order_by(Post.id.desc())
        .limit(500)
    )
    rows = await session.execute(stmt)
    corpus = [r for r in rows.scalars() if r and len(r.split()) >= 4]
    if not corpus:
        return False
    mat = TfidfVectorizer().fit_transform(corpus + [text])
    return float(cosine_similarity(mat[-1], mat[:-1]).max()) >= threshold


# -------------------------------------------------------------------------
# 5. Telethon parser
# -------------------------------------------------------------------------
telethon_client = TelegramClient("setinews_parser", API_ID, API_HASH)


async def resolve_channel(link: str) -> Tuple[int, str, Optional[str]]:
    ent = await telethon_client.get_entity(link)
    chan_id = int(ent.id)
    title = getattr(ent, "title", None) or getattr(ent, "first_name", str(chan_id))
    if link.startswith("http"):
        canon = link
    elif link.startswith("@"):
        canon = f"https://t.me/{link.lstrip('@')}"
    elif getattr(ent, "username", None):
        canon = f"https://t.me/{ent.username}"
    else:
        canon = None
    try:
        await telethon_client(JoinChannelRequest(link))
    except (errors.UserAlreadyParticipantError, errors.InviteHashInvalidError):
        pass
    return chan_id, title, canon


# -------------------------------------------------------------------------
# 6. Donor cache (refresh every N minutes)
# -------------------------------------------------------------------------


class DonorCache:
    def __init__(self):
        self.ids: Set[int] = set()
        self.expires = datetime.min

    async def refresh(self):
        if datetime.utcnow() < self.expires:
            return
        async with SessionLocal() as s:
            rows = await s.execute(select(DonorChannel.channel_id))
            self.ids = set(rows.scalars())
        self.expires = datetime.utcnow() + timedelta(minutes=DONOR_CACHE_TTL_MIN)
        logger.info(f"Donor cache refreshed ({len(self.ids)} ids)")


DONORS = DonorCache()


# -------------------------------------------------------------------------
# 7. Parser handler
# -------------------------------------------------------------------------


@telethon_client.on(events.NewMessage())
async def on_new_message(event: events.NewMessage.Event):
    if event.chat_id is None or event.is_private:
        return
    await DONORS.refresh()
    if event.chat_id not in DONORS.ids:
        return

    async with SessionLocal() as s:
        donor = (
            await s.execute(
                select(DonorChannel).where(DonorChannel.channel_id == event.chat_id)
            )
        ).scalar_one()
        city = donor.city
        text = event.message.message or ""
        if donor.mask_pattern:
            text = re.sub(donor.mask_pattern, "", text, flags=re.S).strip()
        if not text:
            return

        is_ad = await gigachat.detect_ads(text) or contains_ad(text)
        processed: Optional[str] = None
        dup = False
        status = "pending"

        thresh = city.settings.dedup_threshold if city.settings else SIMILARITY_THRESHOLD

        if is_ad:
            status = "rejected"
        else:
            cleaned = clean_text(text)
            dup = await is_duplicate(s, city.id, cleaned, thresh)
            if dup:
                status = "rejected"
            else:
                if not contains_danger(cleaned):
                    cleaned = await gigachat.paraphrase(cleaned)
                processed = f"{cleaned}\n\n{CUSTOM_SIGNATURE.format(city=city.title)}"

        media_path: Optional[str] = None
        if event.message.media:
            fname = f"{donor.channel_id}_{event.id}.jpg"
            media_path = str(MEDIA_ROOT / fname)
            await event.message.download_media(media_path)

        post = Post(
            donor_id=donor.id,
            city_id=city.id,
            original_text=text,
            processed_text=processed,
            media_path=media_path,
            source_link=f"https://t.me/c/{str(donor.channel_id)[4:]}/{event.id}",
            is_ad=is_ad,
            is_duplicate=dup,
            status=status,
        )
        s.add(post)
        await s.commit()
        logger.debug(f"Saved post {post.id} status={status}")

        if status == "pending" and city.auto_mode:
            await publish(s, post)  # autopublish


# -------------------------------------------------------------------------
# 8. Bots (aiogram 3.7+)
# -------------------------------------------------------------------------
BOT_DEFAULTS = DefaultBotProperties(parse_mode=ParseMode.HTML)

news_bot = Bot(NEWS_BOT_TOKEN, default=BOT_DEFAULTS)
admin_bot = Bot(ADMIN_BOT_TOKEN, default=BOT_DEFAULTS)

news_dp = Dispatcher()
admin_dp = Dispatcher()
admin_rt = Router()
admin_dp.include_router(admin_rt)

# ---- Bot commands --------------------------------------------------------
ADMIN_COMMANDS = [
    BotCommand("addcity", "Добавить городской канал"),
    BotCommand("adddonor", "Привязать донор"),
    BotCommand("toggleauto", "Вкл/выкл авто‑режим"),
    BotCommand("pending", "Список ожидающих"),
    BotCommand("publish", "Опубликовать пост"),
    BotCommand("help", "Справка"),
]


async def setup_bot_commands():
    await admin_bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeDefault())


# ---- Publish helper ------------------------------------------------------
async def publish(session: AsyncSession, post: Post) -> None:
    city = await session.get(City, post.city_id)
    if not city:
        return
    if post.media_path and Path(post.media_path).exists():
        with open(post.media_path, "rb") as f:
            await news_bot.send_photo(city.channel_id, f, caption=post.processed_text)
    else:
        await news_bot.send_message(city.channel_id, post.processed_text)
    post.status = "published"
    post.published_at = datetime.utcnow()
    await session.commit()
    logger.info(f"Published post {post.id} to {city.title}")


# ---- Admin handlers ------------------------------------------------------
HELP_TEXT = (
    "<b>Админ‑команды SetiNews</b>\n\n"
    "/addcity &lt;@channel&gt; — зарегистрировать городской канал\n"
    "/adddonor &lt;city_id&gt; &lt;@source&gt; [mask] — добавить донор\n"
    "/toggleauto &lt;city_id&gt; — переключить авто‑режим\n"
    "/pending — показать 10 ожидающих\n"
    "/publish &lt;post_id&gt; — опубликовать ручками\n"
)


@admin_rt.message(CommandStart())
async def cmd_start(msg: Message):
    await msg.answer("Привет! /help — список команд.")


@admin_rt.message(Command("help"))
async def cmd_help(msg: Message):
    await msg.answer(HELP_TEXT)


@admin_rt.message(Command("addcity"))
async def cmd_addcity(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) != 2:
        await msg.answer("Usage: /addcity <@channel>")
        return
    link = parts[1]
    try:
        chan_id, title, canon = await resolve_channel(link)
    except Exception as e:
        await msg.answer(f"Error: {e}")
        return
    async with SessionLocal() as s:
        s.add(City(channel_id=chan_id, title=title, link=canon))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ Added city {title} ({chan_id})")


@admin_rt.message(Command("adddonor"))
async def cmd_adddonor(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=3)
    if len(parts) < 3:
        await msg.answer("Usage: /adddonor <city_id> <@source> [mask]")
        return
    city_id = int(parts[1])
    link = parts[2]
    mask = parts[3] if len(parts) == 4 else None
    try:
        cid, title, _ = await resolve_channel(link)
    except Exception as e:
        await msg.answer(str(e))
        return
    async with SessionLocal() as s:
        city = await s.get(City, city_id)
        if not city:
            await msg.answer("Bad city_id")
            return
        s.add(DonorChannel(channel_id=cid, title=title, city_id=city_id, mask_pattern=mask))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ Donor {title} linked to {city.title}")


@admin_rt.message(Command("toggleauto"))
async def cmd_toggle_auto(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) != 2:
        await msg.answer("Usage: /toggleauto <city_id>")
        return
    city_id = int(parts[1])
    async with SessionLocal() as s:
        city = await s.get(City, city_id)
        if not city:
            await msg.answer("Bad city_id")
            return
        city.auto_mode = not city.auto_mode
        await s.commit()
        await msg.answer(f"Auto‑mode for {city.title}: {'ON' if city.auto_mode else 'OFF'}")


@admin_rt.message(Command("pending"))
async def cmd_pending(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    async with SessionLocal() as s:
        rows = (
            await s.execute(select(Post).where(Post.status == "pending").limit(10))
        ).scalars().all()
        if not rows:
            await msg.answer("No pending posts")
            return
        for p in rows:
            preview = (p.processed_text or p.original_text)[:250]
            await msg.answer(f"ID {p.id}\n{preview}...")


@admin_rt.message(Command("publish"))
async def cmd_publish(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) != 2:
        return
    post_id = int(parts[1])
    async with SessionLocal() as s:
        post = await s.get(Post, post_id)
        if not post or post.status != "pending":
            await msg.answer("Bad post_id")
            return
        await publish(s, post)
        await msg.answer("✅ Published")


@admin_rt.message()
async def ignore(msg: Message):
    pass


# -------------------------------------------------------------------------
# 9. Background tasks
# -------------------------------------------------------------------------
async def refresh_gigachat_token():
    while True:
        # Placeholder: renew token if using real API
        await asyncio.sleep(3600)


async def donor_cache_loop():
    while True:
        await DONORS.refresh()
        await asyncio.sleep(DONOR_CACHE_TTL_MIN * 60)


# -------------------------------------------------------------------------
# 10. Run helpers
# -------------------------------------------------------------------------

def _start_bot(dp: Dispatcher, bot: Bot):
    async def runner():
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, handle_signals=False)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner())

async def main():
    await init_db()
    await setup_bot_commands()
    await DONORS.refresh()
    await telethon_client.start()

    Thread(target=_start_bot, args=(news_dp, news_bot), daemon=True).start()
    Thread(target=_start_bot, args=(admin_dp, admin_bot), daemon=True).start()

    await asyncio.gather(refresh_gigachat_token(), donor_cache_loop())


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-db", action="store_true")
    args = parser.parse_args()

    if args.init_db:
        asyncio.run(init_db())
    else:
        # Graceful shutdown signals on main thread
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: asyncio.get_event_loop().stop())
        asyncio.run(main())


if __name__ == "__main__":
    cli()
