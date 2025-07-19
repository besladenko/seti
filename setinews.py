"""SetiNews – автоматизированная городская новостная сеть для Telegram
=====================================================================

Версия 0.3.1 — исправлена опечатка (`หรือ` → `or`) из‑за которой скрипт не
компилировался, плюс мелкие доработки финального блока.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set

import httpx
import numpy as np
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telethon import TelegramClient, errors, events
from telethon.tl.functions.channels import JoinChannelRequest

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------------------------------------------------------------------
# 1. Конфигурация
# ---------------------------------------------------------------------------

load_dotenv()

API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
NEWS_BOT_TOKEN = os.getenv("NEWS_BOT_TOKEN")
ADMIN_BOT_TOKEN = os.getenv("ADMIN_BOT_TOKEN")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
GIGACHAT_CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID")
GIGACHAT_CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "media"))
DONOR_CACHE_TTL_MIN = int(os.getenv("DONOR_CACHE_TTL_MIN", "10"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setinews")
MEDIA_ROOT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 2. База данных
# ---------------------------------------------------------------------------

Base = declarative_base()
_engine = create_async_engine(POSTGRES_DSN, echo=False, future=True)
SessionLocal = sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


class City(Base):
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(Integer, unique=True, nullable=False)
    link = Column(String)
    auto_mode = Column(Boolean, default=True)

    donors = relationship("DonorChannel", back_populates="city")


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
    status = Column(String, default="pending")

    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime)

    donor = relationship("DonorChannel", back_populates="posts")


class Admin(Base):
    __tablename__ = "admins"

    tg_id = Column(Integer, primary_key=True)
    username = Column(String)
    is_super = Column(Boolean, default=False)


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB schema (re)created ✅")

# ---------------------------------------------------------------------------
# 3. GigaChat (заглушка)
# ---------------------------------------------------------------------------

class GigaChatClient:
    BASE = "https://gigachat.devices.sberbank.ru/api/v1"

    def __init__(self, cid: str, secret: str) -> None:
        self.cid = cid
        self.secret = secret
        self._tok: str | None = None

    async def _get_token(self) -> str:  # noqa: D401
        if self._tok:
            return self._tok
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.BASE}/oauth/token",
                data={"grant_type": "client_credentials", "client_id": self.cid, "client_secret": self.secret},
            )
            r.raise_for_status()
            self._tok = r.json()["access_token"]
            return self._tok

    async def detect_ads(self, text: str) -> bool:
        return False  # TODO: заменить вызовом модели

    async def paraphrase(self, text: str) -> str:
        return text


gigachat = GigaChatClient(GIGACHAT_CLIENT_ID, GIGACHAT_CLIENT_SECRET)

# ---------------------------------------------------------------------------
# 4. Утилиты
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r"https?://\S+|t\.me/\S+|@\w+|#[\wА-Яа-я_]+")
AD_PHRASES = ["подпишись", "жми", "переходи", "смотри канал"]
DANGER = ["бпла", "ракетн", "тревог"]


def clean_text(txt: str) -> str:
    return LINK_RE.sub("", txt).strip()


def contains_ad(txt: str) -> bool:
    t = txt.lower()
    return any(p in t for p in AD_PHRASES)


def contains_danger(txt: str) -> bool:
    t = txt.lower()
    return any(k in t for k in DANGER)


async def is_duplicate(session: AsyncSession, city_id: int, txt: str) -> bool:
    stmt = (
        select(Post.processed_text)
        .where(Post.city_id == city_id, Post.status == "published")
        .order_by(Post.id.desc())
        .limit(500)
    )
    corpus = [t for t in (await session.execute(stmt)).scalars() if t]
    if not corpus:
        return False
    tf = TfidfVectorizer().fit_transform(corpus + [txt])
    return bool(cosine_similarity(tf[-1], tf[:-1]).max() >= SIMILARITY_THRESHOLD)


def signature(city: City) -> str:
    return f"❤️ Подпишись на {city.title} ({city.link})" if city.link else f"❤️ Подпишись на {city.title}"

# ---------------------------------------------------------------------------
# 5. Telegram — резолв ссылок и клиент
# ---------------------------------------------------------------------------

telethon_client = TelegramClient("setinews_parser", API_ID, API_HASH)


async def resolve_channel(link: str) -> tuple[int, str, str | None]:
    ent = await telethon_client.get_entity(link)
    chan_id = ent.id  # type: ignore[attr-defined]
    title = ent.title if hasattr(ent, "title") else ent.first_name  # type: ignore[attr-defined]

    if link.startswith("http"):
        canonical = link
    elif link.startswith("@"):
        canonical = f"https://t.me/{link.lstrip('@')}"
    elif getattr(ent, "username", None):  # type: ignore[attr-defined]
        canonical = f"https://t.me/{ent.username}"
    else:
        canonical = None

    try:
        await telethon_client(JoinChannelRequest(link))
    except (errors.UserAlreadyParticipantError, errors.InviteHashInvalidError):
        pass
    return int(chan_id), title or str(chan_id), canonical

# ---------------------------------------------------------------------------
# 6. Донор‑кэш
# ---------------------------------------------------------------------------

class DonorCache:
    def __init__(self) -> None:
        self.ids: Set[int] = set()
        self.expires = datetime.min

    async def refresh(self) -> None:
        if datetime.utcnow() < self.expires:
            return
        async with SessionLocal() as s:
            self.ids = set((await s.execute(select(DonorChannel.channel_id))).scalars())
        self.expires = datetime.utcnow() + timedelta(minutes=DONOR_CACHE_TTL_MIN)
        logger.info("Donor cache refreshed – %d ids", len(self.ids))


DONORS = DonorCache()

# ---------------------------------------------------------------------------
# 7. Парсер Телеграма
# ---------------------------------------------------------------------------

@telethon_client.on(events.NewMessage())
async def _on_msg(event: events.NewMessage.Event):
    if event.chat_id is None or event.is_private:
        return
    await DONORS.refresh()
    if event.chat_id not in DONORS.ids:
        return

    async with SessionLocal() as session:
        donor = await session.scalar(select(DonorChannel).where(DonorChannel.channel_id == event.chat_id))
        if donor is None:
            return
        city = donor.city
        txt = event.message.message or ""

        if donor.mask_pattern:
            txt = txt.replace(donor.mask_pattern, "").strip()

        is_ad = await gigachat.detect_ads(txt) or contains_ad(txt)
        duplicate = False
        processed = None
        status = "rejected" if is_ad else "pending"
        if not is_ad:
            txt = clean_text(txt)
            duplicate = await is_duplicate(session, city.id, txt)
            if duplicate:
                status = "rejected"
            else:
                if not contains_danger(txt):
                    txt = await gigachat.paraphrase(txt)
                processed = f"{txt}\n\n{signature(city)}"

        media_path: str | None = None
        if event.message.media:
            fname = f"{donor.channel_id}_{event.id}.jpg"
            media_path = str(MEDIA_ROOT / fname)
            await event.message.download_media(media_path)

        post = Post(
            donor_id=donor.id,
            city_id=city.id,
            original_text=event.message.message or "",
            processed_text=processed,
            media_path=media_path,
            source_link=f"https://t.me/c/{str(donor.channel_id)[4:]}/{event.id}",
            is_ad=is_ad,
            is_duplicate=duplicate,
            status=status,
        )
        session.add(post)
        await session.commit()
        logger.info("Post %d saved (%s)", post.id, status)

        if status == "pending" and city.auto_mode:
            await publish_post(session, post)

# ---------------------------------------------------------------------------
# 8. Боты
# ---------------------------------------------------------------------------

news_bot = Bot(NEWS_BOT_TOKEN, parse_mode="HTML")
news_dp = Dispatcher(news_bot)


async def publish_post(session: AsyncSession, post: Post) -> None:
    city = await session.get(City, post.city_id)
    if not city:
        return
    try:
        if post.media_path and Path(post.media_path).exists():
            with open(post.media_path, "rb") as f:
                await news_bot.send_photo(city.channel_id, f, caption=post.processed_text)
        else:
            await news_bot.send_message(city.channel_id, post.processed_text)
        post.status = "published"
        post.published_at = datetime.utcnow()
        await session.commit()
    except Exception as e:  # noqa: BLE001
        logger.error("Publish failed: %s", e)


# ---------------------------------------------------------------------------
# 9. Админ‑бот
# ---------------------------------------------------------------------------

admin_bot = Bot(ADMIN_BOT_TOKEN, parse_mode="HTML")
admin_dp = Dispatcher(admin_bot)


async def is_admin(user_id: int) -> bool:
    async with SessionLocal() as s:
        adm = await s.get(Admin, user_id)
        return bool(adm)


@admin_dp.message_handler(commands=["addcity"])
async def cmd_add_city(msg: types.Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    args = msg.text.split(maxsplit=1)
    if len(args) != 2:
        return await msg.answer("Формат: /addcity <@username|https://t.me/...>")
    link = args[1]
    try:
        ch_id, title, canon = await resolve_channel(link)
    except Exception as e:  # noqa: BLE001
        return await msg.answer(f"Ошибка: {e}")
    async with SessionLocal() as s:
        s.add(City(channel_id=ch_id, title=title, link=canon or link))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ Город {title} добавлен")


@admin_dp.message_handler(commands=["adddonor"])
async def cmd_add_donor(msg: types.Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=3)
    if len(parts) < 3:
        return await msg.answer("Формат: /adddonor <city_id> <ссылка> [mask]")
    city_id = int(parts[1])
    link = parts[2]
    mask = parts[3] if len(parts) == 4 else None
    try:
        ch_id, title, _ = await resolve_channel(link)
    except Exception as e:  # noqa: BLE001
        return await msg.answer(f"Ошибка: {e}")
    async with SessionLocal() as s:
        city = await s.get(City, city_id)
        if not city:
            return await msg.answer("⛔️ неправильный city_id")
        s.add(DonorChannel(channel_id=ch_id, title=title, city_id=city_id, mask_pattern=mask))
        await s.commit()
        await DONORS.refresh()
    await msg.answer("✅ Донор добавлен")


@admin_dp.message_handler(commands=["pending"])
async def cmd_pending(msg: types.Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    async with SessionLocal() as s:
        rows: List[Post] = (
            await s.execute(select(Post).where(Post.status == "pending").limit(15))
        ).scalars().all()
        if not rows:
            return await msg.answer("Пусто")
        for p in rows:
            preview = (p.processed_text or p.original_text)[:350]
            await msg.answer(f"ID {p.id}\n{preview}…")


@admin_dp.message_handler(commands=["publish"])
async def cmd_publish(msg: types.Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) != 2:
        return
    pid = int(parts[1])
    async with SessionLocal() as s:
        p = await s.get(Post, pid)
        if not p or p.status != "pending":
            return await msg.answer("⛔️ Не найдено")
        await publish_post(s, p)
        await msg.answer("✅ Опубликовано")


# ---------------------------------------------------------------------------
# 10. Планировщики
# ---------------------------------------------------------------------------

async def refresh_gigachat_token():
    while True:
        gigachat._tok = None
        try:
            await gigachat._get_token()
            logger.info("GigaChat token refreshed")
        except Exception:
            logger.warning("GigaChat token refresh failed")
        await asyncio.sleep(3600)


async def donor_cache_loop():
    while True:
        await DONORS.refresh()
        await asyncio.sleep(DONOR_CACHE_TTL_MIN * 60)


# ---------------------------------------------------------------------------
# 11. Запуск
# ---------------------------------------------------------------------------

async def main() -> None:
    # проверяем соединение с БД
    async with _engine.begin():
        pass

    await DONORS.refresh()

    # aiogram 2.x имеет coroutine start_polling — запускаем его прямо в главном loop
    tasks = [
        asyncio.create_task(news_dp.start_polling(skip_updates=True)),
        asyncio.create_task(admin_dp.start_polling(skip_updates=True)),
        telethon_client.start(),
        refresh_gigachat_token(),
        donor_cache_loop(),
    ]
    await asyncio.gather(*tasks)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-db", action="store_true")
    args = parser.parse_args()
    if args.init_db:
        asyncio.run(init_db())
    else:
        asyncio.run(main())


if __name__ == "__main__":
    cli()
