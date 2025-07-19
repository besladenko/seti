"""SetiNews – автоматизированная городская новостная сеть для Telegram
=====================================================================

Версия *0.3* — автоматическое удаление рекламной подписи по **маске‑строке**
и генерация собственной подписи вида
`❤️ Подпишись на <Название канала> (<ссылка>)`.

Новые возможности
-----------------
1. **Маска‑строка** у каждого донора: передаётся при `/adddonor … [mask]` и
   хранится как исходный текст (не regex). При обработке постов этот текст
   безусловно вырезается (`str.replace`).
2. **Динамическая подпись**: формируется из `City.title` + `City.link`.
   Параметр `CUSTOM_SIGNATURE` в `.env` больше не нужен.
3. **Колонка `link` в таблице `cities`**: хранит публичный url
   (`https://t.me/...`, invite‑ссылку или `@username`). Заполняется
   автоматически из аргумента `/addcity`.
4. **Команда `/addcity`** принимает только один аргумент — ссылку/юзернейм.
   Название канала берётся из Telegram API.

> ⚠️ Так как структура БД изменилась (добавлена колонка `link`, убран
> `CUSTOM_SIGNATURE`), выполните `python setinews.py --init-db` на чистой
> базе или сделайте миграцию вручную.

"""
from __future__ import annotations

import asyncio
import os
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set

import httpx
import numpy as np
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telethon import TelegramClient, events, errors
from telethon.tl.functions.channels import JoinChannelRequest

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
)
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
    link = Column(String, nullable=True)  # публичная ссылка / invite
    auto_mode = Column(Boolean, default=True)

    donors = relationship("DonorChannel", back_populates="city")


class DonorChannel(Base):
    __tablename__ = "donor_channels"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(Integer, unique=True, nullable=False)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    mask_pattern = Column(Text, nullable=True)  # хранится как raw‑текст

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

    async def _get_token(self) -> str:
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

    async def detect_ads(self, text: str) -> bool:  # TODO
        return False

    async def paraphrase(self, text: str) -> str:
        return text


gigachat = GigaChatClient(GIGACHAT_CLIENT_ID, GIGACHAT_CLIENT_SECRET)

# ---------------------------------------------------------------------------
# 4. Утилиты обработки текста
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r"https?://\S+|t\.me/\S+|@\w+|#[\wА-Яа-я_]+")
AD_PHRASES = ["подпишись", "жми", "переходи", "смотри канал"]
DANGER = ["бпла", "ракетн", "тревог"]


def clean_text(text: str) -> str:
    return LINK_RE.sub("", text).strip()


def contains_ad(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in AD_PHRASES)


def contains_danger(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in DANGER)


async def is_duplicate(session: AsyncSession, city_id: int, text: str) -> bool:
    stmt = (
        select(Post.processed_text)
        .where(Post.city_id == city_id, Post.status == "published")
        .order_by(Post.id.desc())
        .limit(500)
    )
    corpus = [t for t in (await session.execute(stmt)).scalars() if t]
    if not corpus:
        return False
    tf = TfidfVectorizer().fit_transform(corpus + [text])
    sim = cosine_similarity(tf[-1], tf[:-1]).max()
    return bool(sim >= SIMILARITY_THRESHOLD)


def signature(city: City) -> str:
    if city.link:
        return f"❤️ Подпишись на {city.title} ({city.link})"
    return f"❤️ Подпишись на {city.title}"


# ---------------------------------------------------------------------------
# 5. Telegram — общий клиент и резолв ссылок
# ---------------------------------------------------------------------------

telethon_client = TelegramClient("setinews_parser", API_ID, API_HASH)


async def resolve_channel(link_or_username: str) -> tuple[int, str, str]:
    """Возвращает `(channel_id, title, link)` и вступает в канал при необходимости."""
    ent = await telethon_client.get_entity(link_or_username)
    channel_id = ent.id  # type: ignore[attr-defined]
    title = ent.title if hasattr(ent, "title") else ent.first_name  # type: ignore[attr-defined]

    # canonical link
    if isinstance(link_or_username, str) and link_or_username.startswith("http"):
        link = link_or_username
    elif str(link_or_username).startswith("@"):
        link = f"https://t.me/{link_or_username.lstrip('@')}"
    elif getattr(ent, "username", None):  # type: ignore[attr-defined]
        link = f"https://t.me/{ent.username}"
    else:
        link = None  # приватный канал без username

    try:
        await telethon_client(JoinChannelRequest(link_or_username))
    except (errors.UserAlreadyParticipantError, errors.InviteHashInvalidError):
        pass
    return int(channel_id), title or str(channel_id), link


# ---------------------------------------------------------------------------
# 6. Донор‑кэш
# ---------------------------------------------------------------------------

class DonorCache:
    def __init__(self) -> None:
        self.ids: Set[int] = set()
        self.expire = datetime.min

    async def refresh(self) -> None:
        if datetime.utcnow() < self.expire:
            return
        async with SessionLocal() as s:
            self.ids = set((await s.execute(select(DonorChannel.channel_id))).scalars())
        self.expire = datetime.utcnow() + timedelta(minutes=DONOR_CACHE_TTL_MIN)
        logger.info("Donor cache refreshed: %d ids", len(self.ids))


DONORS = DonorCache()


# ---------------------------------------------------------------------------
# 7. Парсер сообщений
# ---------------------------------------------------------------------------

@telethon_client.on(events.NewMessage())
async def on_message(event: events.NewMessage.Event):  # noqa: N802
    if event.chat_id is None or event.is_private:
        return
    await DONORS.refresh()
    if event.chat_id not in DONORS.ids:
        return

    async with SessionLocal() as s:
        donor = await s.scalar(select(DonorChannel).where(DonorChannel.channel_id == event.chat_id))
        if not donor:
            return
        city = donor.city
        text = event.message.message or ""

        # 1. remove donor mask (raw substring)
        if donor.mask_pattern:
            text = text.replace(donor.mask_pattern, "").strip()

        # 2. ad check
        is_ad = await gigachat.detect_ads(text) or contains_ad(text)
        duplicate = False
        processed = None
        status = "rejected" if is_ad else "pending"
        if not is_ad:
            text = clean_text(text)
            duplicate = await is_duplicate(s, city.id, text)
            if duplicate:
                status = "rejected"
            else:
                if not contains_danger(text):
                    text = await gigachat.paraphrase(text)
                processed = f"{text}\n\n{signature(city)}"

        # 3. media
        media_path = None
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
        s.add(post)
        await s.commit()
        logger.info("Post %d saved (%s)", post.id, status)

        if status == "pending" and city.auto_mode:
            await publish_post(s, post)


# ---------------------------------------------------------------------------
# 8. Боты: публикация и админ
# ---------------------------------------------------------------------------

news_bot = Bot(NEWS_BOT_TOKEN, parse_mode="HTML")
news_dp = Dispatcher(news_bot)


async def publish_post(session: AsyncSession, post: Post):
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


admin_bot = Bot(ADMIN_BOT_TOKEN, parse_mode="HTML")
admin_dp = Dispatcher(admin_bot)


async def is_admin(uid: int) -> bool:
    async with SessionLocal() as s:
        adm = await s.get(Admin, uid)
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
        ch_id, title, canonical = await resolve_channel(link)
    except Exception as e:  # noqa: BLE001
        return await msg.answer(f"Ошибка: {e}")
    async with SessionLocal() as s:
        city = City(channel_id=ch_id, title=title, link=canonical or link)
        s.add(city)
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
        donor = DonorChannel(channel_id=ch_id, title=title, city_id=city_id, mask_pattern=mask)
        s.add(donor)
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
            await msg.answer(f"ID {p.id}\n{(p.processed_textหรือ p.original_text)[:350]}…")


@admin_dp.message_handler(commands=["publish"])
async def cmd_publish(msg: types.Message):
    parts = msg.text.split()
    if len(parts) != 2 or not await is_admin(msg.from_user.id):
        return
    pid = int(parts[1])
    async with SessionLocal() as s:
        p = await s.get(Post, pid)
        if not p or p.status != "pending":
            return await msg.answer("⛔️ Не найдено")
        await publish_post(s, p)
        await msg.answer("✅ Опубликовано")


# ---------------------------------------------------------------------------
# 9. Планировщики
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


async def regular_donor_cache():
    while True:
        await DONORS.refresh()
        await asyncio
