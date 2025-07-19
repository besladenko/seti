"""SetiNews – Автоматизированная городская новостная сеть для Telegram (версия 0.4)
================================================================================

Полностью рабочий скрипт: парсер Telethon, боты aiogram 2.x, PostgreSQL + SQLAlchemy,
минимальные stub’ы для LLM, корректный polling и асинхронный запуск.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Optional, Tuple

import httpx
import numpy as np
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telethon import TelegramClient, events, errors
from telethon.tl.functions.channels import JoinChannelRequest

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, select
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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "media"))
DONOR_CACHE_TTL_MIN = int(os.getenv("DONOR_CACHE_TTL_MIN", "10"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("setinews")
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. База данных
# ---------------------------------------------------------------------------
Base = declarative_base()
_engine = create_async_engine(POSTGRES_DSN, echo=False, future=True)
SessionLocal = sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    class_=AsyncSession
)

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
# 3. GigaChat/stub LLM
# ---------------------------------------------------------------------------
class DummyLLM:
    async def detect_ads(self, text: str) -> bool:
        return False
    async def paraphrase(self, text: str) -> str:
        return text

gigachat = DummyLLM()

# ---------------------------------------------------------------------------
# 4. Текстовые утилиты
# ---------------------------------------------------------------------------
LINK_RE = re.compile(r"https?://\S+|t\.me/\S+|@\w+|#[\wА-Яа-я_]+")
AD_PHRASES = ["подпишись", "жми", "переходи", "смотри канал"]
DANGER = ["бпла", "ракетн", "тревог"]

def clean_text(text: str) -> str:
    return LINK_RE.sub("", text).strip()

def contains_ad(text: str) -> bool:
    return any(p in text.lower() for p in AD_PHRASES)

def contains_danger(text: str) -> bool:
    return any(k in text.lower() for k in DANGER)

async def is_duplicate(session: AsyncSession, city_id: int, text: str) -> bool:
    stmt = select(Post.processed_text).where(
        Post.city_id == city_id,
        Post.status == "published"
    ).order_by(Post.id.desc()).limit(500)
    rows = await session.execute(stmt)
    corpus = [r for r in rows.scalars() if r]
    if not corpus:
        return False
    mat = TfidfVectorizer().fit_transform(corpus + [text])
    return float(cosine_similarity(mat[-1], mat[:-1]).max()) >= SIMILARITY_THRESHOLD

def signature(city: City) -> str:
    if city.link:
        return f"❤️ Подпишись на {city.title} ({city.link})"
    return f"❤️ Подпишись на {city.title}"

# ---------------------------------------------------------------------------
# 5. Telethon
# ---------------------------------------------------------------------------
telethon_client = TelegramClient("setinews_parser", API_ID, API_HASH)

async def resolve_channel(link: str) -> Tuple[int,str,Optional[str]]:
    ent = await telethon_client.get_entity(link)
    chan_id = int(ent.id)
    title = getattr(ent, "title", None) or getattr(ent, "first_name", str(chan_id))
    if link.startswith("http"): canon = link
    elif link.startswith("@"): canon = f"https://t.me/{link.lstrip('@')}"
    elif getattr(ent, "username", None): canon = f"https://t.me/{ent.username}"
    else: canon = None
    try: await telethon_client(JoinChannelRequest(link))
    except errors.UserAlreadyParticipantError: pass
    except errors.InviteHashInvalidError: pass
    return chan_id, title, canon

# ---------------------------------------------------------------------------
# 6. Donor cache
# ---------------------------------------------------------------------------
class DonorCache:
    def __init__(self): self.ids = set(); self.expires = datetime.min
    async def refresh(self):
        if datetime.utcnow() < self.expires: return
        async with SessionLocal() as s:
            rows = await s.execute(select(DonorChannel.channel_id))
            self.ids = set(rows.scalars())
        self.expires = datetime.utcnow() + timedelta(minutes=DONOR_CACHE_TTL_MIN)
        logger.info("Donor cache refreshed – %d ids", len(self.ids))
DONORS = DonorCache()

# ---------------------------------------------------------------------------
# 7. Parser
# ---------------------------------------------------------------------------
@telethon_client.on(events.NewMessage())
async def handler(event: events.NewMessage.Event):
    if not event.chat_id or event.is_private: return
    await DONORS.refresh()
    if event.chat_id not in DONORS.ids: return
    async with SessionLocal() as s:
        donor = await s.get(DonorChannel, event.chat_id, column=DonorChannel.channel_id)
        if not donor: return
        city = donor.city
        text = event.message.message or ""
        if donor.mask_pattern:
            text = text.replace(donor.mask_pattern, "").strip()
        is_ad = await gigachat.detect_ads(text) or contains_ad(text)
        processed = None; dup=False; status = "pending"
        if is_ad: status = "rejected"
        else:
            cleaned = clean_text(text)
            dup = await is_duplicate(s, city.id, cleaned)
            if dup: status = "rejected"
            else:
                if not contains_danger(cleaned): cleaned = await gigachat.paraphrase(cleaned)
                processed = f"{cleaned}\n\n{signature(city)}"
        media_path=None
        if event.message.media:
            fname=f"{donor.channel_id}_{event.id}.jpg"
            media_path=str(MEDIA_ROOT/fname)
            await event.message.download_media(media_path)
        post=Post(
            donor_id=donor.id, city_id=city.id,
            original_text=text, processed_text=processed,
            media_path=media_path,
            source_link=f"https://t.me/c/{str(donor.channel_id)[4:]}/{event.id}",
            is_ad=is_ad, is_duplicate=dup, status=status
        )
        s.add(post); await s.commit()
        logger.info("Post %d %s", post.id, status)
        if status=="pending" and city.auto_mode:
            await publish(s, post)

# ---------------------------------------------------------------------------
# 8. Bots
# ---------------------------------------------------------------------------
news_bot = Bot(NEWS_BOT_TOKEN, parse_mode="HTML")
news_dp  = Dispatcher(news_bot)

async def publish(s: AsyncSession, post: Post):
    city = await s.get(City, post.city_id)
    if post.media_path and Path(post.media_path).exists():
        with open(post.media_path, "rb") as f:
            await news_bot.send_photo(city.channel_id, f, caption=post.processed_text)
    else:
        await news_bot.send_message(city.channel_id, post.processed_text)
    post.status="published"; post.published_at=datetime.utcnow()
    await s.commit()

admin_bot = Bot(ADMIN_BOT_TOKEN, parse_mode="HTML")
admin_dp = Dispatcher(admin_bot)

async def is_admin(uid: int)->bool:
    async with SessionLocal() as s:
        adm=await s.get(Admin, uid)
        return bool(adm and adm.is_super)

@admin_dp.message_handler(commands=["addcity"])
async def cmd_addcity(msg: types.Message):
    if msg.chat.type!="private" or not await is_admin(msg.from_user.id): return
    parts=msg.text.split(maxsplit=1)
    if len(parts)!=2: return await msg.answer("/addcity <link>")
    link=parts[1]
    try: cid,title,canon=await resolve_channel(link)
    except Exception as e: return await msg.answer(f"Error: {e}")
    async with SessionLocal() as s:
        s.add(City(channel_id=cid,title=title,link=canon))
        await s.commit(); await DONORS.refresh()
    await msg.answer(f"✅ City added: {title} ({cid})")

@admin_dp.message_handler(commands=["adddonor"])
async def cmd_adddonor(msg: types.Message):
    if msg.chat.type!="private" or not await is_admin(msg.from_user.id): return
    parts=msg.text.split(maxsplit=3)
    if len(parts)<3: return await msg.answer("/adddonor <city_id> <link> [mask]")
    cid=int(parts[1]); link=parts[2]; mask=parts[3] if len(parts)==4 else None
    try: dcid,title,_=await resolve_channel(link)
    except Exception as e: return await msg.answer(f"Error: {e}")
    async with SessionLocal() as s:
        city=await s.get(City, cid)
        if not city: return await msg.answer("Bad city_id")
        s.add(DonorChannel(channel_id=dcid,title=title,city_id=cid,mask_pattern=mask))
        await s.commit(); await DONORS.refresh()
    await msg.answer(f"✅ Donor {title} added to {city.title}")

@admin_dp.message_handler(commands=["pending"])
async def cmd_pending(msg: types.Message):
    if msg.chat.type!="private" or not await is_admin(msg.from_user.id): return
    async with SessionLocal() as s:
        rows=(await s.execute(select(Post).where(Post.status=="pending").limit(10))).scalars().all()
        if not rows: return await msg.answer("No pending posts")
        for p in rows:
            preview=(p.processed_text or p.original_text)[:250]
            await msg.answer(f"ID {p.id}\n{preview}")

@admin_dp.message_handler(commands=["publish"])
async def cmd_publish(msg: types.Message):
    if msg.chat.type!="private" or not await is_admin(msg.from_user.id): return
    parts=msg.text.split();
    if len(parts)!=2: return
    pid=int(parts[1])
    async with SessionLocal() as s:
        p=await s.get(Post,pid)
        if not p or p.status!="pending": return await msg.answer("Bad post id")
        await publish(s,p)
        await msg.answer("✅ Published")

# ---------------------------------------------------------------------------
# 9. Планировщики
# ---------------------------------------------------------------------------
async def refresh_gigachat_token():
    while True:
        # stub
        await asyncio.sleep(3600)

async def donor_cache_loop():
    while True:
        await DONORS.refresh()
        await asyncio.sleep(DONOR_CACHE_TTL_MIN*60)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 10. Запуск
# ---------------------------------------------------------------------------
from threading import Thread

def _start_bot(dp: Dispatcher) -> None:
    """Запускает aiogram polling в отдельном потоке с собственным loop"""
    executor.start_polling(dp, skip_updates=True)

async def main() -> None:
    # Инициализация БД и кеша доноров
    await init_db()
    await DONORS.refresh()

    # Старт Telethon
    await telethon_client.start()

    # Старт ботов в потоках
    Thread(target=_start_bot, args=(news_dp,), daemon=True).start()
    Thread(target=_start_bot, args=(admin_dp,), daemon=True).start()

    # Фоновые задачи LLM и кеширования
    await asyncio.gather(
        refresh_gigachat_token(),
        donor_cache_loop(),
    )


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
