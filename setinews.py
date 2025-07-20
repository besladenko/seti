"""
SetiNews — автоматизированная городская новостная сеть для Telegram

Особенности:
- mask_pattern для донора (очистка поста)
- админ‑бот с командами: /addcity, /adddonor, /pending, /publish, /setmask, /autopost
- публикация с подписью
- TF-IDF дедупликация, фильтр рекламы
- PostgreSQL, Telethon, aiogram 3.x, loguru

Для запуска: pip install aiogram[fast], telethon, loguru, python-dotenv, sqlalchemy[asyncio], asyncpg, scikit-learn
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Optional, Set, Tuple

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, select
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from telethon import TelegramClient, events, errors
from telethon.tl.functions.channels import JoinChannelRequest

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message, BotCommand, BotCommandScopeDefault

# ---------------------------------------------------------------------------
# 1. Config & logging
# ---------------------------------------------------------------------------
load_dotenv()
API_ID               = int(os.getenv("TG_API_ID", 0))
API_HASH             = os.getenv("TG_API_HASH", "")
NEWS_BOT_TOKEN       = os.getenv("NEWS_BOT_TOKEN", "")
ADMIN_BOT_TOKEN      = os.getenv("ADMIN_BOT_TOKEN", "")
POSTGRES_DSN         = os.getenv("POSTGRES_DSN", "postgresql+asyncpg://user:pass@localhost/db")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
MEDIA_ROOT           = Path(os.getenv("MEDIA_ROOT", "media"))
DONOR_CACHE_TTL_MIN  = int(os.getenv("DONOR_CACHE_TTL_MIN", "10"))
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
logger.add("setinews.log", rotation="10 MB", enqueue=True, level="INFO")

# ---------------------------------------------------------------------------
# 2. Database
# ---------------------------------------------------------------------------
Base = declarative_base()
_engine = create_async_engine(POSTGRES_DSN, echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, class_=AsyncSession)

class City(Base):
    __tablename__ = "cities"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(Integer, unique=True, nullable=False)
    link = Column(String)
    auto_mode = Column(Boolean, default=True)
    donors = relationship("DonorChannel", back_populates="city", cascade="all,delete")

class DonorChannel(Base):
    __tablename__ = "donor_channels"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(Integer, unique=True, nullable=False)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    mask_pattern = Column(Text)
    city = relationship("City", back_populates="donors")
    posts = relationship("Post", back_populates="donor", cascade="all,delete")

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
    logger.success("DB schema ready ✅")

async def is_admin(tg_id: int) -> bool:
    async with SessionLocal() as s:
        row = await s.execute(select(Admin.tg_id).where(Admin.tg_id == tg_id))
        return row.scalar() is not None

# ---------------------------------------------------------------------------
# 3. LLM-Stub
# ---------------------------------------------------------------------------
class DummyLLM:
    async def detect_ads(self, text: str) -> bool: return False
    async def paraphrase(self, text: str) -> str: return text

gigachat = DummyLLM()

# ---------------------------------------------------------------------------
# 4. Text helpers
# ---------------------------------------------------------------------------
LINK_RE    = re.compile(r"https?://\S+|t\.me/\S+|@\w+|#[\wА-Яа-я_]+")
AD_PHRASES = ["подпишись", "жми", "переходи", "смотри канал"]
DANGER     = ["бпла", "ракетн", "тревог"]

def clean_text(text: str) -> str:
    return LINK_RE.sub("", text).strip()

def contains_ad(text: str) -> bool:
    return any(p in text.lower() for p in AD_PHRASES)

def contains_danger(text: str) -> bool:
    return any(k in text.lower() for k in DANGER)

async def is_duplicate(session: AsyncSession, city: City, text: str) -> bool:
    rows = await session.execute(
        select(Post.processed_text).where(Post.city_id == city.id, Post.status == "published").order_by(Post.id.desc()).limit(500)
    )
    corpus = [r for r in rows.scalars() if r]
    if not corpus:
        return False
    mat = TfidfVectorizer().fit_transform(corpus + [text])
    return float(cosine_similarity(mat[-1], mat[:-1]).max()) >= SIMILARITY_THRESHOLD

def signature(city: City) -> str:
    return f"❤️ Подпишись на {city.title} ({city.link})" if city.link else f"❤️ Подпишись на {city.title}"

# ---------------------------------------------------------------------------
# 5. Telethon
# ---------------------------------------------------------------------------
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
    except errors.UserAlreadyParticipantError:
        pass
    except errors.InviteHashInvalidError:
        pass
    return chan_id, title, canon

# ---------------------------------------------------------------------------
# 6. Donor cache
# ---------------------------------------------------------------------------
class DonorCache:
    def __init__(self):
        self.ids: Set[int] = set()
        self.expires = datetime.min
    async def refresh(self):
        if datetime.utcnow() < self.expires:
            return
        async with SessionLocal() as s:
            self.ids = set((await s.execute(select(DonorChannel.channel_id))).scalars())
        self.expires = datetime.utcnow() + timedelta(minutes=DONOR_CACHE_TTL_MIN)
        logger.info(f"Donor cache refreshed – {len(self.ids)} ids")
DONORS = DonorCache()

# ---------------------------------------------------------------------------
# 7. Donor message handler (с поддержкой маски)
# ---------------------------------------------------------------------------
@telethon_client.on(events.NewMessage())
async def on_new_message(event: events.NewMessage.Event):
    if event.chat_id is None or event.is_private:
        return
    await DONORS.refresh()
    if event.chat_id not in DONORS.ids:
        return

    async with SessionLocal() as s:
        donor = (await s.execute(
            select(DonorChannel).where(DonorChannel.channel_id == event.chat_id)
        )).scalar_one_or_none()
        if donor is None:
            return
        city = donor.city
        text = event.message.message or ""

        # Применяем маску (удаляет ненужную подпись и т.д.)
        if donor.mask_pattern:
            try:
                text = re.sub(donor.mask_pattern, '', text, flags=re.IGNORECASE).strip()
            except re.error as e:
                logger.warning(f"Некорректная маска у донора {donor.title}: {e}")
        if not text:
            return  # если текст пуст — не публикуем!

        is_ad = await gigachat.detect_ads(text) or contains_ad(text)
        processed = None
        dup = False
        status = "pending"
        if is_ad:
            status = "rejected"
        else:
            cleaned = clean_text(text)
            dup = await is_duplicate(s, city, cleaned)
            if dup:
                status = "rejected"
            else:
                if not contains_danger(cleaned):
                    cleaned = await gigachat.paraphrase(cleaned)
                processed = f"{cleaned}\n\n{signature(city)}"

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
        logger.info(f"Post {post.id} {status}")

        # Автопостинг, если включен auto_mode
        if status == "pending" and city.auto_mode:
            await publish(s, post)

# ---------------------------------------------------------------------------
# 8. Aiogram bots & admin commands
# ---------------------------------------------------------------------------
defaults = DefaultBotProperties(parse_mode=ParseMode.HTML)
news_bot  = Bot(token=NEWS_BOT_TOKEN, default=defaults)
news_dp   = Dispatcher()
admin_bot = Bot(token=ADMIN_BOT_TOKEN, default=defaults)
admin_dp  = Dispatcher()

async def publish(session: AsyncSession, post: Post) -> None:
    city = await session.get(City, post.city_id)
    if not city:
        return
    try:
        if post.media_path and Path(post.media_path).exists():
            with open(post.media_path, "rb") as f:
                msg = await news_bot.send_photo(city.channel_id, f, caption=post.processed_text)
        else:
            msg = await news_bot.send_message(city.channel_id, post.processed_text)
        post.status = "published"
        post.published_at = datetime.utcnow()
        await session.commit()
    except Exception as e:
        logger.error(f"Publish error: {e}")

ADMIN_COMMANDS = [
    BotCommand(command="addcity",   description="Добавить городской канал"),
    BotCommand(command="adddonor",  description="Добавить канал-донор"),
    BotCommand(command="pending",   description="Неодобренные посты"),
    BotCommand(command="publish",   description="Опубликовать пост"),
    BotCommand(command="setmask",   description="Задать маску для донора"),
    BotCommand(command="autopost",  description="Вкл/выкл авто-постинг"),
]

async def setup_bot_commands():
    await admin_bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeDefault())

@admin_dp.message(Command("addcity"))
async def cmd_addcity(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) != 2:
        return await msg.answer("/addcity <@username|https://t.me/...>")
    link = parts[1]
    try:
        cid, title, canon = await resolve_channel(link)
    except Exception as e:
        return await msg.answer(f"Error: {e}")
    async with SessionLocal() as s:
        s.add(City(channel_id=cid, title=title, link=canon))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ City added: {title} ({cid})")

@admin_dp.message(Command("adddonor"))
async def cmd_adddonor(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=3)
    if len(parts) < 3:
        return await msg.answer("/adddonor <city_id> <@username|https://t.me/...> [mask]")
    cid = int(parts[1]); link = parts[2]; mask = parts[3] if len(parts) == 4 else None
    try:
        dcid, title, _ = await resolve_channel(link)
    except Exception as e:
        return await msg.answer(f"Error: {e}")
    async with SessionLocal() as s:
        city = await s.get(City, cid)
        if not city:
            return await msg.answer("⛔️ Bad city_id")
        s.add(DonorChannel(channel_id=dcid, title=title, city_id=cid, mask_pattern=mask))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ Donor {title} added to {city.title}")

@admin_dp.message(Command("pending"))
async def cmd_pending(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    async with SessionLocal() as s:
        rows = (await s.execute(select(Post).where(Post.status == "pending").limit(10))).scalars().all()
        if not rows:
            return await msg.answer("No pending posts")
        for p in rows:
            preview = (p.processed_text or p.original_text)[:250]
            await msg.answer(f"ID {p.id}\n{preview}...")

@admin_dp.message(Command("publish"))
async def cmd_publish(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) != 2:
        return
    pid = int(parts[1])
    async with SessionLocal() as s:
        p = await s.get(Post, pid)
        if not p or p.status != "pending":
            return await msg.answer("⛔️ Bad post id")
        await publish(s, p)
        await msg.answer("✅ Published")

@admin_dp.message(Command("setmask"))
async def cmd_setmask(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=2)
    if len(parts) < 3:
        return await msg.answer("/setmask <donor_id> <regex>")
    did = int(parts[1]); mask = parts[2]
    async with SessionLocal() as s:
        donor = await s.get(DonorChannel, did)
        if not donor:
            return await msg.answer("⛔️ Bad donor id")
        donor.mask_pattern = mask
        await s.commit()
    await msg.answer(f"✅ Mask for donor {did} updated.")

@admin_dp.message(Command("autopost"))
async def cmd_autopost(msg: Message):
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=2)
    if len(parts) < 3:
        return await msg.answer("/autopost <city_id> <on|off>")
    cid = int(parts[1]); val = parts[2].strip().lower()
    async with SessionLocal() as s:
        city = await s.get(City, cid)
        if not city:
            return await msg.answer("⛔️ Bad city id")
        city.auto_mode = (val == "on")
        await s.commit()
    await msg.answer(f"Автопостинг {'включён' if val == 'on' else 'выключен'} для {city.title}")

# ---------------------------------------------------------------------------
# 9. RUN
# ---------------------------------------------------------------------------
def _start_bot(dp: Dispatcher, bot: Bot) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(dp.start_polling(bot, handle_signals=False))
    loop.run_forever()

async def main() -> None:
    await init_db()
    await DONORS.refresh()
    await setup_bot_commands()
    await telethon_client.start()
    Thread(target=_start_bot, args=(news_dp, news_bot), daemon=True).start()
    Thread(target=_start_bot, args=(admin_dp, admin_bot), daemon=True).start()
    while True:
        await asyncio.sleep(3600)

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
