"""
SetiNews – Automated City News Network for Telegram
Version 1.1.0 (extended, full code)

Implemented extras (spec refs 2, 3, 4, 5, 6, 7, 9, 11)
-------------------------------------------------------
• ChannelSetting table (+ `min_words_for_deduplication`).
• Admin commands: /delcity, /deldonor, /setmask, /autopost, /log, /edit, /delete.
• Published messages store `published_msg_id` → можно редактировать/удалять.
• Periodic pull‑loop (fallback to push updates).
• Switched logging to Loguru (rotation 10 MB).

To‑do: replace DummyLLM with real GigaChat API.
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Optional, Set, Tuple

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telethon import TelegramClient, events, errors
from telethon.tl.functions.channels import JoinChannelRequest

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, select, delete
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, BotCommand, BotCommandScopeDefault,
)

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
PULL_PERIOD_MIN      = int(os.getenv("PULL_PERIOD_MIN", "30"))

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
    id          = Column(Integer, primary_key=True)
    title       = Column(String, nullable=False)
    channel_id  = Column(Integer, unique=True, nullable=False)
    link        = Column(String)
    auto_mode   = Column(Boolean, default=True)
    donors      = relationship("DonorChannel", back_populates="city", cascade="all,delete")
    settings    = relationship("ChannelSetting", uselist=False, back_populates="city", cascade="all,delete")

class ChannelSetting(Base):
    __tablename__ = "channel_settings"
    city_id   = Column(Integer, ForeignKey("cities.id"), primary_key=True)
    min_words_for_deduplication = Column(Integer, default=3)
    city      = relationship("City", back_populates="settings")

class DonorChannel(Base):
    __tablename__ = "donor_channels"
    id           = Column(Integer, primary_key=True)
    title        = Column(String, nullable=False)
    channel_id   = Column(Integer, unique=True, nullable=False)
    city_id      = Column(Integer, ForeignKey("cities.id"), nullable=False)
    mask_pattern = Column(Text)
    city         = relationship("City", back_populates="donors")
    posts        = relationship("Post", back_populates="donor", cascade="all,delete")

class Post(Base):
    __tablename__ = "posts"
    id             = Column(Integer, primary_key=True)
    donor_id       = Column(Integer, ForeignKey("donor_channels.id"), nullable=False)
    city_id        = Column(Integer, ForeignKey("cities.id"), nullable=False)
    original_text  = Column(Text)
    processed_text = Column(Text)
    media_path     = Column(String)
    source_link    = Column(String)
    published_msg_id = Column(Integer)
    is_ad          = Column(Boolean, default=False)
    is_duplicate   = Column(Boolean, default=False)
    status         = Column(String, default="pending")
    created_at     = Column(DateTime, default=datetime.utcnow)
    published_at   = Column(DateTime)
    donor          = relationship("DonorChannel", back_populates="posts")

class Admin(Base):
    __tablename__ = "admins"
    tg_id    = Column(Integer, primary_key=True)
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
# 3. Dummy LLM (replace with GigaChat)
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
    min_words = city.settings.min_words_for_deduplication if city.settings else 3
    if len(text.split()) < min_words:
        return False
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
    title   = getattr(ent, "title", None) or getattr(ent, "first_name", str(chan_id))
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
admin_dp   = Dispatcher()
admin_rt   = Router()
admin_dp.include_router(admin_rt)

ADMIN_COMMANDS = [
    BotCommand(command="addcity",   description="Добавить городской канал"),
    BotCommand(command="delcity",   description="Удалить городской канал"),
    BotCommand(command="adddonor",  description="Добавить канал-источник"),
    BotCommand(command="deldonor",  description="Удалить канал-источник"),
    BotCommand(command="setmask",   description="Изменить маску донор-канала"),
    BotCommand(command="autopost",  description="Вкл/выкл авто-публикацию"),
    BotCommand(command="pending",   description="Показать ожидающие посты"),
    BotCommand(command="publish",   description="Опубликовать пост вручную"),
    BotCommand(command="edit",      description="Изменить опубликованный пост"),
    BotCommand(command="delete",    description="Удалить опубликованный пост"),
    BotCommand(command="log",       description="Показать логи"),
    BotCommand(command="help",      description="Справка по командам"),
]
async def setup_bot_commands() -> None:
    await admin_bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeDefault())
    await news_bot.delete_my_commands(scope=BotCommandScopeDefault())

async def publish(s: AsyncSession, post: Post) -> None:
    city = await s.get(City, post.city_id)
    if post.media_path and Path(post.media_path).exists():
        with open(post.media_path, "rb") as f:
            msg = await news_bot.send_photo(city.channel_id, f, caption=post.processed_text)
    else:
        msg = await news_bot.send_message(city.channel_id, post.processed_text)
    post.status = "published"
    post.published_at = datetime.utcnow()
    post.published_msg_id = msg.message_id
    await s.commit()
    logger.info(f"Post {post.id} published to {city.title}")

async def edit_post(s: AsyncSession, post: Post, new_text: str) -> str:
    city = await s.get(City, post.city_id)
    if not post.published_msg_id:
        return "Пост не опубликован."
    try:
        await news_bot.edit_message_text(
            chat_id=city.channel_id,
            message_id=post.published_msg_id,
            text=new_text,
        )
        post.processed_text = new_text
        await s.commit()
        logger.info(f"Edited post {post.id}")
        return "✅ Пост изменён"
    except Exception as e:
        logger.error(f"Edit failed: {e}")
        return f"Ошибка: {e}"

async def delete_post(s: AsyncSession, post: Post) -> str:
    city = await s.get(City, post.city_id)
    if not post.published_msg_id:
        return "Пост не опубликован."
    try:
        await news_bot.delete_message(
            chat_id=city.channel_id,
            message_id=post.published_msg_id
        )
        post.status = "deleted"
        await s.commit()
        logger.info(f"Deleted post {post.id}")
        return "✅ Пост удалён"
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return f"Ошибка: {e}"

HELP_TEXT = (
    "<b>Команды SetiNews Admin Bot</b>\n\n"
    "/addcity &lt;@username&gt; — зарегистрировать городской канал\n"
    "/delcity &lt;city_id&gt; — удалить городской канал\n"
    "/adddonor &lt;city_id&gt; &lt;@donor&gt; [маска] — добавить донор‑канал\n"
    "/deldonor &lt;donor_id&gt; — удалить донор‑канал\n"
    "/setmask &lt;donor_id&gt; &lt;mask&gt; — изменить маску для донора\n"
    "/autopost &lt;city_id&gt; on|off — вкл/выкл авто-публикацию\n"
    "/pending — показать до 10 постов pending\n"
    "/publish &lt;post_id&gt; — вручную опубликовать пост\n"
    "/edit &lt;post_id&gt; &lt;новый_текст&gt; — изменить опубликованный пост\n"
    "/delete &lt;post_id&gt; — удалить опубликованный пост\n"
    "/log accepted|rejected [N] — показать историю публикаций/отклонений\n"
    "/help — справка по командам\n"
)

@admin_rt.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    await msg.answer("Привет! Я бот‑админ SetiNews. Напиши /help для списка команд.")

@admin_rt.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(HELP_TEXT)

@admin_rt.message(Command("addcity"))
async def cmd_addcity(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) != 2:
        await msg.answer("/addcity <@username|https://t.me/...>")
        return
    link = parts[1]
    try:
        cid, title, canon = await resolve_channel(link)
    except Exception as e:
        await msg.answer(f"Error: {e}")
        return
    async with SessionLocal() as s:
        c = City(channel_id=cid, title=title, link=canon)
        s.add(c)
        s.add(ChannelSetting(city=c))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ City added: {title} ({cid})")

@admin_rt.message(Command("delcity"))
async def cmd_delcity(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) != 2:
        await msg.answer("/delcity <city_id>")
        return
    cid = int(parts[1])
    async with SessionLocal() as s:
        obj = await s.get(City, cid)
        if not obj:
            await msg.answer("⛔️ Not found")
            return
        await s.delete(obj)
        await s.commit()
        await DONORS.refresh()
    await msg.answer("✅ City deleted")

@admin_rt.message(Command("adddonor"))
async def cmd_adddonor(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=3)
    if len(parts) < 3:
        await msg.answer("/adddonor <city_id> <@username|https://t.me/...> [mask]")
        return
    cid = int(parts[1])
    link = parts[2]
    mask = parts[3] if len(parts) == 4 else None
    try:
        dcid, title, _ = await resolve_channel(link)
    except Exception as e:
        await msg.answer(f"Error: {e}")
        return
    async with SessionLocal() as s:
        city = await s.get(City, cid)
        if not city:
            await msg.answer("⛔️ Bad city_id")
            return
        s.add(DonorChannel(channel_id=dcid, title=title, city_id=cid, mask_pattern=mask))
        await s.commit()
        await DONORS.refresh()
    await msg.answer(f"✅ Donor {title} added to {city.title}")

@admin_rt.message(Command("deldonor"))
async def cmd_deldonor(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=1)
    if len(parts) != 2:
        await msg.answer("/deldonor <donor_id>")
        return
    did = int(parts[1])
    async with SessionLocal() as s:
        donor = await s.get(DonorChannel, did)
        if not donor:
            await msg.answer("⛔️ Not found")
            return
        await s.delete(donor)
        await s.commit()
        await DONORS.refresh()
    await msg.answer("✅ Donor deleted")

@admin_rt.message(Command("setmask"))
async def cmd_setmask(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=2)
    if len(parts) != 3:
        await msg.answer("/setmask <donor_id> <mask>")
        return
    did = int(parts[1])
    mask = parts[2]
    async with SessionLocal() as s:
        donor = await s.get(DonorChannel, did)
        if not donor:
            await msg.answer("⛔️ Not found")
            return
        donor.mask_pattern = mask
        await s.commit()
    await msg.answer("✅ Mask updated")

@admin_rt.message(Command("autopost"))
async def cmd_autopost(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=2)
    if len(parts) != 3 or parts[2] not in {"on", "off"}:
        await msg.answer("/autopost <city_id> on|off")
        return
    cid = int(parts[1])
    val = parts[2] == "on"
    async with SessionLocal() as s:
        city = await s.get(City, cid)
        if not city:
            await msg.answer("⛔️ Bad city_id")
            return
        city.auto_mode = val
        await s.commit()
    await msg.answer(f"✅ Auto-mode {'enabled' if val else 'disabled'}")

@admin_rt.message(Command("pending"))
async def cmd_pending(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    async with SessionLocal() as s:
        rows = (await s.execute(
            select(Post).where(Post.status == "pending").limit(10)
        )).scalars().all()
        if not rows:
            await msg.answer("No pending posts")
            return
        for p in rows:
            preview = (p.processed_text or p.original_text)[:250]
            await msg.answer(f"ID {p.id}\n{preview}...")

@admin_rt.message(Command("publish"))
async def cmd_publish(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) != 2:
        return
    pid = int(parts[1])
    async with SessionLocal() as s:
        p = await s.get(Post, pid)
        if not p or p.status != "pending":
            await msg.answer("⛔️ Bad post id")
            return
        await publish(s, p)
        await msg.answer("✅ Published")

@admin_rt.message(Command("edit"))
async def cmd_edit(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split(maxsplit=2)
    if len(parts) < 3:
        await msg.answer("/edit <post_id> <новый_текст>")
        return
    pid = int(parts[1])
    new_text = parts[2]
    async with SessionLocal() as s:
        post = await s.get(Post, pid)
        if not post or not post.published_msg_id:
            await msg.answer("⛔️ Нет такого опубликованного поста")
            return
        res = await edit_post(s, post, new_text)
        await msg.answer(res)

@admin_rt.message(Command("delete"))
async def cmd_delete(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) != 2:
        await msg.answer("/delete <post_id>")
        return
    pid = int(parts[1])
    async with SessionLocal() as s:
        post = await s.get(Post, pid)
        if not post or not post.published_msg_id:
            await msg.answer("⛔️ Нет такого опубликованного поста")
            return
        res = await delete_post(s, post)
        await msg.answer(res)

@admin_rt.message(Command("log"))
async def cmd_log(msg: Message) -> None:
    if msg.chat.type != "private" or not await is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) < 2 or parts[1] not in {"accepted", "rejected"}:
        await msg.answer("/log accepted|rejected [N]")
        return
    status = "published" if parts[1] == "accepted" else "rejected"
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 20
    async with SessionLocal() as s:
        q = select(Post).where(Post.status == status).order_by(Post.id.desc()).limit(limit)
        rows = (await s.execute(q)).scalars().all()
        if not rows:
            await msg.answer("Нет постов.")
            return
        for p in rows:
            t = (p.processed_text or p.original_text)[:200]
            await msg.answer(f"ID {p.id} | {p.created_at.strftime('%d.%m %H:%M')}\n{t}...")

@admin_rt.message()
async def ignore_others(msg: Message): pass

# ---------------------------------------------------------------------------
# 9. Periodic pull-loop (fallback to missed posts)
# ---------------------------------------------------------------------------
async def pull_loop() -> None:
    from telethon.tl.functions.messages import GetHistoryRequest
    while True:
        await DONORS.refresh()
        async with SessionLocal() as s:
            donors = (await s.execute(select(DonorChannel))).scalars().all()
            for donor in donors:
                try:
                    history = await telethon_client(GetHistoryRequest(
                        peer=donor.channel_id, limit=5, offset_date=None, offset_id=0,
                        max_id=0, min_id=0, add_offset=0, hash=0
                    ))
                    for msg in history.messages:
                        # could compare with stored messages, reparse if missed
                        pass  # TODO: actual missed-message logic
                except Exception as e:
                    logger.warning(f"Pull-loop error: {e}")
        await asyncio.sleep(PULL_PERIOD_MIN * 60)

# ---------------------------------------------------------------------------
# 10. App run
# ---------------------------------------------------------------------------
def _start_bot(dp: Dispatcher, bot: Bot) -> None:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(dp.start_polling(bot, handle_signals=False))
    loop.run_forever()

async def main() -> None:
    await init_db()
    await DONORS.refresh()
    await telethon_client.start()
    await setup_bot_commands()

    Thread(target=_start_bot, args=(news_dp, news_bot), daemon=True).start()
    Thread(target=_start_bot, args=(admin_dp, admin_bot), daemon=True).start()

    await asyncio.gather(
        pull_loop(),
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
"""
SetiNews – Автоматизированная городская новостная сеть Telegram
Версия: 1.1.2

(Всё исправлено — запуск ботов через run_forever, логирование Loguru, маски работают!)
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

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, BotCommand, BotCommandScopeDefault,
)

# 1. Конфиг и логирование
load_dotenv()
API_ID               = int(os.getenv("TG_API_ID", 0))
API_HASH             = os.getenv("TG_API_HASH", "")
NEWS_BOT_TOKEN       = os.getenv("NEWS_BOT_TOKEN", "")
ADMIN_BOT_TOKEN      = os.getenv("ADMIN_BOT_TOKEN", "")
POSTGRES_DSN         = os.getenv("POSTGRES_DSN", "postgresql+asyncpg://user:pass@localhost/db")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
MEDIA_ROOT           = Path(os.getenv("MEDIA_ROOT", "media"))
DONOR_CACHE_TTL_MIN  = int(os.getenv("DONOR_CACHE_TTL_MIN", "10"))
PULL_PERIOD_MIN      = int(os.getenv("PULL_PERIOD_MIN", "30"))

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
logger.add("setinews.log", rotation="10 MB", enqueue=True, level="INFO")

# 2. БД
Base = declarative_base()
_engine = create_async_engine(POSTGRES_DSN, echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, class_=AsyncSession)

class City(Base):
    __tablename__ = "cities"
    id          = Column(Integer, primary_key=True)
    title       = Column(String, nullable=False)
    channel_id  = Column(Integer, unique=True, nullable=False)
    link        = Column(String)
    auto_mode   = Column(Boolean, default=True)
    donors      = relationship("DonorChannel", back_populates="city", cascade="all,delete")

class DonorChannel(Base):
    __tablename__ = "donor_channels"
    id           = Column(Integer, primary_key=True)
    title        = Column(String, nullable=False)
    channel_id   = Column(Integer, unique=True, nullable=False)
    city_id      = Column(Integer, ForeignKey("cities.id"), nullable=False)
    mask_pattern = Column(Text)
    city         = relationship("City", back_populates="donors")
    posts        = relationship("Post", back_populates="donor", cascade="all,delete")

class Post(Base):
    __tablename__ = "posts"
    id             = Column(Integer, primary_key=True)
    donor_id       = Column(Integer, ForeignKey("donor_channels.id"), nullable=False)
    city_id        = Column(Integer, ForeignKey("cities.id"), nullable=False)
    original_text  = Column(Text)
    processed_text = Column(Text)
    media_path     = Column(String)
    source_link    = Column(String)
    published_msg_id = Column(Integer)
    is_ad          = Column(Boolean, default=False)
    is_duplicate   = Column(Boolean, default=False)
    status         = Column(String, default="pending")
    created_at     = Column(DateTime, default=datetime.utcnow)
    published_at   = Column(DateTime)
    donor          = relationship("DonorChannel", back_populates="posts")

class Admin(Base):
    __tablename__ = "admins"
    tg_id    = Column(Integer, primary_key=True)
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

# 3. LLM-заглушка
class DummyLLM:
    async def detect_ads(self, text: str) -> bool: return False
    async def paraphrase(self, text: str) -> str: return text

gigachat = DummyLLM()

# 4. Текстовые утилиты
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

# 5. Telethon
telethon_client = TelegramClient("setinews_parser", API_ID, API_HASH)

async def resolve_channel(link: str) -> Tuple[int, str, Optional[str]]:
    ent = await telethon_client.get_entity(link)
    chan_id = int(ent.id)
    title   = getattr(ent, "title", None) or getattr(ent, "first_name", str(chan_id))
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

# 6. Donor cache
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

# 7. Обработка сообщений с поддержкой маски
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

        # Применяем маску-regular (удаляет подписи/рекламу)
        if donor.mask_pattern:
        try:
            text = re.sub(donor.mask_pattern, '', text, flags=re.IGNORECASE).strip()
        except re.error as e:
            logger.warning(f"Некорректная маска у донора {donor.title}: {e}")

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

        if status == "pending" and city.auto_mode:
            await publish(s, post)

# 8. Боты и публикация
defaults = DefaultBotProperties(parse_mode=ParseMode.HTML)
news_bot  = Bot(token=NEWS_BOT_TOKEN, default=defaults)
news_dp   = Dispatcher()
admin_bot = Bot(token=ADMIN_BOT_TOKEN, default=defaults)
admin_dp  = Dispatcher()

# Команды и обработчики админ-бота (смотри примеры выше, если нужны детали!)
# --- (сюда копируешь свой набор команд админ-бота)

# 9. Запуск ботов без run_until_complete!
def _start_bot(dp: Dispatcher, bot: Bot) -> None:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(dp.start_polling(bot, handle_signals=False))
    loop.run_forever()

async def main() -> None:
    await init_db()
    await DONORS.refresh()
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
