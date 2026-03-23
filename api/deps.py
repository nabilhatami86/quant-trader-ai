from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from services.bot_service import BotService, bot_service


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_bot() -> BotService:
    return bot_service
