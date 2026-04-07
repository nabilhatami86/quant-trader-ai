from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)


def make_thread_engine():
    """
    Engine tanpa connection pool — aman dipakai dari asyncio.run() di background
    thread (bot_service, db_logger). NullPool tidak menyimpan koneksi antar loop
    sehingga tidak terjadi 'unknown protocol state' asyncpg.
    """
    return create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        poolclass=NullPool,
    )

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables() -> None:
    """Buat semua tabel jika belum ada (auto-migrate sederhana)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
