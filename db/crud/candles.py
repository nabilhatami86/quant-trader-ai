from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
import pandas as pd

from db.models import Candle


async def bulk_upsert_candles(
    db: AsyncSession, symbol: str, timeframe: str, df: pd.DataFrame
) -> int:
    """Insert candles dari DataFrame. Skip duplikat (upsert by unique key)."""
    if df.empty:
        return 0

    import pandas as _pd

    # Normalisasi index ke UTC sebelum iterasi
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    rows = []
    for ts, row in df.iterrows():
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "open": float(row.get("Open", row.get("open", 0))),
                "high": float(row.get("High", row.get("high", 0))),
                "low": float(row.get("Low", row.get("low", 0))),
                "close": float(row.get("Close", row.get("close", 0))),
                "volume": float(row.get("Volume", row.get("volume", 0))),
            }
        )

    # Insert dalam batch 1000 baris — hindari limit 32767 parameter Postgres
    BATCH_SIZE = 1000
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        stmt = pg_insert(Candle).values(batch)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_candle")
        await db.execute(stmt)

    await db.commit()
    return len(rows)


async def get_candles(
    db: AsyncSession, symbol: str, timeframe: str, limit: int = 500
) -> list[Candle]:
    result = await db.execute(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def get_candle_count(db: AsyncSession, symbol: str, timeframe: str) -> int:
    from sqlalchemy import func, select
    result = await db.execute(
        select(func.count()).where(
            Candle.symbol == symbol, Candle.timeframe == timeframe
        )
    )
    return result.scalar_one()
