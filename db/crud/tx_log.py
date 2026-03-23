from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import TxLog


async def log_event(
    db: AsyncSession,
    event_type: str,
    symbol: str | None = None,
    timeframe: str | None = None,
    direction: str | None = None,
    price: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    lot: float | None = None,
    ticket: int | None = None,
    pnl_usd: float | None = None,
    pips: float | None = None,
    message: str | None = None,
    meta: dict | None = None,
) -> TxLog:
    """Catat satu event ke tx_log. Selalu append, tidak pernah update."""
    entry = TxLog(
        event_type=event_type,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        price=price,
        sl=sl,
        tp=tp,
        lot=lot,
        ticket=ticket,
        pnl_usd=pnl_usd,
        pips=pips,
        message=message,
        meta=meta,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def get_tx_logs(
    db: AsyncSession,
    symbol: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
) -> list[TxLog]:
    q = select(TxLog).order_by(TxLog.created_at.desc()).limit(limit)
    if symbol:
        q = q.where(TxLog.symbol == symbol)
    if event_type:
        q = q.where(TxLog.event_type == event_type)
    result = await db.execute(q)
    return result.scalars().all()


async def get_tx_summary(db: AsyncSession, symbol: str | None = None) -> dict:
    from sqlalchemy import func
    q = select(
        TxLog.event_type,
        func.count(TxLog.id).label("count"),
    ).group_by(TxLog.event_type).order_by(func.count(TxLog.id).desc())
    if symbol:
        q = q.where(TxLog.symbol == symbol)
    result = await db.execute(q)
    return {row.event_type: row.count for row in result.all()}
