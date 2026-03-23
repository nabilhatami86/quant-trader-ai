from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from db.models import Trade


async def insert_trade(db: AsyncSession, data: dict) -> Trade:
    trade = Trade(**data)
    db.add(trade)
    await db.commit()
    await db.refresh(trade)
    return trade


async def get_trades(
    db: AsyncSession, symbol: str | None = None, limit: int = 100
) -> list[Trade]:
    q = select(Trade).order_by(Trade.created_at.desc()).limit(limit)
    if symbol:
        q = q.where(Trade.symbol == symbol)
    result = await db.execute(q)
    return result.scalars().all()


async def get_trade_stats(db: AsyncSession, symbol: str | None = None) -> dict:
    q = select(
        func.count(Trade.id).label("total"),
        func.coalesce(func.sum(Trade.pnl_usd), 0).label("total_pnl"),
        func.coalesce(func.avg(Trade.pnl_usd), 0).label("avg_pnl"),
        func.count(Trade.id).filter(Trade.result == "WIN").label("wins"),
        func.count(Trade.id).filter(Trade.result == "LOSS").label("losses"),
    )
    if symbol:
        q = q.where(Trade.symbol == symbol)
    row = (await db.execute(q)).one()

    win_rate = round(row.wins / row.total * 100, 1) if row.total > 0 else 0.0
    return {
        "total": row.total,
        "wins": row.wins,
        "losses": row.losses,
        "win_rate": win_rate,
        "total_pnl": round(float(row.total_pnl), 2),
        "avg_pnl": round(float(row.avg_pnl), 2),
    }
