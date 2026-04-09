from fastapi import APIRouter, Depends

from app.api.deps import get_db
from app.database.crud.trades import get_trade_stats, get_trades
from app.api.schemas.trade import TradeOut, TradeStatsOut

router = APIRouter(prefix="/journal", tags=["Journal"])


@router.get("", summary="Riwayat semua trade dari database")
async def get_journal(
    symbol: str = None,
    limit: int = 100,
    db=Depends(get_db),
):
    trades = await get_trades(db, symbol, limit)
    return {
        "total": len(trades),
        "trades": [TradeOut.model_validate(t).model_dump() for t in trades],
    }


@router.get("/stats", summary="Statistik trading (win rate, total PnL)")
async def get_stats(symbol: str = None, db=Depends(get_db)):
    stats = await get_trade_stats(db, symbol)
    return TradeStatsOut(**stats).model_dump()
