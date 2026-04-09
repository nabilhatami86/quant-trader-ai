from fastapi import APIRouter, Depends

from app.api.deps import get_db
from app.database.crud.tx_log import get_tx_logs, get_tx_summary

router = APIRouter(prefix="/txlog", tags=["Transaction Log"])


@router.get("", summary="Semua transaction log (signal, order, error, dsb)")
async def get_logs(
    symbol: str = None,
    event_type: str = None,
    limit: int = 100,
    db=Depends(get_db),
):
    logs = await get_tx_logs(db, symbol=symbol, event_type=event_type, limit=limit)
    return {
        "total": len(logs),
        "logs": [
            {
                "id":         l.id,
                "created_at": str(l.created_at),
                "event_type": l.event_type,
                "symbol":     l.symbol,
                "timeframe":  l.timeframe,
                "direction":  l.direction,
                "price":      l.price,
                "sl":         l.sl,
                "tp":         l.tp,
                "lot":        l.lot,
                "ticket":     l.ticket,
                "pnl_usd":    l.pnl_usd,
                "pips":       l.pips,
                "message":    l.message,
                "meta":       l.meta,
            }
            for l in logs
        ],
    }


@router.get("/summary", summary="Ringkasan event per tipe")
async def get_summary(symbol: str = None, db=Depends(get_db)):
    return await get_tx_summary(db, symbol=symbol)


@router.get("/events", summary="Daftar tipe event yang tersedia")
async def list_event_types():
    return {
        "event_types": [
            "SIGNAL_BUY",
            "SIGNAL_SELL",
            "SIGNAL_WAIT",
            "ORDER_OPEN",
            "ORDER_CLOSE",
            "ORDER_SKIP",
            "BOT_START",
            "BOT_STOP",
            "BOT_ERROR",
            "CYCLE_DONE",
            "MIGRATE",
        ]
    }
