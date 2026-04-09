import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_bot, get_db
from app.database.crud.trades import insert_trade
from app.api.schemas.trade import ForceTradeRequest
from app.services.bot_service import BotService
from app.utils.response import err, ok

logger = logging.getLogger("trader_ai.routes.trade")
router = APIRouter(prefix="/trade", tags=["Trade"])


@router.post("/force", summary="Paksa buka order (bypass filter sinyal)")
async def force_trade(
    req: ForceTradeRequest,
    bot: BotService = Depends(get_bot),
    db=Depends(get_db),
):
    if not bot.bot:
        raise HTTPException(400, "Bot belum diinisialisasi")

    mt5 = getattr(bot.bot, "mt5_conn", None)
    if not mt5 or not getattr(mt5, "connected", False):
        raise HTTPException(400, "MT5 tidak terhubung — jalankan dengan --mt5")

    result = bot.get_last_signal()
    if not result:
        raise HTTPException(400, "Belum ada data sinyal — tunggu siklus pertama")

    sig = result.get("signal", {})
    close = result.get("close", 0)
    atr = result.get("atr", close * 0.001) or close * 0.001

    # Hitung SL/TP jika belum ada
    if not sig.get("sl"):
        from app.engine.signals.signals import calculate_smart_tp_sl
        tp_sl = calculate_smart_tp_sl(req.direction, close, atr, bot.bot.df_ind, 5.0)
        sl, tp = tp_sl["sl"], tp_sl["tp"]
    else:
        sl, tp = sig["sl"], sig["tp"]

    order_result = mt5.place_order(
        symbol_key=req.symbol,
        direction=req.direction,
        sl=sl,
        tp=tp,
    )

    if order_result.get("success"):
        # Catat ke DB
        await insert_trade(db, {
            "symbol": req.symbol,
            "direction": req.direction,
            "entry": close,
            "sl": sl,
            "tp": tp,
            "lot": req.lot,
            "ticket": order_result.get("ticket"),
            "result": "OPEN",
            "source": "force",
            "opened_at": datetime.now(tz=timezone.utc),
        })

    return order_result


@router.get("/positions", summary="Posisi aktif di MT5")
async def get_positions(bot: BotService = Depends(get_bot)):
    mt5 = getattr(bot.bot, "mt5_conn", None) if bot.bot else None
    if not mt5 or not getattr(mt5, "connected", False):
        return {"positions": [], "message": "MT5 tidak terhubung"}

    positions = mt5.get_positions() if hasattr(mt5, "get_positions") else []
    return {"positions": positions, "total": len(positions)}
