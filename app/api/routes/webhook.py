import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_bot, get_db
from app.database.crud.signals import build_signal_payload, insert_signal
from app.services.bot_service import BotService

logger = logging.getLogger("trader_ai.routes.webhook")
router = APIRouter(prefix="/webhook", tags=["Webhook"])


@router.post("/tradingview", summary="Terima alert dari TradingView")
async def tradingview_webhook(
    request: Request,
    bot: BotService = Depends(get_bot),
    db=Depends(get_db),
):
    """
    Format alert TradingView (JSON):
    ```json
    {"direction": "BUY", "symbol": "XAUUSD", "price": 2300.0}
    ```
    Bot akan trigger satu siklus analisis dan menyimpan hasilnya.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Payload bukan JSON yang valid")

    direction = str(payload.get("direction", "")).upper()
    symbol = str(payload.get("symbol", "XAUUSD")).upper()
    price = payload.get("price")

    logger.info(f"TradingView webhook: {direction} {symbol} @ {price}")

    if direction not in ("BUY", "SELL", "CLOSE", ""):
        return {"received": True, "action": "ignored", "reason": "direction tidak dikenal"}

    # Jalankan satu siklus analisis
    if bot.bot:
        result = bot.run_once()
        if result and "error" not in result:
            try:
                await insert_signal(db, build_signal_payload(result))
            except Exception as exc:
                logger.warning(f"Gagal simpan sinyal dari webhook: {exc}")

            return {
                "received": True,
                "tv_direction": direction,
                "bot_direction": result.get("exec_direction", "WAIT"),
                "score": result.get("signal", {}).get("score", 0),
                "action": "analyzed",
            }

    return {"received": True, "action": "bot_not_running"}


@router.post("/mt5", summary="Terima notifikasi dari MT5 EA")
async def mt5_webhook(request: Request, db=Depends(get_db)):
    """Terima data order dari EA MetaTrader (file bridge opsional)."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Payload bukan JSON yang valid")

    logger.info(f"MT5 webhook: {payload}")
    return {"received": True, "payload": payload}


@router.post("/admin/migrate", summary="Migrate CSV lama ke PostgreSQL")
async def migrate_csv(db=Depends(get_db)):
    """Jalankan migrasi data/history/*.csv dan logs/journal_*.csv ke Postgres."""
    from app.services.migrate_service import run_migration
    result = await run_migration(db)
    return result
