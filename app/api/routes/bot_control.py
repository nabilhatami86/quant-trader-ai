"""
/bot — Kontrol bot (pause/resume/status) + ScalpML signal
"""
import logging
from fastapi import APIRouter, Depends
from app.api.deps import get_bot
from app.services.bot_service import BotService
from app.utils.response import ok, err

logger = logging.getLogger("trader_ai.routes.bot_control")
router = APIRouter(prefix="/bot", tags=["Bot"])


@router.get("/status", summary="Status lengkap bot + akun + circuit breaker")
async def get_bot_status(bot: BotService = Depends(get_bot)):
    status   = bot.get_status()
    account  = bot.get_account()
    circuit  = bot.get_circuit_status()
    daily    = bot.get_daily_stats()
    return ok({
        "bot"     : {**status, "paused": getattr(bot, '_paused', False)},
        "account" : account,
        "circuit" : circuit,
        "daily"   : daily,
    })


@router.post("/pause", summary="Pause bot (tidak ada trade baru)")
async def pause_bot(bot: BotService = Depends(get_bot)):
    bot.pause()
    return ok({"paused": True, "message": "Bot di-pause — tidak ada trade baru sampai di-resume"})


@router.post("/resume", summary="Resume bot")
async def resume_bot(bot: BotService = Depends(get_bot)):
    bot.resume()
    return ok({"paused": False, "message": "Bot aktif kembali"})


@router.get("/scalp", summary="ScalpML signal terbaru + analisis candle")
async def get_scalp_signal(bot: BotService = Depends(get_bot)):
    return ok(bot.get_scalp_signal())


@router.post("/analyze", summary="Trigger analisis manual sekarang")
async def trigger_analyze(bot: BotService = Depends(get_bot)):
    result = bot.run_once()
    if result.get("error"):
        return err(result["error"])
    return ok({
        "direction"   : result.get("exec_direction", "WAIT"),
        "signal"      : result.get("signal", {}),
        "scalp_pred"  : result.get("scalp_pred", {}),
        "timestamp"   : result.get("timestamp"),
    })
