"""
/adaptive — Adaptive learning stats + circuit breaker status
"""
import logging
from fastapi import APIRouter, Depends
from app.api.deps import get_bot
from app.services.bot_service import BotService
from app.utils.response import ok

logger = logging.getLogger("trader_ai.routes.adaptive")
router = APIRouter(prefix="/adaptive", tags=["Adaptive"])


@router.get("", summary="Statistik adaptive learning")
async def get_adaptive(bot: BotService = Depends(get_bot)):
    return ok(bot.get_adaptive_stats())


@router.get("/circuit", summary="Status circuit breaker & consecutive loss")
async def get_circuit(bot: BotService = Depends(get_bot)):
    return ok(bot.get_circuit_status())


@router.get("/recent", summary="10 trade terakhir dari adaptive state")
async def get_recent_trades(bot: BotService = Depends(get_bot)):
    stats = bot.get_adaptive_stats()
    return ok({
        "recent_trades"  : stats.get("recent_trades", []),
        "performance_mode": stats.get("performance_mode"),
        "win_rate_20"    : stats.get("win_rate_20"),
    })
