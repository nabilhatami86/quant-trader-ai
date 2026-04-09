"""
/positions — Posisi terbuka + akun MT5
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_bot
from app.services.bot_service import BotService
from app.utils.response import ok, err

logger = logging.getLogger("trader_ai.routes.positions")
router = APIRouter(prefix="/positions", tags=["Positions"])


@router.get("", summary="Semua posisi terbuka")
async def get_positions(bot: BotService = Depends(get_bot)):
    positions = bot.get_positions()
    account   = bot.get_account()
    return ok({
        "account"  : account,
        "positions": positions,
        "count"    : len(positions),
    })


@router.delete("/{ticket}", summary="Tutup posisi by ticket")
async def close_position(ticket: int, bot: BotService = Depends(get_bot)):
    mt5 = bot._get_mt5()
    if not mt5 or not getattr(mt5, "connected", False):
        raise HTTPException(400, "MT5 tidak terhubung")

    result = mt5.close_position(ticket)
    if result.get("success"):
        return ok({"ticket": ticket, "closed": True, "pnl": result.get("profit", 0)})
    return err(result.get("error", "Gagal tutup posisi"))


@router.delete("", summary="Tutup semua posisi")
async def close_all_positions(bot: BotService = Depends(get_bot)):
    mt5 = bot._get_mt5()
    if not mt5 or not getattr(mt5, "connected", False):
        raise HTTPException(400, "MT5 tidak terhubung")

    positions = bot.get_positions()
    results = []
    for p in positions:
        r = mt5.close_position(p["ticket"])
        results.append({"ticket": p["ticket"], "success": r.get("success"), "pnl": r.get("profit", 0)})

    closed = sum(1 for r in results if r["success"])
    return ok({"closed": closed, "total": len(results), "results": results})
