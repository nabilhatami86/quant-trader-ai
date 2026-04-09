"""
/stats — Statistik trading harian/mingguan
"""
import logging
from fastapi import APIRouter, Depends, Query
from app.api.deps import get_bot
from app.services.bot_service import BotService
from app.utils.response import ok

logger = logging.getLogger("trader_ai.routes.stats")
router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get("/daily", summary="Statistik trading hari ini")
async def get_daily(bot: BotService = Depends(get_bot)):
    return ok(bot.get_daily_stats())


@router.get("/history", summary="Statistik per hari dari journal CSV")
async def get_history(
    days: int = Query(7, ge=1, le=90),
    bot: BotService = Depends(get_bot),
):
    try:
        import pandas as pd
        from datetime import date, timedelta
        from app.services.journal import JOURNAL_PATH

        from pathlib import Path
        jpath = Path(JOURNAL_PATH)
        if not jpath.exists():
            return ok({"days": [], "message": "Journal tidak ditemukan"})

        df = pd.read_csv(jpath, dtype=str)
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)

        result_days = []
        today = date.today()
        for i in range(days):
            d = (today - timedelta(days=i)).isoformat()
            day_df = df[df.get('entry_time', df.iloc[:, 0]).str.startswith(d, na=False)]
            if day_df.empty:
                result_days.append({"date": d, "total": 0, "win": 0, "loss": 0,
                                    "win_rate": 0, "net_pnl": 0})
                continue
            wins  = int((day_df['result'].str.upper() == 'WIN').sum()) if 'result' in day_df.columns else 0
            losses = int((day_df['result'].str.upper() == 'LOSS').sum()) if 'result' in day_df.columns else 0
            total = wins + losses
            result_days.append({
                "date"    : d,
                "total"   : total,
                "win"     : wins,
                "loss"    : losses,
                "win_rate": round(wins / total * 100, 1) if total else 0,
                "net_pnl" : round(float(day_df['pnl'].sum()), 2),
            })

        return ok({"days": result_days})
    except Exception as e:
        return ok({"error": str(e)})
