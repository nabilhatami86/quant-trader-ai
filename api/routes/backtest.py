import logging

from fastapi import APIRouter, Depends

from api.deps import get_db
from utils.response import err

logger = logging.getLogger("trader_ai.routes.backtest")
router = APIRouter(prefix="/backtest", tags=["Backtest"])


@router.post("/run", summary="Jalankan backtest (blocking ~5-30 detik)")
async def run_backtest(
    symbol: str = "XAUUSD",
    timeframe: str = "1h",
    period: str = "1y",
):
    try:
        from backend.bot import fetch_data
        from backtest.engine import run_backtest as _run

        raw = fetch_data(symbol, timeframe, period)
        if raw.empty:
            return err("Tidak ada data untuk backtest")

        result = _run(raw, symbol=symbol)

        trades_df = result.get("trades")
        last_trades = []
        if hasattr(trades_df, "tail"):
            for _, t in trades_df.tail(10).iterrows():
                last_trades.append({
                    "direction": t.get("direction"),
                    "entry": t.get("entry"),
                    "exit": t.get("exit"),
                    "pnl_usd": round(float(t.get("pnl_usd", 0)), 2),
                    "result": t.get("result"),
                })

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": period,
            "total_trades": result.get("total_trades", 0),
            "win_rate": result.get("win_rate", 0),
            "total_pnl_usd": result.get("total_pnl_usd", 0),
            "max_drawdown_pct": result.get("max_drawdown_pct", 0),
            "profit_factor": result.get("profit_factor", 0),
            "last_10_trades": last_trades,
        }

    except Exception as exc:
        logger.error(f"Backtest error: {exc}", exc_info=True)
        return err(str(exc))
