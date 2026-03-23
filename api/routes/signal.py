import logging

from fastapi import APIRouter, Depends

from api.deps import get_bot, get_db
from db.crud.signals import build_signal_payload, get_recent_signals, insert_signal
from schemas.signal import SignalHistoryOut, SignalOut
from services.bot_service import BotService
from utils.response import err, ok

logger = logging.getLogger("trader_ai.routes.signal")
router = APIRouter(prefix="/signal", tags=["Signal"])


@router.get("", summary="Sinyal terakhir dari bot")
async def get_current_signal(bot: BotService = Depends(get_bot)):
    result = bot.get_last_signal()
    if not result:
        return err("Bot belum menghasilkan sinyal — tunggu beberapa detik")

    sig = result.get("signal", {})
    return {
        "symbol": result.get("symbol"),
        "timeframe": result.get("timeframe"),
        "timestamp": result.get("timestamp"),
        "direction": sig.get("direction", "WAIT"),
        "score": sig.get("score", 0),
        "sl": sig.get("sl"),
        "tp": sig.get("tp"),
        "rr_ratio": sig.get("rr_ratio"),
        "close": result.get("close"),
        "rsi": result.get("rsi"),
        "adx": result.get("adx"),
        "ml_direction": result.get("ml_pred", {}).get("direction"),
        "ml_confidence": result.get("ml_pred", {}).get("confidence"),
        "news_risk": result.get("news_risk"),
        "exec_direction": result.get("exec_direction", "WAIT"),
        "exec_source": result.get("exec_source"),
        "final_advice": result.get("final_advice"),
    }


@router.post("/analyze", summary="Trigger analisis manual (1 siklus)")
async def trigger_analyze(
    bot: BotService = Depends(get_bot),
    db=Depends(get_db),
):
    result = bot.run_once()
    if "error" in result:
        return err(result["error"])

    # Simpan sinyal ke DB
    try:
        payload = build_signal_payload(result)
        await insert_signal(db, payload)
    except Exception as exc:
        logger.warning(f"Gagal simpan sinyal ke DB: {exc}")

    sig = result.get("signal", {})
    return ok(
        message="Analisis selesai",
        data={
            "exec_direction": result.get("exec_direction", "WAIT"),
            "score": sig.get("score", 0),
            "cycle": bot.cycle,
        },
    )


@router.get("/full", summary="Hasil analisis lengkap (semua indikator + ML + news)")
async def get_full_analysis(bot: BotService = Depends(get_bot)):
    result = bot.get_last_signal()
    if not result:
        return {"error": "Bot belum menghasilkan sinyal — tunggu beberapa detik"}

    sig = result.get("signal", {})
    ml  = result.get("ml_pred", {})
    nb  = result.get("news_bias", {})

    return {
        # ── Info umum ──────────────────────────────────────────
        "symbol":       result.get("symbol"),
        "timeframe":    result.get("timeframe"),
        "timestamp":    result.get("timestamp"),
        "session":      result.get("session"),
        "cycle":        bot.cycle,

        # ── OHLC ───────────────────────────────────────────────
        "candle": {
            "open":        result.get("open"),
            "high":        result.get("high"),
            "low":         result.get("low"),
            "close":       result.get("close"),
            "volume":      result.get("volume"),
            "candle_name": result.get("candle_name"),
        },

        # ── Indikator ──────────────────────────────────────────
        "indicators": {
            "rsi":       result.get("rsi"),
            "macd":      result.get("macd"),
            "histogram": result.get("histogram"),
            "adx":       result.get("adx"),
            "atr":       result.get("atr"),
            "stoch_k":   result.get("stoch_k"),
            "stoch_d":   result.get("stoch_d"),
            "bb_upper":  result.get("bb_upper"),
            "bb_lower":  result.get("bb_lower"),
        },

        # ── Rule-based signal ──────────────────────────────────
        "signal": {
            "direction":      sig.get("direction", "WAIT"),
            "score":          sig.get("score", 0),
            "score_technical":sig.get("score_technical", 0),
            "score_news":     sig.get("score_news", 0),
            "sl":             sig.get("sl"),
            "tp":             sig.get("tp"),
            "rr_ratio":       sig.get("rr_ratio"),
            "method_sl":      sig.get("method_sl"),
            "method_tp":      sig.get("method_tp"),
            "reasons":        sig.get("reasons", []),
        },

        # ── ML prediction ──────────────────────────────────────
        "ml": {
            "direction":   ml.get("direction", "WAIT"),
            "confidence":  ml.get("confidence", 0),
            "proba_buy":   ml.get("proba_buy", 0),
            "proba_sell":  ml.get("proba_sell", 0),
            "uncertain":   ml.get("uncertain", False),
        },

        # ── News ───────────────────────────────────────────────
        "news": {
            "risk":      result.get("news_risk"),
            "bias":      nb.get("bias", "NEUTRAL"),
            "score":     nb.get("score", 0),
            "confidence":nb.get("confidence", "LOW"),
            "reasons":   nb.get("reasons", [])[:5],
        },

        # ── Keputusan final ────────────────────────────────────
        "decision": {
            "exec_direction": result.get("exec_direction", "WAIT"),
            "exec_source":    result.get("exec_source"),
            "consensus":      result.get("consensus"),
            "final_advice":   result.get("final_advice"),
        },
    }


@router.get("/history", summary="Riwayat sinyal dari database")
async def get_signal_history(
    symbol: str = "XAUUSD",
    limit: int = 50,
    db=Depends(get_db),
):
    signals = await get_recent_signals(db, symbol, limit)
    return {
        "symbol": symbol,
        "total": len(signals),
        "signals": [SignalHistoryOut.model_validate(s).model_dump() for s in signals],
    }
