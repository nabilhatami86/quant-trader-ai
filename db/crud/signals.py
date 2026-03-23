from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import Signal


async def insert_signal(db: AsyncSession, data: dict) -> Signal:
    signal = Signal(**data)
    db.add(signal)
    await db.commit()
    await db.refresh(signal)
    return signal


async def get_recent_signals(
    db: AsyncSession, symbol: str, limit: int = 50
) -> list[Signal]:
    result = await db.execute(
        select(Signal)
        .where(Signal.symbol == symbol)
        .order_by(Signal.timestamp.desc())
        .limit(limit)
    )
    return result.scalars().all()


def build_signal_payload(result: dict) -> dict:
    """Konversi hasil bot.analyze() ke dict yang sesuai model Signal."""
    sig = result.get("signal", {})
    ml  = result.get("ml_pred", {})
    return {
        "symbol":            result.get("symbol", ""),
        "timeframe":         result.get("timeframe", ""),
        "direction":         sig.get("direction", "WAIT"),
        "score":             sig.get("score", 0.0),
        "sl":                sig.get("sl"),
        "tp":                sig.get("tp"),
        "rr_ratio":          sig.get("rr_ratio"),
        "close_price":       result.get("close", 0.0),
        "rsi":               result.get("rsi"),
        "adx":               result.get("adx"),
        "atr":               result.get("atr"),
        "macd":              result.get("macd"),
        # ML — sertakan info symbol training agar bisa diaudit
        "ml_direction":      ml.get("direction"),
        "ml_confidence":     ml.get("confidence"),
        "ml_trained_symbol": ml.get("trained_symbol", ""),
        "ml_symbol_match":   ml.get("symbol_match", True),
        # Score breakdown
        "score_technical":   sig.get("score_technical"),
        "score_volume":      sig.get("score_volume"),
        "score_smc":         sig.get("score_smc"),
        "regime":            sig.get("regime"),
        # Eksekusi
        "exec_direction":    result.get("exec_direction", "WAIT"),
        "exec_source":       result.get("exec_source"),
        "news_risk":         result.get("news_risk"),
        "raw_result":        {k: v for k, v in result.items() if k not in ("signal", "raw_result")},
    }
