from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

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


async def update_signal_pnl(
    db: AsyncSession, ticket: int, pnl_usd: float
) -> bool:
    """Update pnl_usd di signals saat trade yang terkait tutup."""
    result = await db.execute(
        update(Signal)
        .where(Signal.ticket == ticket)
        .values(pnl_usd=pnl_usd)
    )
    await db.commit()
    return result.rowcount > 0


def build_signal_payload(result: dict, ticket: int = None) -> dict:
    """Konversi hasil bot.analyze() ke dict yang sesuai model Signal."""
    sig     = result.get("signal", {})
    ml      = result.get("ml_pred", {})
    filters = sig.get("filters", {})

    # Candle type dari last row df_ind (jika tersedia di result)
    _candle_type = None
    _df = result.get("_df_last_row")
    if _df is not None:
        ct = str(_df.get("candle_type", "") or "").upper()
        if ct in ("BULLISH", "BEARISH"):
            _candle_type = ct

    # Session bias HTF
    _session_bias = None
    try:
        from data.session_bias import get_current_bias
        _b = get_current_bias()
        if _b:
            _session_bias = _b.get("direction", "NEUTRAL")
    except Exception:
        pass

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
        # ML
        "ml_direction":      ml.get("direction"),
        "ml_confidence":     ml.get("confidence"),
        "ml_trained_symbol": ml.get("trained_symbol", ""),
        "ml_symbol_match":   ml.get("symbol_match", True),
        # Score breakdown — semua komponen
        "score_technical":   sig.get("score_technical"),
        "score_volume":      sig.get("score_volume"),
        "score_smc":         sig.get("score_smc"),
        "score_structure":   sig.get("score_structure"),
        "score_news":        sig.get("score_news"),
        "score_memory":      sig.get("score_memory"),
        "regime":            sig.get("regime"),
        # Candle info
        "candle_type":       _candle_type,
        "candle_pattern":    filters.get("candle_pattern"),
        # Session bias HTF
        "session_bias":      _session_bias,
        # Eksekusi
        "exec_direction":    result.get("exec_direction", "WAIT"),
        "exec_source":       result.get("exec_source"),
        "news_risk":         result.get("news_risk"),
        # Link ke trade (jika sinyal ini menghasilkan order)
        "ticket":            ticket,
        # Raw full result (audit)
        "raw_result":        {k: v for k, v in result.items()
                              if k not in ("signal", "raw_result", "_df_last_row")},
    }
