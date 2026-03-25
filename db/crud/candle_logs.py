from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
import pandas as pd

from db.models import CandleLog


async def bulk_upsert_candle_logs(
    db: AsyncSession, symbol: str, timeframe: str, df: pd.DataFrame
) -> int:
    """Insert candle logs dari DataFrame. Skip duplikat."""
    if df.empty:
        return 0

    # Normalisasi index ke UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    def _safe(row, col, cast=float):
        v = row.get(col)
        if v is None or (hasattr(v, "__class__") and v.__class__.__name__ == "float" and str(v) == "nan"):
            return None
        try:
            return cast(v)
        except Exception:
            return None

    rows = []
    for ts, row in df.iterrows():
        signal_raw = str(row.get("signal", "")) if row.get("signal") else None
        signal_dir = signal_raw if signal_raw in ("BUY", "SELL", "WAIT") else None

        logged_raw = row.get("logged_at")
        logged_at = None
        if logged_raw and str(logged_raw) != "nan":
            try:
                import pandas as _pd
                logged_at = _pd.Timestamp(logged_raw).tz_localize("UTC")
            except Exception:
                pass

        def _safe_int(row, col):
            v = row.get(col)
            if v is None or str(v) == "nan":
                return None
            try:
                return int(float(v))
            except Exception:
                return None

        rows.append({
            "symbol":      symbol,
            "timeframe":   timeframe,
            "timestamp":   ts,
            "open":        _safe(row, "open"),
            "high":        _safe(row, "high"),
            "low":         _safe(row, "low"),
            "close":       _safe(row, "close"),
            "candle_type": str(row.get("candle", ""))[:10] if row.get("candle") else None,
            "body":        _safe(row, "body"),
            "wick_up":     _safe(row, "wick_up"),
            "wick_down":   _safe(row, "wick_down"),
            "pattern":     str(row.get("candle_name", row.get("pattern", "")))[:50]
                           if row.get("candle_name") and str(row.get("candle_name")) not in ("nan", "None") else None,
            "rsi":         _safe(row, "rsi"),
            "ema20":       _safe(row, f"ema_20") or _safe(row, "ema20"),
            "ema50":       _safe(row, f"ema_50") or _safe(row, "ema50"),
            "macd":        _safe(row, "macd"),
            "histogram":   _safe(row, "histogram"),
            "adx":         _safe(row, "adx"),
            "atr":         _safe(row, "atr"),
            "signal_dir":  signal_dir,
            "score":       _safe(row, "score"),
            "sl":          _safe(row, "sl"),
            "tp":          _safe(row, "tp"),
            # Volume
            "obv":         _safe(row, "obv"),
            "vwap":        _safe(row, "vwap"),
            "williams_r":  _safe(row, "williams_r"),
            "cci":         _safe(row, "cci"),
            "vol_ratio":   _safe(row, "vol_ratio"),
            # SMC
            "fvg_bull":       _safe_int(row, "fvg_bull"),
            "fvg_bear":       _safe_int(row, "fvg_bear"),
            "ob_bull":        _safe_int(row, "ob_bull"),
            "ob_bear":        _safe_int(row, "ob_bear"),
            "bos_bull":       _safe_int(row, "bos_bull"),
            "bos_bear":       _safe_int(row, "bos_bear"),
            "choch_bull":     _safe_int(row, "choch_bull"),
            "choch_bear":     _safe_int(row, "choch_bear"),
            "liq_bull_sweep": _safe_int(row, "liq_bull_sweep"),
            "liq_bear_sweep": _safe_int(row, "liq_bear_sweep"),
            "regime":         str(row.get("regime", ""))[:10] if row.get("regime") and str(row.get("regime")) != "nan" else None,
            "candle_ex":      _safe_int(row, "candle_ex"),
            "logged_at":   logged_at,
        })

    BATCH_SIZE = 1000
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        stmt = pg_insert(CandleLog).values(batch)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_candle_log")
        await db.execute(stmt)

    await db.commit()
    return len(rows)


async def get_candle_logs(
    db: AsyncSession, symbol: str, timeframe: str, limit: int = 200
) -> list[CandleLog]:
    result = await db.execute(
        select(CandleLog)
        .where(CandleLog.symbol == symbol, CandleLog.timeframe == timeframe)
        .order_by(CandleLog.timestamp.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def get_candle_logs_df(
    db: AsyncSession, symbol: str, timeframe: str, limit: int = 5000
) -> "pd.DataFrame":
    """Fetch candle_logs sebagai DataFrame — kolom sama dengan CSV candle log."""
    result = await db.execute(
        select(CandleLog)
        .where(CandleLog.symbol == symbol, CandleLog.timeframe == timeframe)
        .order_by(CandleLog.timestamp.asc())
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        records.append({
            "time":        str(r.timestamp)[:19],
            "open":        r.open,
            "high":        r.high,
            "low":         r.low,
            "close":       r.close,
            "candle":      r.candle_type or "",
            "body":        r.body,
            "wick_up":     r.wick_up,
            "wick_down":   r.wick_down,
            "pattern":     r.pattern or "",
            "rsi":         r.rsi,
            "ema20":       r.ema20,
            "ema50":       r.ema50,
            "macd":        r.macd,
            "histogram":   r.histogram,
            "adx":         r.adx,
            "atr":         r.atr,
            "signal":      r.signal_dir or "",
            "score":       r.score,
            "sl":          r.sl,
            "tp":          r.tp,
            # SMC
            "fvg_bull":    r.fvg_bull,
            "fvg_bear":    r.fvg_bear,
            "ob_bull":     r.ob_bull,
            "ob_bear":     r.ob_bear,
            "bos_bull":    r.bos_bull,
            "bos_bear":    r.bos_bear,
            "choch_bull":  r.choch_bull,
            "choch_bear":  r.choch_bear,
            "regime":      r.regime or "",
            # Volume
            "obv":         r.obv,
            "vwap":        r.vwap,
            "williams_r":  r.williams_r,
            "cci":         r.cci,
            "vol_ratio":   r.vol_ratio,
            # Outcome
            "outcome":     r.outcome or "",
            "outcome_pct": r.outcome_pct,
            "logged_at":   str(r.logged_at)[:19] if r.logged_at else "",
        })
    return pd.DataFrame(records)


async def update_outcomes_batch(
    db: AsyncSession, symbol: str, timeframe: str,
    updates: list[dict]
) -> int:
    """
    Update outcome + outcome_pct untuk baris tertentu.
    updates: [{"timestamp": datetime, "outcome": "WIN", "outcome_pct": 0.05}, ...]
    """
    from sqlalchemy import update as sa_update
    count = 0
    for u in updates:
        stmt = (
            sa_update(CandleLog)
            .where(
                CandleLog.symbol    == symbol,
                CandleLog.timeframe == timeframe,
                CandleLog.timestamp == u["timestamp"],
            )
            .values(outcome=u["outcome"], outcome_pct=u["outcome_pct"])
        )
        await db.execute(stmt)
        count += 1
    await db.commit()
    return count
