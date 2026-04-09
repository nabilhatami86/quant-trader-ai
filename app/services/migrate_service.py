"""
migrate_service — migrasi SEMUA data CSV ke PostgreSQL.

Jalankan via endpoint:  POST /admin/migrate
Atau via Python:
    python -c "import asyncio; from app.services.migrate_service import run_migration; asyncio.run(run_migration())"
"""
import glob
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("trader_ai.migrate")


# ── 1. data/history/*.csv → tabel candles (OHLCV) ────────────────────────────

async def migrate_candles(db) -> dict:
    from app.database.crud.candles import bulk_upsert_candles

    csv_files = glob.glob("data/history/*.csv")
    total, results = 0, []

    for path in csv_files:
        stem = Path(path).stem          # XAUUSD_5m
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        symbol, timeframe = parts[0], parts[1]
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.capitalize() for c in df.columns]
            count = await bulk_upsert_candles(db, symbol, timeframe, df)
            total += count
            results.append({"file": stem, "inserted": count, "status": "ok"})
            logger.info(f"[candles] {stem} -> {count} baris")
        except Exception as exc:
            results.append({"file": stem, "error": str(exc), "status": "error"})
            logger.error(f"[candles] Gagal {stem}: {exc}")

    return {"total_inserted": total, "files": results}


# ── 2. logs/candles_*.csv → tabel candle_logs (OHLC + indikator) ─────────────

async def migrate_candle_logs(db) -> dict:
    from app.database.crud.candle_logs import bulk_upsert_candle_logs

    csv_files = glob.glob("logs/candles_*.csv")
    total, results = 0, []

    for path in csv_files:
        stem = Path(path).stem          # candles_XAUUSD_1h
        # hapus prefix "candles_"
        name = stem.replace("candles_", "", 1)
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        symbol, timeframe = parts[0], parts[1]
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            count = await bulk_upsert_candle_logs(db, symbol, timeframe, df)
            total += count
            results.append({"file": stem, "inserted": count, "status": "ok"})
            logger.info(f"[candle_logs] {stem} -> {count} baris")
        except Exception as exc:
            results.append({"file": stem, "error": str(exc), "status": "error"})
            logger.error(f"[candle_logs] Gagal {stem}: {exc}")

    return {"total_inserted": total, "files": results}


# ── 3. logs/journal_*.csv → tabel trades ─────────────────────────────────────

async def migrate_journal(db) -> dict:
    from app.database.crud.trades import insert_trade

    journal_files = glob.glob("logs/journal_*.csv")
    total, results = 0, []

    for path in journal_files:
        fname = Path(path).name
        try:
            df = pd.read_csv(path)
            file_count = 0

            for _, row in df.iterrows():
                def _f(col, default=None):
                    v = row.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return default
                    return v

                def _dt(col):
                    v = _f(col)
                    if not v:
                        return None
                    try:
                        ts = pd.Timestamp(v)
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        return ts.to_pydatetime()
                    except Exception:
                        return None

                data = {
                    "ticket":    int(_f("ticket", 0)) or None,
                    "symbol":    str(_f("symbol", "XAUUSD")),
                    "timeframe": str(_f("timeframe", "")),
                    "direction": str(_f("direction", "")),
                    "entry":     float(_f("entry_price", 0) or 0),
                    "sl":        float(_f("sl")) if _f("sl") else None,
                    "tp":        float(_f("tp")) if _f("tp") else None,
                    "lot":       float(_f("lot", 0.01) or 0.01),
                    "close_price": float(_f("exit_price")) if _f("exit_price") else None,
                    "pnl_usd":   float(_f("pnl")) if _f("pnl") else None,
                    "result":    str(_f("result", "OPEN")),
                    "source":    str(_f("source", "csv_import")),
                    "opened_at": _dt("entry_time"),
                    "closed_at": _dt("exit_time"),
                }
                try:
                    await insert_trade(db, data)
                    file_count += 1
                except Exception:
                    pass   # skip duplikat atau baris invalid

            total += file_count
            results.append({"file": fname, "inserted": file_count, "status": "ok"})
            logger.info(f"[journal] {fname} -> {file_count} baris")
        except Exception as exc:
            results.append({"file": fname, "error": str(exc), "status": "error"})
            logger.error(f"[journal] Gagal {fname}: {exc}")

    return {"total_inserted": total, "files": results}


# ── 4. Catat event MIGRATE ke tx_log ─────────────────────────────────────────

async def _log_migration(db, candles: dict, candle_logs: dict, journal: dict):
    try:
        from app.database.crud.tx_log import log_event
        total = (
            candles["total_inserted"]
            + candle_logs["total_inserted"]
            + journal["total_inserted"]
        )
        await log_event(
            db,
            event_type="MIGRATE",
            message=f"CSV migration selesai: {total} total baris masuk DB",
            meta={
                "candles":     candles,
                "candle_logs": candle_logs,
                "journal":     journal,
            },
        )
    except Exception as exc:
        logger.warning(f"Gagal catat tx_log migrasi: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_migration(db=None) -> dict:
    """Jalankan semua migrasi sekaligus."""

    async def _run(session):
        candles     = await migrate_candles(session)
        candle_logs = await migrate_candle_logs(session)
        journal     = await migrate_journal(session)
        await _log_migration(session, candles, candle_logs, journal)
        return {
            "candles":     candles,
            "candle_logs": candle_logs,
            "journal":     journal,
            "grand_total": (
                candles["total_inserted"]
                + candle_logs["total_inserted"]
                + journal["total_inserted"]
            ),
        }

    if db is not None:
        return await _run(db)

    from app.database.session import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        return await _run(session)
