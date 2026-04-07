"""
db_logger.py — Sync wrapper untuk menyimpan hasil analisis bot ke PostgreSQL.

Dipanggil dari main.py (sync context) maupun bot_service.py (thread).
Semua fungsi di sini bersifat fire-and-forget — kegagalan DB tidak mengganggu bot.

Fungsi utama:
  save_cycle(result, bot, order_result)
      Simpan satu siklus analisis ke DB sekaligus:
      - signals       : tabel signals (direction, score, indikator)
      - candle_logs   : tabel candle_logs (OHLCV + 99 kolom indikator)
      - tx_log        : event ANALYSIS
      - trades        : jika order_result ada (OPEN position)

  save_order_skip(result, reason)
      Catat ke tx_log mengapa sinyal tidak dieksekusi (cooldown, filter, dll)

  save_candles_batch(df, symbol, tf)
      Bulk insert candle OHLCV ke tabel candles (histori)

  save_candle_logs_batch(df_ind, symbol, tf)
      Bulk insert candle + indikator ke tabel candle_logs

  save_bot_start(symbol, tf, mode)
      Catat event BOT_START ke tx_log

  save_bot_stop(symbol, tf)
      Catat event BOT_STOP ke tx_log

  close_trade_in_db(ticket, close_price, pnl, result, ...)
      Update tabel trades: set exit_price, pnl, result, closed_at
      + catat ORDER_CLOSE ke tx_log + update signals.pnl_usd

  get_trade_journal_df(symbol, limit) -> DataFrame
      Ambil histori trade dari DB (fallback ke CSV jika DB tidak tersedia)

  get_trade_stats_db(symbol) -> dict
      Statistik trade dari DB: total, wins, losses, win_rate, total_pnl

  update_outcomes_in_db(symbol, tf, updates)
      Update kolom outcome di candle_logs (dari signal accuracy tracker)

Database:
  PostgreSQL via SQLAlchemy async (asyncpg driver)
  Schema: db/models.py — setiap fungsi buat engine NullPool baru (thread-safe)
  Koneksi: db/database.py (DATABASE_URL dari environment)
  Fallback: CSV journal jika DB tidak tersedia (lihat get_trade_journal_df)

Catatan threading:
  Semua fungsi async di sini membuat engine NullPool baru via make_thread_engine()
  sehingga aman dipanggil dari asyncio.run() di background thread tanpa conflict
  dengan asyncpg connection pool milik FastAPI.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger("trader_ai.db_logger")


# ── Helper: NullPool session — aman dari background thread ────────────────

@asynccontextmanager
async def _tdb():
    """
    Buat sesi DB dengan NullPool engine — tidak ada connection pool sehingga
    aman dipakai dari asyncio.run() di thread mana pun tanpa conflict asyncpg.
    """
    from db.database import make_thread_engine
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    _engine = make_thread_engine()
    _Session = async_sessionmaker(bind=_engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with _Session() as db:
            yield db
    finally:
        await _engine.dispose()


# ── Save cycle (sinyal + candle + tx_log + order) ──────────────────────────

def save_cycle(result: dict, bot=None, order_result: dict = None) -> None:
    """
    Simpan satu siklus analisis ke DB:
    - signals       → tabel signals
    - candle + ind  → tabel candle_logs
    - event         → tabel tx_log
    - order (opsional) → tabel trades + tx_log ORDER_OPEN
    """
    try:
        asyncio.run(_async_save_cycle(result, bot, order_result))
    except RuntimeError:
        # Jika event loop sudah jalan (FastAPI context), skip — ditangani bot_service
        pass
    except Exception as exc:
        logger.warning(f"db_logger.save_cycle error: {exc}")


async def _async_save_cycle(result: dict, bot=None, order_result: dict = None) -> None:
    from db.crud.signals import insert_signal, build_signal_payload
    from db.crud.candle_logs import bulk_upsert_candle_logs
    from db.crud.tx_log import log_event

    # Ticket dari order_result (jika sinyal menghasilkan order)
    _ticket = order_result.get("ticket") if order_result and order_result.get("success") else None

    try:
        async with _tdb() as db:

            # ── 1. Simpan sinyal ──────────────────────────────────────
            try:
                # Sisipkan last row df_ind agar build_signal_payload bisa
                # ambil candle_type BULLISH/BEARISH dari data frame
                if bot and bot.df_ind is not None and not bot.df_ind.empty:
                    _last = bot.df_ind.iloc[-1]
                    result["_df_last_row"] = _last.to_dict()
                await insert_signal(db, build_signal_payload(result, ticket=_ticket))
            except Exception as exc:
                logger.debug(f"insert_signal: {exc}")

            # ── 2. Simpan candle log (candle terakhir + indikator) ────
            try:
                if bot and bot.df_ind is not None and not bot.df_ind.empty:
                    df_last = bot.df_ind.iloc[[-1]].copy()
                    df_last.columns = [c.lower() for c in df_last.columns]
                    sig = result.get("signal", {})
                    df_last["signal"]    = sig.get("direction", "")
                    df_last["score"]     = sig.get("score", 0)
                    df_last["sl"]        = sig.get("sl")
                    df_last["tp"]        = sig.get("tp")
                    df_last["logged_at"] = result.get("timestamp")
                    await bulk_upsert_candle_logs(
                        db,
                        result.get("symbol", ""),
                        result.get("timeframe", ""),
                        df_last,
                    )
            except Exception as exc:
                logger.debug(f"insert_candle_log: {exc}")

            # ── 3. Catat ke tx_log ────────────────────────────────────
            try:
                sig      = result.get("signal", {})
                exec_dir = result.get("exec_direction", "WAIT")
                await log_event(
                    db,
                    event_type=f"SIGNAL_{exec_dir}",
                    symbol=result.get("symbol"),
                    timeframe=result.get("timeframe"),
                    direction=exec_dir,
                    price=result.get("close"),
                    sl=sig.get("sl"),
                    tp=sig.get("tp"),
                    message=(
                        f"score={sig.get('score', 0):.2f} | "
                        f"{result.get('final_advice', '')}"
                    ),
                    meta={
                        "exec_source":       result.get("exec_source"),
                        "data_source":       result.get("data_source", "MT5"),
                        "ml_direction":      result.get("ml_pred", {}).get("direction"),
                        "ml_confidence":     result.get("ml_pred", {}).get("confidence"),
                        "ml_trained_symbol": result.get("ml_pred", {}).get("trained_symbol"),
                        "ml_symbol_match":   result.get("ml_pred", {}).get("symbol_match", True),
                        "score_technical":   sig.get("score_technical"),
                        "score_volume":      sig.get("score_volume"),
                        "score_smc":         sig.get("score_smc"),
                        "score_structure":   sig.get("score_structure"),
                        "score_news":        sig.get("score_news"),
                        "score_memory":      sig.get("score_memory"),
                        "regime":            sig.get("regime"),
                        "news_risk":         result.get("news_risk"),
                        "session":           result.get("session"),
                    },
                )
            except Exception as exc:
                logger.debug(f"tx_log signal: {exc}")

            # ── 4. Catat order ke trades + tx_log (jika ada) ──────────
            if order_result and order_result.get("success"):
                try:
                    from db.crud.trades import insert_trade
                    from datetime import datetime, timezone

                    sig = result.get("signal", {})
                    await insert_trade(db, {
                        "ticket":    order_result.get("ticket"),
                        "symbol":    result.get("symbol", ""),
                        "timeframe": result.get("timeframe", ""),
                        "direction": sig.get("direction", ""),
                        "entry":     result.get("close", 0),
                        "sl":        sig.get("sl"),
                        "tp":        sig.get("tp"),
                        "lot":       order_result.get("lot", 0.01),
                        "result":    "OPEN",
                        "source":    result.get("exec_source", "bot"),
                        "opened_at": datetime.now(tz=timezone.utc),
                    })

                    await log_event(
                        db,
                        event_type="ORDER_OPEN",
                        symbol=result.get("symbol"),
                        timeframe=result.get("timeframe"),
                        direction=sig.get("direction"),
                        price=result.get("close"),
                        sl=sig.get("sl"),
                        tp=sig.get("tp"),
                        lot=order_result.get("lot", 0.01),
                        ticket=order_result.get("ticket"),
                        message=f"Order #{order_result.get('ticket')} opened",
                    )
                except Exception as exc:
                    logger.debug(f"insert_trade/order_log: {exc}")

    except Exception as exc:
        logger.debug(f"_async_save_cycle: {exc}")


def save_order_skip(result: dict, reason: str) -> None:
    """Catat ORDER_SKIP ke tx_log."""
    try:
        asyncio.run(_async_log_skip(result, reason))
    except Exception:
        pass


async def _async_log_skip(result: dict, reason: str) -> None:
    from db.crud.tx_log import log_event
    sig = result.get("signal", {})
    try:
        async with _tdb() as db:
            await log_event(
                db,
                event_type="ORDER_SKIP",
                symbol=result.get("symbol"),
                timeframe=result.get("timeframe"),
                direction=sig.get("direction"),
                price=result.get("close"),
                message=reason[:255],
            )
    except Exception as exc:
        logger.debug(f"_async_log_skip: {exc}")


def save_candles_batch(df, symbol: str, timeframe: str) -> int:
    """
    Simpan semua candle OHLCV dari DataFrame ke tabel `candles` di PostgreSQL.
    Dipanggil dari bot.load_data() setiap siklus.
    """
    try:
        return asyncio.run(_async_save_candles(df, symbol, timeframe))
    except RuntimeError:
        pass
    except Exception as exc:
        logger.warning(f"save_candles_batch error: {exc}")
    return 0


async def _async_save_candles(df, symbol: str, timeframe: str) -> int:
    from db.crud.candles import bulk_upsert_candles
    try:
        async with _tdb() as db:
            return await bulk_upsert_candles(db, symbol, timeframe, df)
    except Exception as exc:
        logger.debug(f"_async_save_candles: {exc}")
        return 0


def save_candle_logs_batch(df_ind, symbol: str, timeframe: str) -> int:
    """
    Simpan candle + indikator ke tabel `candle_logs` di PostgreSQL.
    Dipanggil dari bot.load_data() setelah indikator dihitung.
    """
    try:
        return asyncio.run(_async_save_candle_logs(df_ind, symbol, timeframe))
    except RuntimeError:
        pass
    except Exception as exc:
        logger.warning(f"save_candle_logs_batch error: {exc}")
    return 0


async def _async_save_candle_logs(df_ind, symbol: str, timeframe: str) -> int:
    from db.crud.candle_logs import bulk_upsert_candle_logs
    try:
        df = df_ind.copy()
        df.columns = [c.lower() for c in df.columns]
        async with _tdb() as db:
            return await bulk_upsert_candle_logs(db, symbol, timeframe, df)
    except Exception as exc:
        logger.debug(f"_async_save_candle_logs: {exc}")
        return 0


def save_bot_start(symbol: str, timeframe: str, mode: str = "CLI") -> None:
    try:
        asyncio.run(_async_bot_event("BOT_START", symbol, timeframe, f"CLI started ({mode})"))
    except Exception:
        pass


def save_bot_stop(symbol: str, timeframe: str) -> None:
    try:
        asyncio.run(_async_bot_event("BOT_STOP", symbol, timeframe, "CLI stopped"))
    except Exception:
        pass


async def _async_bot_event(event_type: str, symbol: str, timeframe: str, msg: str) -> None:
    from db.crud.tx_log import log_event
    try:
        async with _tdb() as db:
            await log_event(db, event_type=event_type, symbol=symbol,
                            timeframe=timeframe, message=msg)
    except Exception as exc:
        logger.debug(f"_async_bot_event: {exc}")


# ── Trade close / journal dari DB ─────────────────────────────────────────

def close_trade_in_db(ticket: int, close_price: float,
                      pnl_usd: float, result: str, note: str = "",
                      symbol: str = "", timeframe: str = "",
                      direction: str = "") -> bool:
    """Update trade di DB saat posisi tutup (WIN/LOSS/MANUAL) + catat ORDER_CLOSE ke tx_log."""
    try:
        return asyncio.run(_async_close_trade(
            ticket, close_price, pnl_usd, result, note, symbol, timeframe, direction
        ))
    except RuntimeError:
        return False
    except Exception as exc:
        logger.warning(f"close_trade_in_db: {exc}")
        return False


async def _async_close_trade(ticket: int, close_price: float,
                              pnl_usd: float, result: str, note: str,
                              symbol: str, timeframe: str, direction: str) -> bool:
    from db.crud.trades import close_trade
    from db.crud.tx_log import log_event
    from db.crud.signals import update_signal_pnl
    try:
        async with _tdb() as db:
            ok = await close_trade(db, ticket, close_price, pnl_usd, result, note)

            # Catat ORDER_CLOSE ke tx_log
            try:
                await log_event(
                    db,
                    event_type="ORDER_CLOSE",
                    symbol=symbol or None,
                    timeframe=timeframe or None,
                    direction=direction or None,
                    price=close_price,
                    ticket=ticket,
                    pnl_usd=pnl_usd,
                    message=f"#{ticket} {result} | {note[:100] if note else ''} | PnL=${pnl_usd:.2f}",
                )
            except Exception as exc:
                logger.debug(f"ORDER_CLOSE tx_log: {exc}")

            # Update pnl_usd di tabel signals yang linked ke ticket ini
            try:
                await update_signal_pnl(db, ticket, pnl_usd)
            except Exception as exc:
                logger.debug(f"update_signal_pnl: {exc}")

        return ok
    except Exception as exc:
        logger.debug(f"_async_close_trade: {exc}")
        return False


def update_signal_pnl_in_db(ticket: int, pnl_usd: float) -> bool:
    """Update pnl_usd di tabel signals yang terkait ticket ini."""
    try:
        return asyncio.run(_async_update_signal_pnl(ticket, pnl_usd))
    except Exception:
        return False


async def _async_update_signal_pnl(ticket: int, pnl_usd: float) -> bool:
    from db.crud.signals import update_signal_pnl
    try:
        async with _tdb() as db:
            return await update_signal_pnl(db, ticket, pnl_usd)
    except Exception as exc:
        logger.debug(f"_async_update_signal_pnl: {exc}")
        return False


def get_trade_journal_df(symbol: str = "", limit: int = 30) -> "pd.DataFrame":
    """Ambil riwayat trade dari DB sebagai DataFrame (DB-first, fallback CSV)."""
    try:
        return asyncio.run(_async_get_trade_journal(symbol, limit))
    except Exception:
        return _csv_journal_df(symbol, limit)


async def _async_get_trade_journal(symbol: str, limit: int) -> "pd.DataFrame":
    import pandas as pd
    from db.crud.trades import get_recent_trades
    try:
        async with _tdb() as db:
            trades = await get_recent_trades(db, symbol, limit)
        if not trades:
            return _csv_journal_df(symbol, limit)
        rows = []
        for t in trades:
            rows.append({
                "ticket":      t.ticket,
                "symbol":      t.symbol,
                "timeframe":   t.timeframe,
                "direction":   t.direction,
                "entry_price": t.entry,
                "sl":          t.sl,
                "tp":          t.tp,
                "lot":         t.lot,
                "source":      t.source or "",
                "entry_time":  str(t.opened_at)[:16] if t.opened_at else "",
                "exit_time":   str(t.closed_at)[:16] if t.closed_at else "",
                "exit_price":  t.close_price or "",
                "result":      t.result,
                "pnl":         t.pnl_usd if t.pnl_usd is not None else "",
                "note":        "",
            })
        return pd.DataFrame(rows)
    except Exception:
        return _csv_journal_df(symbol, limit)


def _csv_journal_df(symbol: str, limit: int) -> "pd.DataFrame":
    """Fallback: baca dari CSV journal."""
    from data.trade_journal import get_recent_trades as csv_recent
    return csv_recent(symbol, limit)


def get_trade_stats_db(symbol: str = "") -> dict:
    """Ambil statistik trade dari DB."""
    try:
        return asyncio.run(_async_trade_stats(symbol))
    except Exception:
        return {}


async def _async_trade_stats(symbol: str) -> dict:
    from db.crud.trades import get_trade_stats
    try:
        async with _tdb() as db:
            return await get_trade_stats(db, symbol or None)
    except Exception:
        return {}


# ── Candle log dari DB ─────────────────────────────────────────────────────

def ensure_outcome_columns() -> None:
    """
    Tambah kolom outcome & outcome_pct ke tabel candle_logs,
    dan kolom baru ke tabel signals — kalau belum ada.
    Dipanggil sekali saat bot start.
    """
    try:
        asyncio.run(_async_ensure_outcome_cols())
    except Exception:
        pass


async def _async_ensure_outcome_cols() -> None:
    import sqlalchemy
    from db.database import make_thread_engine
    _engine = make_thread_engine()
    _text = sqlalchemy.text
    try:
        async with _engine.begin() as conn:
            for sql in [
                "ALTER TABLE candle_logs ADD COLUMN IF NOT EXISTS outcome VARCHAR(5);",
                "ALTER TABLE candle_logs ADD COLUMN IF NOT EXISTS outcome_pct FLOAT;",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS score_structure FLOAT;",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS score_news FLOAT;",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS score_memory FLOAT;",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS candle_type VARCHAR(10);",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS candle_pattern VARCHAR(100);",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS session_bias VARCHAR(20);",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ticket INTEGER;",
                "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pnl_usd FLOAT;",
            ]:
                try:
                    await conn.execute(_text(sql))
                except Exception:
                    pass
    finally:
        await _engine.dispose()


def load_candle_logs_df(symbol: str, timeframe: str, limit: int = 5000) -> "pd.DataFrame":
    """
    Ambil candle_logs dari PostgreSQL sebagai DataFrame.
    Fallback ke DataFrame kosong kalau DB tidak tersedia.
    """
    try:
        import pandas as pd
        return asyncio.run(_async_load_candle_logs_df(symbol, timeframe, limit))
    except Exception:
        import pandas as pd
        return pd.DataFrame()


async def _async_load_candle_logs_df(symbol: str, timeframe: str, limit: int) -> "pd.DataFrame":
    from db.crud.candle_logs import get_candle_logs_df
    try:
        async with _tdb() as db:
            return await get_candle_logs_df(db, symbol, timeframe, limit)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


def update_outcomes_in_db(symbol: str, timeframe: str, updates: list) -> int:
    """Update outcome/outcome_pct di DB. updates = list of dict."""
    if not updates:
        return 0
    try:
        return asyncio.run(_async_update_outcomes(symbol, timeframe, updates))
    except Exception:
        return 0


async def _async_update_outcomes(symbol: str, timeframe: str, updates: list) -> int:
    from db.crud.candle_logs import update_outcomes_batch
    try:
        async with _tdb() as db:
            return await update_outcomes_batch(db, symbol, timeframe, updates)
    except Exception:
        return 0
