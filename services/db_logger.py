"""
db_logger.py — Sync wrapper untuk menyimpan hasil analisis bot ke PostgreSQL.
Dipanggil dari main.py (sync context) maupun bot_service.py (thread).
"""
import asyncio
import logging

logger = logging.getLogger("trader_ai.db_logger")


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
    from db.database import AsyncSessionLocal, engine
    from db.crud.signals import insert_signal, build_signal_payload
    from db.crud.candle_logs import bulk_upsert_candle_logs
    from db.crud.tx_log import log_event

    try:
        async with AsyncSessionLocal() as db:

            # ── 1. Simpan sinyal ──────────────────────────────────────
            try:
                await insert_signal(db, build_signal_payload(result))
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
                        # Dari mana sinyal ini berasal
                        "exec_source":       result.get("exec_source"),         # Rule-Based / ML(70%) / WAIT
                        "data_source":       result.get("data_source", "MT5"),  # MT5 / Yahoo / TV
                        # ML info
                        "ml_direction":      result.get("ml_pred", {}).get("direction"),
                        "ml_confidence":     result.get("ml_pred", {}).get("confidence"),
                        "ml_trained_symbol": result.get("ml_pred", {}).get("trained_symbol"),
                        "ml_symbol_match":   result.get("ml_pred", {}).get("symbol_match", True),
                        # Score breakdown
                        "score_technical":   sig.get("score_technical"),
                        "score_volume":      sig.get("score_volume"),
                        "score_smc":         sig.get("score_smc"),
                        "regime":            sig.get("regime"),
                        # Context
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
    finally:
        await engine.dispose()


def save_order_skip(result: dict, reason: str) -> None:
    """Catat ORDER_SKIP ke tx_log."""
    try:
        asyncio.run(_async_log_skip(result, reason))
    except Exception:
        pass


async def _async_log_skip(result: dict, reason: str) -> None:
    from db.database import AsyncSessionLocal, engine
    from db.crud.tx_log import log_event
    sig = result.get("signal", {})
    try:
        async with AsyncSessionLocal() as db:
            await log_event(
                db,
                event_type="ORDER_SKIP",
                symbol=result.get("symbol"),
                timeframe=result.get("timeframe"),
                direction=sig.get("direction"),
                price=result.get("close"),
                message=reason[:255],
            )
    finally:
        await engine.dispose()


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
    from db.database import AsyncSessionLocal, engine
    from db.crud.candles import bulk_upsert_candles
    try:
        async with AsyncSessionLocal() as db:
            return await bulk_upsert_candles(db, symbol, timeframe, df)
    except Exception as exc:
        logger.debug(f"_async_save_candles: {exc}")
        return 0
    finally:
        await engine.dispose()


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
    from db.database import AsyncSessionLocal, engine
    from db.crud.candle_logs import bulk_upsert_candle_logs
    try:
        df = df_ind.copy()
        df.columns = [c.lower() for c in df.columns]
        async with AsyncSessionLocal() as db:
            return await bulk_upsert_candle_logs(db, symbol, timeframe, df)
    except Exception as exc:
        logger.debug(f"_async_save_candle_logs: {exc}")
        return 0
    finally:
        await engine.dispose()


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
    from db.database import AsyncSessionLocal, engine
    from db.crud.tx_log import log_event
    try:
        async with AsyncSessionLocal() as db:
            await log_event(db, event_type=event_type, symbol=symbol,
                            timeframe=timeframe, message=msg)
    finally:
        await engine.dispose()
