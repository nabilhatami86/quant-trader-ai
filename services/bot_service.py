"""
BotService — menjalankan TradingBot di background thread.

Arsitektur:
- background thread memanggil run_analysis() dari main.py
  (logika identik dengan `python main.py --live --mt5 --real`)
- DB save dilakukan via run_coroutine_threadsafe ke main FastAPI event loop
- Singleton `bot_service` diakses oleh semua route FastAPI
"""
import asyncio
import logging
import threading
import time
from datetime import date
from typing import Optional

from config import REFRESH_INTERVAL

logger = logging.getLogger("trader_ai.bot_service")


class BotService:
    def __init__(self) -> None:
        self.bot        = None
        self.executor   = None          # SignalExecutor (MT5 auto-order)
        self.running    : bool                   = False
        self.cycle      : int                    = 0
        self.last_result: dict                   = {}
        self.last_error : str                    = ""
        self._thread    : Optional[threading.Thread] = None
        self._lock      = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    # ──────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────

    def initialize(
        self,
        symbol: str,
        timeframe: str,
        use_news: bool = True,
        mt5_connector=None,
        executor=None,
    ) -> None:
        from backend.bot import TradingBot

        self.bot = TradingBot(
            symbol=symbol,
            timeframe=timeframe,
            use_news=use_news,
            mt5_connector=mt5_connector,
        )
        self.executor = executor
        self.bot.fetch_news()
        logger.info(f"Bot initialized: {symbol} {timeframe}")
        if executor:
            logger.info("SignalExecutor attached — auto-order AKTIF")

    # ──────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.running or not self.bot:
            return
        self.running = True
        # ⚠ target harus method _bot_loop, bukan self._event_loop (asyncio loop object)
        self._thread = threading.Thread(
            target=self._bot_loop, daemon=True, name="BotThread"
        )
        self._thread.start()
        logger.info("Bot service started")

    def stop(self) -> None:
        self.running = False
        logger.info("Bot service stopped")

    # ──────────────────────────────────────────────────────────────
    # Background loop — identik dengan main.py --live --mt5
    # ──────────────────────────────────────────────────────────────

    def _bot_loop(self) -> None:
        from main import run_analysis

        news_date = date.today()

        while self.running:
            try:
                # Refresh berita setiap hari baru
                if date.today() != news_date:
                    logger.info("Hari baru — refresh berita")
                    self.bot.fetch_news()
                    news_date = date.today()

                # Jalankan analisis — lock agar run_once() tidak tabrakan
                with self._lock:
                    success, result = run_analysis(self.bot, executor=self.executor)

                if success and result:
                    with self._lock:
                        self.last_result = result
                        self.cycle      += 1
                        self.last_error  = ""

                    logger.info(
                        f"Cycle #{self.cycle} | "
                        f"{result.get('exec_direction', 'WAIT')} | "
                        f"score={result.get('signal', {}).get('score', 0):.2f}"
                    )
                    # DB save ditangani oleh save_cycle() di dalam run_analysis()

            except Exception as exc:
                with self._lock:
                    self.last_error = str(exc)
                logger.error(f"Bot error: {exc}", exc_info=True)
                self._save_error_log(str(exc))

            time.sleep(REFRESH_INTERVAL)

    # ──────────────────────────────────────────────────────────────
    # Getters
    # ──────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running"    : self.running,
                "cycle"      : self.cycle,
                "last_update": self.last_result.get("timestamp"),
                "last_error" : self.last_error,
                "symbol"     : self.bot.symbol    if self.bot else None,
                "timeframe"  : self.bot.timeframe if self.bot else None,
                "mt5_active" : self.executor is not None,
            }

    def get_last_signal(self) -> dict:
        with self._lock:
            return dict(self.last_result)

    def run_once(self) -> dict:
        """Trigger analisis manual — dipakai oleh route /signal/analyze."""
        if not self.bot:
            return {"error": "Bot belum diinisialisasi"}
        try:
            from main import run_analysis
            with self._lock:
                success, result = run_analysis(self.bot, executor=self.executor)
            if not success or result is None:
                return {"error": "Gagal load data atau analisis"}
            with self._lock:
                self.last_result = result
                self.cycle      += 1
            return result
        except Exception as exc:
            logger.error(f"run_once error: {exc}", exc_info=True)
            return {"error": str(exc)}

    # ──────────────────────────────────────────────────────────────
    # DB helpers (thread-safe via run_coroutine_threadsafe)
    # ──────────────────────────────────────────────────────────────

    def _save_to_db(self, result: dict, order_result: dict = None) -> None:
        """
        Simpan ke DB dari background thread.
        Pakai run_coroutine_threadsafe agar asyncpg pool tetap
        di main FastAPI event loop yang membuatnya.
        """
        coro = self._async_save(result, order_result)
        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            try:
                future.result(timeout=10)
            except Exception as exc:
                logger.warning(f"DB save error: {exc}")
        else:
            # Fallback: CLI context tanpa event loop aktif
            try:
                asyncio.run(coro)
            except Exception as exc:
                logger.debug(f"DB save (fallback) error: {exc}")

    async def _async_save(self, result: dict, order_result: dict = None) -> None:
        from db.database import AsyncSessionLocal
        from db.crud.signals import insert_signal, build_signal_payload
        from db.crud.candle_logs import bulk_upsert_candle_logs
        from db.crud.tx_log import log_event

        async with AsyncSessionLocal() as db:
            # 1. Signal
            try:
                await insert_signal(db, build_signal_payload(result))
            except Exception as exc:
                logger.debug(f"save signal: {exc}")

            # 2. Candle log (candle terakhir + indikator)
            try:
                if self.bot and self.bot.df_ind is not None and not self.bot.df_ind.empty:
                    df_last = self.bot.df_ind.iloc[[-1]].copy()
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
                logger.debug(f"save candle_log: {exc}")

            # 3. Tx log (SIGNAL_*)
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
                        f"Cycle #{self.cycle} | score={sig.get('score', 0):.2f} | "
                        f"{result.get('final_advice', '')}"
                    ),
                    meta={
                        "ml_direction":  result.get("ml_pred", {}).get("direction"),
                        "ml_confidence": result.get("ml_pred", {}).get("confidence"),
                        "news_risk":     result.get("news_risk"),
                        "exec_source":   result.get("exec_source"),
                        "session":       result.get("session"),
                    },
                )
            except Exception as exc:
                logger.debug(f"save tx_log: {exc}")

            # 4. Order result (ORDER_OPEN / ORDER_SKIP)
            if order_result:
                await self._save_order_to_db(db, result, order_result)

    async def _save_order_to_db(self, db, result: dict, order_result: dict) -> None:
        from db.crud.tx_log import log_event

        success  = order_result.get("success", False)
        ticket   = order_result.get("ticket")
        exec_dir = result.get("exec_direction", "WAIT")
        sig      = result.get("signal", {})

        try:
            if success:
                from db.crud.trades import insert_trade
                from datetime import datetime, timezone
                await insert_trade(db, {
                    "ticket"    : ticket,
                    "symbol"    : result.get("symbol", ""),
                    "timeframe" : result.get("timeframe", ""),
                    "direction" : sig.get("direction", ""),
                    "entry"     : result.get("close", 0),
                    "sl"        : sig.get("sl"),
                    "tp"        : sig.get("tp"),
                    "lot"       : order_result.get("lot", 0.01),
                    "result"    : "OPEN",
                    "source"    : result.get("exec_source", "bot"),
                    "opened_at" : datetime.now(tz=timezone.utc),
                })

            await log_event(
                db,
                event_type=f"ORDER_{'OPEN' if success else 'SKIP'}",
                symbol=result.get("symbol"),
                timeframe=result.get("timeframe"),
                direction=exec_dir,
                price=result.get("close"),
                sl=sig.get("sl"),
                tp=sig.get("tp"),
                ticket=ticket,
                message=(
                    f"Order {'dibuka' if success else 'dilewati'}: "
                    f"{exec_dir} ticket={ticket} | {order_result.get('error','')}"
                )[:255],
                meta={"order_result": order_result, "news_risk": result.get("news_risk")},
            )
            logger.info(
                f"Order {'OPEN' if success else 'SKIP'}: {exec_dir} ticket={ticket}"
            )
        except Exception as exc:
            logger.debug(f"save order: {exc}")

    def _save_error_log(self, error_msg: str) -> None:
        coro = self._async_save_error(error_msg)
        if self._event_loop and self._event_loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        else:
            try:
                asyncio.run(coro)
            except Exception:
                pass

    async def _async_save_error(self, error_msg: str) -> None:
        from db.database import AsyncSessionLocal
        from db.crud.tx_log import log_event
        async with AsyncSessionLocal() as db:
            await log_event(
                db,
                event_type="BOT_ERROR",
                symbol=self.bot.symbol if self.bot else None,
                message=error_msg[:255],
            )


# Singleton — diakses oleh semua route
bot_service = BotService()
