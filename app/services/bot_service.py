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
        self._paused    : bool                   = False
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
        from app.engine.bot import TradingBot

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

                # Skip jika di-pause
                if self._paused:
                    time.sleep(REFRESH_INTERVAL)
                    continue

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
        from app.database.session import AsyncSessionLocal
        from app.database.crud.signals import insert_signal, build_signal_payload
        from app.database.crud.candle_logs import bulk_upsert_candle_logs
        from app.database.crud.tx_log import log_event

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
        from app.database.crud.tx_log import log_event

        success  = order_result.get("success", False)
        ticket   = order_result.get("ticket")
        exec_dir = result.get("exec_direction", "WAIT")
        sig      = result.get("signal", {})

        try:
            if success:
                from app.database.crud.trades import insert_trade
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

    # ──────────────────────────────────────────────────────────────
    # Data getters — dipakai oleh route FE
    # ──────────────────────────────────────────────────────────────

    def get_scalp_signal(self) -> dict:
        """Return ScalpML sinyal + analisis candle terbaru."""
        with self._lock:
            r = self.last_result
            sp = r.get("scalp_pred", {})
            return {
                "direction"       : sp.get("direction", "WAIT"),
                "prob_buy"        : sp.get("prob_buy", 0.0),
                "prob_sell"       : sp.get("prob_sell", 0.0),
                "confidence"      : sp.get("confidence", ""),
                "sl"              : sp.get("sl"),
                "tp"              : sp.get("tp"),
                "rr"              : sp.get("rr"),
                "close"           : sp.get("close"),
                "atr"             : sp.get("atr"),
                "signal_notes"    : sp.get("signal_notes", []),
                "signal_warnings" : sp.get("signal_warnings", []),
                "analysis"        : sp.get("analysis", {}),
                "swing_based"     : sp.get("swing_based", False),
                "reason"          : sp.get("reason", ""),
                "timestamp"       : r.get("timestamp"),
            }

    def get_positions(self) -> list:
        """Return posisi MT5 terbuka saat ini."""
        try:
            mt5 = self._get_mt5()
            if not mt5:
                return []
            sym = self.bot.symbol if self.bot else "XAUUSD"
            return mt5.get_all_positions(sym) or []
        except Exception:
            return []

    def get_account(self) -> dict:
        """Return balance, equity, profit dari MT5."""
        try:
            mt5 = self._get_mt5()
            if not mt5:
                return {}
            mt5._refresh_account()
            return mt5.account or {}
        except Exception:
            return {}

    def get_daily_stats(self) -> dict:
        """Return statistik trading hari ini dari journal CSV."""
        try:
            import pandas as _pd
            from app.services.journal import JOURNAL_PATH as _JOURNAL_PATH
            from pathlib import Path as _P
            _jpath = _P(_JOURNAL_PATH)
            if not _jpath.exists():
                return {}
            _df = _pd.read_csv(_jpath, dtype=str)
            _today = __import__('datetime').date.today().isoformat()
            _td = _df[_df.get('entry_time', _df.iloc[:, 0]).str.startswith(_today, na=False)]
            if _td.empty:
                return {"date": _today, "total": 0, "win": 0, "loss": 0,
                        "win_rate": 0.0, "gross_win": 0.0, "gross_loss": 0.0, "net_pnl": 0.0}
            _pnl = _pd.to_numeric(_td['pnl'], errors='coerce').fillna(0)
            _res = _td['result'].str.upper() if 'result' in _td.columns else _pd.Series([])
            _win  = int((_res == 'WIN').sum())
            _loss = int((_res == 'LOSS').sum())
            _total = _win + _loss
            return {
                "date"      : _today,
                "total"     : _total,
                "win"       : _win,
                "loss"      : _loss,
                "win_rate"  : round(_win / _total * 100, 1) if _total else 0.0,
                "gross_win" : round(float(_pnl[_pnl > 0].sum()), 2),
                "gross_loss": round(float(_pnl[_pnl < 0].abs().sum()), 2),
                "net_pnl"   : round(float(_pnl.sum()), 2),
            }
        except Exception:
            return {}

    def get_adaptive_stats(self) -> dict:
        """Return adaptive learning stats."""
        try:
            from app.services.ai.adaptive import get_learner
            _l = get_learner()
            _s = _l.state
            _recents = _s.get("recent_trades", [])[-20:]
            _wins = sum(1 for t in _recents if t.get("result") == "WIN")
            _total = len(_recents)
            return {
                "total_trades"       : _s.get("total_trades", 0),
                "win_rate_20"        : round(_wins / _total * 100, 1) if _total else 0.0,
                "wins_20"            : _wins,
                "total_20"           : _total,
                "performance_mode"   : _l.get_performance_mode(),
                "min_score"          : _l.min_score,
                "trades_since_retrain": _s.get("trades_since_retrain", 0),
                "retrain_every"      : _s.get("retrain_every", 15),
                "last_retrain"       : _s.get("last_retrain", ""),
                "source_hits"        : _s.get("source_hits", {}),
                "recent_trades"      : _recents[-10:],
            }
        except Exception as e:
            return {"error": str(e)}

    def get_signal_analysis_log(self, limit: int = 50) -> list:
        """Return log sinyal + analisis dari signal_analysis.jsonl."""
        try:
            import json as _json
            from pathlib import Path as _P
            _path = _P(__file__).parent.parent / 'logs' / 'signal_analysis.jsonl'
            if not _path.exists():
                return []
            _lines = _path.read_text(encoding='utf-8').splitlines()
            _records = []
            for _line in reversed(_lines):
                if not _line.strip():
                    continue
                try:
                    _records.append(_json.loads(_line))
                except Exception:
                    pass
                if len(_records) >= limit:
                    break
            return _records
        except Exception:
            return []

    def get_circuit_status(self) -> dict:
        """Return status circuit breaker dan consecutive loss block."""
        try:
            import json as _json
            from pathlib import Path as _P
            from datetime import datetime as _dt
            _path = _P(__file__).parent.parent / 'ai' / 'adaptive_state.json'
            if not _path.exists():
                return {}
            _st = _json.loads(_path.read_text(encoding='utf-8'))
            _rts = _st.get('recent_trades', [])
            _today = _dt.now().strftime('%Y-%m-%d')
            _today_losses = [t for t in _rts if t.get('time', '').startswith(_today) and t.get('result') == 'LOSS']

            # Consecutive loss streak
            _streaks = {}
            for _dir in ('BUY', 'SELL'):
                _s = 0
                for _t in reversed(_rts[-10:]):
                    if _t.get('direction') == _dir and _t.get('result') == 'LOSS':
                        _s += 1
                    else:
                        break
                _streaks[_dir] = _s

            # Circuit breaker
            _cb_active = False
            _cb_remaining = 0
            if len(_today_losses) >= 5:
                try:
                    _llt = _dt.strptime(_today_losses[-1].get('time', ''), '%Y-%m-%d %H:%M:%S')
                    _elapsed = (_dt.now() - _llt).total_seconds() / 3600
                    if _elapsed < 2.0:
                        _cb_active = True
                        _cb_remaining = int((2.0 - _elapsed) * 60)
                except Exception:
                    pass

            return {
                "circuit_breaker_active" : _cb_active,
                "circuit_breaker_remaining_min": _cb_remaining,
                "losses_today"           : len(_today_losses),
                "consec_loss_buy"        : _streaks.get('BUY', 0),
                "consec_loss_sell"       : _streaks.get('SELL', 0),
                "buy_blocked"            : _streaks.get('BUY', 0) >= 5,
                "sell_blocked"           : _streaks.get('SELL', 0) >= 5,
            }
        except Exception as e:
            return {"error": str(e)}

    def pause(self) -> bool:
        """Pause bot loop (tidak stop thread, hanya skip analisis)."""
        self._paused = True
        return True

    def resume(self) -> bool:
        """Resume bot loop."""
        self._paused = False
        return True

    def _get_mt5(self):
        """Helper ambil MT5Connector dari executor atau bot."""
        if self.executor:
            return self.executor.mt5
        if self.bot and hasattr(self.bot, 'mt5_conn'):
            return self.bot.mt5_conn
        return None

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
        from app.database.session import AsyncSessionLocal
        from app.database.crud.tx_log import log_event
        async with AsyncSessionLocal() as db:
            await log_event(
                db,
                event_type="BOT_ERROR",
                symbol=self.bot.symbol if self.bot else None,
                message=error_msg[:255],
            )


# Singleton — diakses oleh semua route
bot_service = BotService()
