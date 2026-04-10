"""
app.api.main — FastAPI entrypoint for Trader AI
Jalankan dengan: python -m app.api.main
Atau production: uvicorn app.api.main:app --host 0.0.0.0 --port 8000
"""
import sys, io, os
# Force UTF-8 agar karakter unicode (→ ★ ✓ dll) tidak error di Windows
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import verify_api_key
from app.database.session import create_tables
from app.services.bot_service import bot_service
from app.api.routes import signal, trade, journal, backtest, webhook, txlog
from app.api.routes import positions, adaptive, analysis, stats, bot_control, ws as ws_route, settings as settings_route
from app.api.routes import runner as runner_route

# Setup logging sebelum apapun
setup_logging(debug=settings.DEBUG)

# Restore runtime settings override dari file (jika ada)
try:
    from app.api.routes.settings import _apply_runtime_on_startup
    _apply_runtime_on_startup()
except Exception:
    pass
logger = logging.getLogger("trader_ai")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    logger.info("=" * 55)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 55)

    # Buat tabel Postgres jika belum ada
    await create_tables()
    logger.info("Database tables ready")

    # Terapkan mode real — override config persis seperti main.py --real
    if settings.BOT_REAL_MODE:
        import config as _cfg
        _cfg.MULTI_TP_ENABLED  = True
        _cfg.ADX_TREND_MIN     = _cfg.REAL_ADX_MIN
        _cfg.MIN_SIGNAL_SCORE  = 6
        _cfg.ATR_MULTIPLIER_SL = _cfg.REAL_ATR_SL
        _cfg.ATR_MULTIPLIER_TP = _cfg.REAL_ATR_TP
        logger.info(
            f"REAL MODE aktif — ADX>={_cfg.REAL_ADX_MIN}, "
            f"score>=6, SL={_cfg.REAL_ATR_SL}xATR, TP={_cfg.REAL_ATR_TP}xATR, "
            f"Multi-TP ON, Trail={_cfg.REAL_TRAIL_PIPS}pips"
        )

    # Inisialisasi MT5 (opsional)
    mt5_connector = None
    executor = None
    if settings.BOT_USE_MT5:
        try:
            from app.engine.broker.mt5_connector import MT5Connector, SignalExecutor
            mt5_connector = MT5Connector()
            if mt5_connector.connect(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
            ):
                import config as _cfg
                if settings.BOT_REAL_MODE:
                    trail  = _cfg.REAL_TRAIL_PIPS
                    orders = _cfg.REAL_MAX_ORDERS
                    lot    = None   # auto-lot dari saldo MT5
                else:
                    trail  = settings.BOT_TRAIL_PIPS
                    orders = settings.BOT_ORDERS
                    lot    = settings.BOT_LOT if settings.BOT_LOT > 0 else None

                executor = SignalExecutor(
                    connector=mt5_connector,
                    symbol_key=settings.BOT_SYMBOL,
                    trailing_pips=trail,
                    dca_minutes=settings.BOT_DCA_MINUTES,
                    bulk_orders=orders,
                    fixed_lot=lot,
                    strict_mode=settings.BOT_MICRO_MODE,
                    risk_per_trade=settings.BOT_RISK_USD,
                    real_mode=settings.BOT_REAL_MODE,
                )
                logger.info(
                    f"MT5 connected — auto-order AKTIF "
                    f"({'REAL MODE' if settings.BOT_REAL_MODE else 'NORMAL'}, "
                    f"lot={'auto' if lot is None else lot}, orders={orders})"
                )
            else:
                logger.warning("MT5 gagal connect — bot jalan tanpa auto-order")
                mt5_connector = None
        except Exception as exc:
            logger.warning(f"MT5 init error: {exc} — bot jalan tanpa auto-order")
            mt5_connector = None

    # Inisialisasi bot (hanya jika AI enabled)
    if settings.BOT_AI_ENABLED:
        bot_service.initialize(
            symbol=settings.BOT_SYMBOL,
            timeframe=settings.BOT_TIMEFRAME,
            use_news=settings.BOT_USE_NEWS,
            mt5_connector=mt5_connector,
            executor=executor,
        )

        # Simpan referensi main event loop agar background thread bisa save ke DB
        import asyncio
        bot_service._event_loop = asyncio.get_running_loop()

        # Jalankan bot di background thread
        bot_service.start()
        logger.info(f"Bot started -> {settings.BOT_SYMBOL} {settings.BOT_TIMEFRAME}")
    else:
        logger.info(
            "BOT_AI_ENABLED=false — Backend API berjalan tanpa AI/ML. "
            "Endpoint /signal/analyze tidak aktif. DB, journal, MT5 tetap OK."
        )

    # ── Status panel: posisi terbuka saat startup ──────────────────────
    _G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"
    _C = "\033[96m"; _B = "\033[1m";  _D = "\033[2m"; _X = "\033[0m"
    _sep = "-" * 55
    _sep2 = "=" * 55
    print(f"\n{_B}{_sep2}{_X}")
    print(f"  {_B}{_C}TRADER AI -- STATUS STARTUP{_X}")
    print(f"{_B}{_sep2}{_X}")
    _bot_status = f"{_G}{_B}RUNNING{_X}" if settings.BOT_AI_ENABLED else f"{_Y}API ONLY{_X}"
    print(f"  Bot      : {_bot_status}  ({settings.BOT_SYMBOL} {settings.BOT_TIMEFRAME})")
    _mode_str = f"{_R}REAL MODE{_X}" if settings.BOT_REAL_MODE else f"{_Y}DEMO MODE{_X}"
    print(f"  Mode     : {_B}{_mode_str}")
    print(f"  API      : {_G}http://0.0.0.0:8000{_X}")
    print(_sep)

    if mt5_connector and mt5_connector.connected:
        try:
            mt5_connector._refresh_account()
            _bal = mt5_connector.account.get("balance", 0)
            _eq  = mt5_connector.account.get("equity", 0)
            _flt = mt5_connector.account.get("profit", 0)
            _pc  = _G if _flt >= 0 else _R
            print(f"  MT5      : {_G}Connected{_X}  |  Balance: {_B}${_bal:,.2f}{_X}  Equity: ${_eq:,.2f}")
            print(f"  Float P&L: {_pc}{_B}${_flt:+,.2f}{_X}")
            print(_sep)

            _positions = mt5_connector.get_all_positions(settings.BOT_SYMBOL)
            if _positions:
                print(f"  {_Y}{_B}POSISI TERBUKA  ({len(_positions)} posisi){_X}")
                for _p in _positions:
                    _dc  = _G if _p["direction"] == "BUY" else _R
                    _pc2 = _G if _p["profit"] >= 0 else _R
                    _sl  = f"SL:{_p['sl']:.2f}" if _p.get("sl") else "SL:--"
                    _tp  = f"TP:{_p['tp']:.2f}" if _p.get("tp") else "TP:--"
                    print(f"    #{_p['ticket']}  {_dc}{_B}{_p['direction']}{_X}  "
                          f"{_p['lot']}lot @ {_p['open_price']:.2f}  "
                          f"{_sl}  {_tp}  P&L: {_pc2}{_B}${_p['profit']:+.2f}{_X}")
            else:
                print(f"  Posisi   : {_D}Belum ada posisi terbuka{_X}")
        except Exception:
            print(f"  MT5      : {_G}Connected{_X}")
    else:
        print(f"  MT5      : {_R}Tidak terhubung{_X}")

    print(f"{_B}{_sep2}{_X}\n")

    # Catat BOT_START ke tx_log
    try:
        from app.database.session import AsyncSessionLocal
        from app.database.crud.tx_log import log_event
        async with AsyncSessionLocal() as db:
            await log_event(
                db,
                event_type="BOT_START",
                symbol=settings.BOT_SYMBOL,
                timeframe=settings.BOT_TIMEFRAME,
                message=f"API v{settings.APP_VERSION} started",
            )
    except Exception:
        pass

    yield

    # ── SHUTDOWN ──
    bot_service.stop()

    # Putuskan koneksi MT5 jika aktif
    if mt5_connector and mt5_connector.connected:
        try:
            mt5_connector.disconnect()
            logger.info("MT5 disconnected")
        except Exception:
            pass

    try:
        from app.database.session import AsyncSessionLocal
        from app.database.crud.tx_log import log_event
        async with AsyncSessionLocal() as db:
            await log_event(
                db,
                event_type="BOT_STOP",
                symbol=settings.BOT_SYMBOL,
                message="API stopped gracefully",
            )
    except Exception:
        pass

    logger.info("Trader AI API stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="REST API untuk Trading Bot XAUUSD/EURUSD dengan ML + MT5",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],  # global auth (jika API_KEY di .env diset)
)

# CORS — izinkan semua origin (sesuaikan untuk production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(signal.router)
app.include_router(trade.router)
app.include_router(journal.router)
app.include_router(backtest.router)
app.include_router(webhook.router)
app.include_router(txlog.router)
# ── New routes ────────────────────────────────────────────────────────────────
app.include_router(positions.router)
app.include_router(adaptive.router)
app.include_router(analysis.router)
app.include_router(stats.router)
app.include_router(bot_control.router)
app.include_router(ws_route.router)   # WebSocket /ws/live
app.include_router(settings_route.router)  # /settings
app.include_router(runner_route.router)    # /runner — subprocess main.py


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Info API")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["Health"], summary="Status bot dan API")
async def health():
    return {
        "api": "ok",
        "bot": bot_service.get_status(),
    }


# ── Run langsung ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
        workers=1,  # 1 worker — bot service pakai thread, tidak bisa multi-worker
    )
