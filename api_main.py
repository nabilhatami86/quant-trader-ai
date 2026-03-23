"""
api_main.py — Entry point FastAPI Trader AI
Jalankan dengan: python api_main.py
Atau production: uvicorn api_main:app --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import setup_logging
from core.security import verify_api_key
from db.database import create_tables
from services.bot_service import bot_service
from api.routes import signal, trade, journal, backtest, webhook, txlog

# Setup logging sebelum apapun
setup_logging(debug=settings.DEBUG)
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
            from broker.mt5_connector import MT5Connector, SignalExecutor
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

    # Inisialisasi bot
    bot_service.initialize(
        symbol=settings.BOT_SYMBOL,
        timeframe=settings.BOT_TIMEFRAME,
        use_news=settings.BOT_USE_NEWS,
        mt5_connector=mt5_connector,
        executor=executor,
    )

    # Simpan referensi main event loop agar background thread bisa save ke DB
    import asyncio
    bot_service._event_loop = asyncio.get_event_loop()

    # Jalankan bot di background thread
    bot_service.start()
    logger.info(f"Bot started -> {settings.BOT_SYMBOL} {settings.BOT_TIMEFRAME}")

    # Catat BOT_START ke tx_log
    try:
        from db.database import AsyncSessionLocal
        from db.crud.tx_log import log_event
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
        from db.database import AsyncSessionLocal
        from db.crud.tx_log import log_event
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
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
        workers=1,  # 1 worker — bot service pakai thread, tidak bisa multi-worker
    )
