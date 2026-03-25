from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from db.database import Base


class Candle(Base):
    """OHLCV candle data — menggantikan CSV di data/history/."""

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
        Index("ix_candle_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )


class Signal(Base):
    """Log setiap sinyal yang dihasilkan bot."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Signal utama
    direction: Mapped[str] = mapped_column(String(10))   # BUY / SELL / WAIT
    score: Mapped[float] = mapped_column(Float, default=0.0)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    rr_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_price: Mapped[float] = mapped_column(Float, default=0.0)

    # Indikator snapshot
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ML
    ml_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_trained_symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ml_symbol_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Eksekusi
    exec_direction: Mapped[str] = mapped_column(String(10), default="WAIT")
    exec_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    news_risk: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Score breakdown
    score_technical: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_smc: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Raw full result (untuk audit)
    raw_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Trade(Base):
    """History order / trade — menggantikan logs/journal_*.csv."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), default="")
    direction: Mapped[str] = mapped_column(String(10))   # BUY / SELL
    entry: Mapped[float] = mapped_column(Float, default=0.0)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot: Mapped[float] = mapped_column(Float, default=0.01)
    ticket: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Hasil
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[str] = mapped_column(String(10), default="OPEN")   # WIN/LOSS/OPEN

    # Waktu
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Metadata
    source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # rule-based/ML/force/csv_import


class CandleLog(Base):
    """
    Candle + indikator + sinyal per bar — dari logs/candles_*.csv.
    Lebih kaya dari Candle biasa karena sudah include RSI, MACD, pattern, dsb.
    """

    __tablename__ = "candle_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # OHLC
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Candle info
    candle_type: Mapped[str | None] = mapped_column(String(10), nullable=True)   # BULLISH/BEARISH
    body: Mapped[float | None] = mapped_column(Float, nullable=True)
    wick_up: Mapped[float | None] = mapped_column(Float, nullable=True)
    wick_down: Mapped[float | None] = mapped_column(Float, nullable=True)
    pattern: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Indikator
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema20: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema50: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd: Mapped[float | None] = mapped_column(Float, nullable=True)
    histogram: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Sinyal saat itu
    signal_dir: Mapped[str | None] = mapped_column(String(10), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Volume indicators
    obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    williams_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    cci: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Smart Money Concepts
    fvg_bull: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fvg_bear: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ob_bull: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ob_bear: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bos_bull: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bos_bear: Mapped[int | None] = mapped_column(Integer, nullable=True)
    choch_bull: Mapped[int | None] = mapped_column(Integer, nullable=True)
    choch_bear: Mapped[int | None] = mapped_column(Integer, nullable=True)
    liq_bull_sweep: Mapped[int | None] = mapped_column(Integer, nullable=True)
    liq_bear_sweep: Mapped[int | None] = mapped_column(Integer, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(10), nullable=True)
    candle_ex: Mapped[int | None] = mapped_column(Integer, nullable=True)

    logged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Outcome — diisi setelah lookahead candle berlalu
    outcome:     Mapped[str | None]   = mapped_column(String(5),  nullable=True)  # WIN/LOSS/FLAT
    outcome_pct: Mapped[float | None] = mapped_column(Float,      nullable=True)  # % perubahan harga

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle_log"),
        Index("ix_candle_log_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )


class TxLog(Base):
    """
    Transaction log — catat SEMUA event: sinyal, order masuk, order tutup, error, dsb.
    Tidak pernah dihapus, hanya append.
    """

    __tablename__ = "tx_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Tipe event
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # Contoh: SIGNAL_BUY, SIGNAL_SELL, SIGNAL_WAIT, ORDER_OPEN, ORDER_CLOSE,
    #         ORDER_SKIP, BOT_START, BOT_STOP, BOT_ERROR, CYCLE_DONE, MIGRATE

    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(5), nullable=True)
    direction: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Harga & order
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot: Mapped[float | None] = mapped_column(Float, nullable=True)
    ticket: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    pips: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Detail tambahan (JSON bebas)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Pesan singkat (untuk log viewer)
    message: Mapped[str | None] = mapped_column(String(255), nullable=True)


class MLResult(Base):
    """Log hasil training ML per sesi."""

    __tablename__ = "ml_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    model_type: Mapped[str] = mapped_column(String(20), default="ensemble")

    # Metrik
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    conf_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    precision_buy: Mapped[float] = mapped_column(Float, default=0.0)
    recall_buy: Mapped[float] = mapped_column(Float, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Info dataset
    n_features: Mapped[int] = mapped_column(Integer, default=0)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    n_sideways_removed: Mapped[int] = mapped_column(Integer, default=0)

    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
