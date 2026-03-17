"""
=============================================================
  TRADING BOT CONFIG - Tuning Parameters
=============================================================
  Edit nilai di bawah untuk tuning strategi trading
=============================================================
"""

# ─── PAIR TRADING ──────────────────────────────────────────
SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GOLD":   "GC=F",       # Gold Futures
    "XAUUSD": "GLD",        # Gold ETF (alternatif)
}

# Pilih pair default: "EURUSD" atau "GOLD"
DEFAULT_SYMBOL = "EURUSD"

# ─── TIMEFRAME ─────────────────────────────────────────────
TIMEFRAMES = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",   # Yahoo = "1h" ambil lebih panjang
    "1d":  "1d",
}
DEFAULT_TIMEFRAME = "1h"

# Berapa banyak data historis (periode yfinance)
DATA_PERIOD = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "1h":  "730d",
    "4h":  "730d",
    "1d":  "5y",
}

# ─── INDIKATOR - RSI ───────────────────────────────────────
RSI_PERIOD       = 14
RSI_OVERBOUGHT   = 70     # Di atas ini = potensi SELL
RSI_OVERSOLD     = 30     # Di bawah ini = potensi BUY

# ─── INDIKATOR - MACD ──────────────────────────────────────
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

# ─── INDIKATOR - EMA ───────────────────────────────────────
EMA_FAST         = 9
EMA_SLOW         = 21
EMA_TREND        = 50
EMA_LONG         = 200

# ─── INDIKATOR - BOLLINGER BANDS ───────────────────────────
BB_PERIOD        = 20
BB_STD           = 2.0    # Deviasi standar

# ─── INDIKATOR - STOCHASTIC ────────────────────────────────
STOCH_K          = 14
STOCH_D          = 3
STOCH_SMOOTH     = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD   = 20

# ─── INDIKATOR - ATR (Average True Range) ──────────────────
ATR_PERIOD       = 14

# ─── INDIKATOR - ADX (Trend Strength) ─────────────────────
ADX_PERIOD       = 14
ADX_TREND_MIN    = 25     # ADX > 25 = trending kuat

# ─── SINYAL & SKOR ─────────────────────────────────────────
# Minimum score (0-10) untuk generate sinyal
MIN_SIGNAL_SCORE = 5      # Turunkan = lebih banyak sinyal, naikkan = lebih selektif

# Bobot setiap indikator dalam scoring (total tidak harus 10)
WEIGHTS = {
    "rsi":        2.0,
    "macd":       2.0,
    "ema_cross":  2.0,
    "bb":         1.0,
    "stoch":      1.5,
    "adx":        1.0,
    "candle":     0.5,
}

# ─── MACHINE LEARNING ──────────────────────────────────────
ML_ENABLED        = True
ML_LOOKBACK       = 20    # Candle sebelumnya sebagai fitur
ML_TRAIN_SPLIT    = 0.8   # 80% train, 20% test
ML_MODEL_TYPE     = "rf"  # "rf" = Random Forest, "gb" = Gradient Boosting, "et" = ExtraTrees, "ensemble" = Voting

# ─── RISK MANAGEMENT ───────────────────────────────────────
RISK_REWARD_RATIO = 2.0   # TP = 2x SL
ATR_MULTIPLIER_SL = 1.5   # SL = 1.5 × ATR
ATR_MULTIPLIER_TP = 3.0   # TP = 3.0 × ATR

# ─── BACKTEST ──────────────────────────────────────────────
BACKTEST_PERIOD   = "1y"  # Periode backtest
INITIAL_CAPITAL   = 10000 # Modal awal backtest ($)
LOT_SIZE          = 0.01  # Ukuran lot (untuk simulasi)

# ─── TAMPILAN ──────────────────────────────────────────────
REFRESH_INTERVAL  = 60    # Detik refresh real-time (default: 60s)
SHOW_CHART        = True
