# ==============================================================
#  UMUM — berlaku untuk semua mode kecuali di-override
# ==============================================================

SYMBOLS = {
    "GOLD":     "GC=F",
    "XAUUSD":   "GC=F",
    "XAUEUR":   "GLD",
    "EURUSD":   "EURUSD=X",
    "GBPUSD":   "GBPUSD=X",
    "USDJPY":   "JPY=X",
    "DXY":      "DX-Y.NYB",
    "BTCUSD":   "BTC-USD",
}
DEFAULT_SYMBOL    = "XAUUSD"
DEFAULT_TIMEFRAME = "1h"

SYMBOL_DESC = {
    "GOLD":   "Gold Futures (CME)",
    "XAUUSD": "Gold / US Dollar",
    "XAUEUR": "Gold / Euro",
    "EURUSD": "Euro / US Dollar",
    "GBPUSD": "British Pound / US Dollar",
    "USDJPY": "US Dollar / Japanese Yen",
    "DXY":    "US Dollar Index",
    "BTCUSD": "Bitcoin / US Dollar",
}

TIMEFRAMES  = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
DATA_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d",
    "1h": "730d", "4h": "730d", "1d": "5y",
}

# Indikator
RSI_PERIOD       = 14
RSI_OVERBOUGHT   = 70
RSI_OVERSOLD     = 30

MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

EMA_FAST         = 9
EMA_SLOW         = 20
EMA_TREND        = 50
EMA_LONG         = 200

BB_PERIOD        = 20
BB_STD           = 2.0

STOCH_K          = 14
STOCH_D          = 3
STOCH_SMOOTH     = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD   = 20

ATR_PERIOD       = 14
ADX_PERIOD       = 14
ADX_TREND_MIN    = 25

# Sinyal
MIN_SIGNAL_SCORE = 5
WEIGHTS = {
    "rsi":       2.0,
    "macd":      2.0,
    "ema_cross": 2.0,
    "bb":        1.0,
    "stoch":     1.5,
    "adx":       1.0,
    "candle":    0.5,
}

# ML
ML_ENABLED     = True
ML_LOOKBACK    = 20
ML_TRAIN_SPLIT = 0.8
ML_MODEL_TYPE  = "ensemble"

# SL / TP umum  —  RR 1:10
AUTO_TP_SL        = False  # OFF = pakai ATR multiplier di bawah
ATR_MULTIPLIER_SL = 1.0    # SL = 1x ATR
ATR_MULTIPLIER_TP = 10.0   # TP = 10x ATR  →  RR 1:10
MIN_RR_RATIO      = 5.0    # tolak sinyal jika RR < nilai ini

# Breakeven
BREAKEVEN_ENABLED = True
BREAKEVEN_TRIGGER = 1.0
BREAKEVEN_BUFFER  = 0.1

# SL Guard
SL_DANGER_PCT   = 0.30
SL_CRITICAL_PCT = 0.10

# Multi-TP / Partial Close
MULTI_TP_ENABLED = False
TP1_RATIO        = 0.5
TP1_CLOSE_PCT    = 50.0

# Lain-lain
BACKTEST_PERIOD  = "1y"
INITIAL_CAPITAL  = 10000
REFRESH_INTERVAL = 60
SHOW_CHART       = True


# ==============================================================
#  MODE MIKRO  —  aktif dengan flag --micro
#  Untuk akun demo / latihan / modal < Rp 1 juta
# ==============================================================

MICRO_LOT        = 0.01
MICRO_MAX_ORDERS = 1
MICRO_MIN_SCORE  = 7
MICRO_ML_CONF    = 75
MICRO_ADX_MIN    = 30
MICRO_RISK_PCT   = 0.5


# ==============================================================
#  MODE REAL  —  aktif dengan flag --real
#  Untuk akun real ~$60 USD, target profit $15-20 per trade
# ==============================================================

REAL_LOT        = 0.3   # lot per order  (ubah di sini jika mau lebih besar)
REAL_MAX_ORDERS = 1      # maksimal 1 posisi aktif, tidak tumpuk
REAL_ML_CONF    = 70     # ML wajib setuju minimal 70%
REAL_ADX_MIN    = 28     # hanya masuk saat trending
REAL_TRAIL_PIPS = 15.0   # trailing stop 15 pips
REAL_RISK_PCT   = 8.0    # maks 8% balance per trade


# RR 1:10 — SL kecil, TP sangat jauh
# 1 trade menang = tutup 10 trade rugi
REAL_ATR_SL     = 1.0    # SL = 1x ATR
REAL_ATR_TP     = 10.0   # TP = 10x ATR
