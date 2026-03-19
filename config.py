# ─── PAIR & TIMEFRAME ─────────────────────────────────────
SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GOLD":   "GC=F",
    "XAUUSD": "GLD",
}
DEFAULT_SYMBOL    = "EURUSD"
DEFAULT_TIMEFRAME = "1h"

TIMEFRAMES = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}

DATA_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d",
    "1h": "730d", "4h": "730d", "1d": "5y",
}

# ─── INDIKATOR ─────────────────────────────────────────────
RSI_PERIOD       = 14
RSI_OVERBOUGHT   = 70
RSI_OVERSOLD     = 30

MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

EMA_FAST         = 9
EMA_SLOW         = 21
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

# ─── SINYAL ────────────────────────────────────────────────
MIN_SIGNAL_SCORE = 5   # 3-4 = lebih banyak sinyal | 6-7 = lebih selektif

WEIGHTS = {
    "rsi":       2.0,
    "macd":      2.0,
    "ema_cross": 2.0,
    "bb":        1.0,
    "stoch":     1.5,
    "adx":       1.0,
    "candle":    0.5,
}

# ─── MACHINE LEARNING ──────────────────────────────────────
ML_ENABLED     = True
ML_LOOKBACK    = 20
ML_TRAIN_SPLIT = 0.8
ML_MODEL_TYPE  = "ensemble"   # "rf" | "gb" | "et" | "ensemble"

# ─── RISK MANAGEMENT ───────────────────────────────────────
# SL = bawah/atas candle + buffer (0.3× ATR)
# TP = swing high/low terdekat | BB | ATR fallback
SL_BUFFER_MULT  = 0.3
SL_MIN_MULT     = 0.3
TP_MIN_SL_RATIO = 1.5
TP_ATR_FALLBACK = 1.5
RISK_REWARD_RATIO  = 2.0
ATR_MULTIPLIER_SL  = 0.3
ATR_MULTIPLIER_TP  = 2.0

# ─── BACKTEST ──────────────────────────────────────────────
BACKTEST_PERIOD  = "1y"
INITIAL_CAPITAL  = 10000
LOT_SIZE         = 0.01

# ─── DISPLAY ───────────────────────────────────────────────
REFRESH_INTERVAL = 60
SHOW_CHART       = True
