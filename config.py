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

TIMEFRAMES = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}

DATA_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d",
    "1h": "730d", "4h": "730d", "1d": "5y",
}

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

MIN_SIGNAL_SCORE  = 5
MICRO_MIN_SCORE   = 7

WEIGHTS = {
    "rsi":       2.0,
    "macd":      2.0,
    "ema_cross": 2.0,
    "bb":        1.0,
    "stoch":     1.5,
    "adx":       1.0,
    "candle":    0.5,
}

ML_ENABLED     = True
ML_LOOKBACK    = 20
ML_TRAIN_SPLIT = 0.8
ML_MODEL_TYPE  = "ensemble"

ATR_MULTIPLIER_SL  = 1.5
ATR_MULTIPLIER_TP  = 2.0
RISK_REWARD_RATIO  = 2.0
TP_MIN_SL_RATIO    = 1.5

BREAKEVEN_ENABLED  = True
BREAKEVEN_TRIGGER  = 1.0
BREAKEVEN_BUFFER   = 0.1

SL_DANGER_PCT      = 0.30
SL_CRITICAL_PCT    = 0.10

MULTI_TP_ENABLED   = False
TP1_RATIO          = 0.5
TP1_CLOSE_PCT      = 50.0

MICRO_LOT         = 0.01
MICRO_MAX_ORDERS  = 1
MICRO_ML_CONF     = 75
MICRO_ADX_MIN     = 30
MICRO_RISK_PCT    = 0.5

BACKTEST_PERIOD  = "1y"
INITIAL_CAPITAL  = 10000
LOT_SIZE         = 0.01

REFRESH_INTERVAL = 60
SHOW_CHART       = True
