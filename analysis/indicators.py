import pandas as pd
import numpy as np
from config import *


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(close: pd.Series) -> pd.DataFrame:
    ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram  = macd_line - signal
    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal,
        "histogram": histogram,
    })


def calculate_bollinger_bands(close: pd.Series) -> pd.DataFrame:
    sma    = close.rolling(BB_PERIOD).mean()
    std    = close.rolling(BB_PERIOD).std()
    upper  = sma + BB_STD * std
    lower  = sma - BB_STD * std
    bw     = (upper - lower) / sma
    pct_b  = (close - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   sma,
        "bb_lower": lower,
        "bb_bw":    bw,
        "bb_pct":   pct_b,
    })


def calculate_stochastic(high, low, close) -> pd.DataFrame:
    low_min  = low.rolling(STOCH_K).min()
    high_max = high.rolling(STOCH_K).max()
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k.fillna(50)
    d = k.rolling(STOCH_D).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def calculate_atr(high, low, close, period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calculate_adx(high, low, close, period: int = ADX_PERIOD) -> pd.DataFrame:
    tr     = calculate_atr(high, low, close, period)
    dm_pos = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0.0)
    dm_neg = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0.0)
    di_pos = 100 * dm_pos.ewm(span=period, adjust=False).mean() / tr.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(span=period, adjust=False).mean() / tr.replace(0, np.nan)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    adx    = dx.ewm(span=period, adjust=False).mean()
    return pd.DataFrame({"adx": adx, "di_pos": di_pos, "di_neg": di_neg})


def calculate_emas(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        f"ema_{EMA_FAST}":  close.ewm(span=EMA_FAST,  adjust=False).mean(),
        f"ema_{EMA_SLOW}":  close.ewm(span=EMA_SLOW,  adjust=False).mean(),
        f"ema_{EMA_TREND}": close.ewm(span=EMA_TREND, adjust=False).mean(),
        f"ema_{EMA_LONG}":  close.ewm(span=EMA_LONG,  adjust=False).mean(),
    })


def detect_candle_pattern(open_, high, low, close) -> pd.Series:
    body       = close - open_
    range_     = (high - low).replace(0, np.nan)
    body_ratio = body.abs() / range_

    doji           = body_ratio < 0.1
    hammer         = (body > 0) & ((open_ - low) / range_ > 0.6)
    shooting       = (body < 0) & ((high - open_) / range_ > 0.6)
    bullish_engulf = (body > 0) & (body.shift() < 0) & (close > open_.shift()) & (open_ < close.shift())
    bearish_engulf = (body < 0) & (body.shift() > 0) & (close < open_.shift()) & (open_ > close.shift())

    pattern = pd.Series(0, index=close.index)
    pattern = pattern.where(~(hammer | bullish_engulf), 1)
    pattern = pattern.where(~(shooting | bearish_engulf), -1)
    pattern = pattern.where(~doji, 0)
    return pattern


def detect_candle_name(open_, high, low, close) -> pd.Series:
    body       = close - open_
    range_     = (high - low).replace(0, np.nan)
    body_ratio = body.abs() / range_

    upper_shadow = (high - close.where(close > open_, open_)) / range_
    lower_shadow = (close.where(close < open_, open_) - low) / range_

    doji          = body_ratio < 0.1
    spinning_top  = (body_ratio >= 0.1) & (body_ratio < 0.3)
    marubozu_bull = (body > 0) & (body_ratio > 0.9)
    marubozu_bear = (body < 0) & (body_ratio > 0.9)

    hammer        = (body > 0) & (lower_shadow > 0.6) & (upper_shadow < 0.1)
    inv_hammer    = (body > 0) & (upper_shadow > 0.6) & (lower_shadow < 0.1)
    hanging_man   = (body < 0) & (lower_shadow > 0.6) & (upper_shadow < 0.1)
    shooting_star = (body < 0) & (upper_shadow > 0.6) & (lower_shadow < 0.1)

    bullish_engulf = (body > 0) & (body.shift() < 0) & (close > open_.shift()) & (open_ < close.shift())
    bearish_engulf = (body < 0) & (body.shift() > 0) & (close < open_.shift()) & (open_ > close.shift())

    tweezer_bot = (body.abs() / range_ > 0.3) & (low.round(2) == low.shift().round(2)) & (body > 0)
    tweezer_top = (body.abs() / range_ > 0.3) & (high.round(2) == high.shift().round(2)) & (body < 0)

    name = pd.Series("None", index=close.index)
    name = name.where(~spinning_top,   "Spinning Top")
    name = name.where(~doji,           "Doji")
    name = name.where(~hanging_man,    "Hanging Man ↓")
    name = name.where(~inv_hammer,     "Inverted Hammer")
    name = name.where(~shooting_star,  "Shooting Star ↓")
    name = name.where(~hammer,         "Hammer ↑")
    name = name.where(~marubozu_bear,  "Marubozu Bear ↓")
    name = name.where(~marubozu_bull,  "Marubozu Bull ↑")
    name = name.where(~tweezer_top,    "Tweezer Top ↓")
    name = name.where(~tweezer_bot,    "Tweezer Bottom ↑")
    name = name.where(~bearish_engulf, "Bearish Engulfing ↓")
    name = name.where(~bullish_engulf, "Bullish Engulfing ↑")
    return name


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    open_ = df["Open"]

    df["rsi"]         = calculate_rsi(close)
    macd_df           = calculate_macd(close)
    df                = pd.concat([df, macd_df], axis=1)
    bb_df             = calculate_bollinger_bands(close)
    df                = pd.concat([df, bb_df], axis=1)
    stoch_df          = calculate_stochastic(high, low, close)
    df                = pd.concat([df, stoch_df], axis=1)
    df["atr"]         = calculate_atr(high, low, close)
    adx_df            = calculate_adx(high, low, close)
    df                = pd.concat([df, adx_df], axis=1)
    ema_df            = calculate_emas(close)
    df                = pd.concat([df, ema_df], axis=1)
    df["candle_pat"]  = detect_candle_pattern(open_, high, low, close)
    df["candle_name"] = detect_candle_name(open_, high, low, close)

    df["price_change"]  = close.pct_change()
    df["volatility"]    = close.rolling(20).std() / close.rolling(20).mean()
    df["momentum"]      = close - close.shift(10)
    df["higher_high"]   = (high > high.shift(1)).astype(int)
    df["lower_low"]     = (low < low.shift(1)).astype(int)

    return df.dropna()
