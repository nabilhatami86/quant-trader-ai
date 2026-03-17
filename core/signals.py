"""
Signal Engine - Scoring & Rule-Based Signal Generation
"""
import pandas as pd
from config import *


def score_rsi(row: pd.Series) -> tuple[float, str]:
    rsi = row.get("rsi", 50)
    if rsi <= RSI_OVERSOLD:
        return WEIGHTS["rsi"], "RSI Oversold (BUY)"
    elif rsi >= RSI_OVERBOUGHT:
        return -WEIGHTS["rsi"], "RSI Overbought (SELL)"
    elif rsi < 45:
        return WEIGHTS["rsi"] * 0.5, "RSI Bullish"
    elif rsi > 55:
        return -WEIGHTS["rsi"] * 0.5, "RSI Bearish"
    return 0, "RSI Neutral"


def score_macd(row: pd.Series) -> tuple[float, str]:
    macd = row.get("macd", 0)
    hist = row.get("histogram", 0)
    if macd > 0 and hist > 0:
        return WEIGHTS["macd"], "MACD Bullish (above signal)"
    elif macd < 0 and hist < 0:
        return -WEIGHTS["macd"], "MACD Bearish (below signal)"
    elif hist > 0:
        return WEIGHTS["macd"] * 0.5, "MACD Histogram Positive"
    elif hist < 0:
        return -WEIGHTS["macd"] * 0.5, "MACD Histogram Negative"
    return 0, "MACD Neutral"


def score_ema(row: pd.Series, close: float) -> tuple[float, str]:
    e_fast  = row.get(f"ema_{EMA_FAST}", close)
    e_slow  = row.get(f"ema_{EMA_SLOW}", close)
    e_trend = row.get(f"ema_{EMA_TREND}", close)

    if e_fast > e_slow and close > e_trend:
        return WEIGHTS["ema_cross"], "EMA Bullish Cross (BUY)"
    elif e_fast < e_slow and close < e_trend:
        return -WEIGHTS["ema_cross"], "EMA Bearish Cross (SELL)"
    elif e_fast > e_slow:
        return WEIGHTS["ema_cross"] * 0.5, "EMA Fast > Slow"
    elif e_fast < e_slow:
        return -WEIGHTS["ema_cross"] * 0.5, "EMA Fast < Slow"
    return 0, "EMA Neutral"


def score_bb(row: pd.Series, close: float) -> tuple[float, str]:
    pct  = row.get("bb_pct", 0.5)
    upper = row.get("bb_upper", close)
    lower = row.get("bb_lower", close)

    if close <= lower:
        return WEIGHTS["bb"], "Price at BB Lower (BUY)"
    elif close >= upper:
        return -WEIGHTS["bb"], "Price at BB Upper (SELL)"
    elif pct < 0.2:
        return WEIGHTS["bb"] * 0.5, "Near BB Lower"
    elif pct > 0.8:
        return -WEIGHTS["bb"] * 0.5, "Near BB Upper"
    return 0, "BB Neutral"


def score_stoch(row: pd.Series) -> tuple[float, str]:
    k = row.get("stoch_k", 50)
    d = row.get("stoch_d", 50)
    if k <= STOCH_OVERSOLD and k > d:
        return WEIGHTS["stoch"], "Stoch Oversold + Cross Up (BUY)"
    elif k >= STOCH_OVERBOUGHT and k < d:
        return -WEIGHTS["stoch"], "Stoch Overbought + Cross Down (SELL)"
    elif k < STOCH_OVERSOLD:
        return WEIGHTS["stoch"] * 0.5, "Stoch Oversold"
    elif k > STOCH_OVERBOUGHT:
        return -WEIGHTS["stoch"] * 0.5, "Stoch Overbought"
    return 0, "Stoch Neutral"


def score_adx(row: pd.Series) -> tuple[float, str]:
    adx    = row.get("adx", 20)
    di_pos = row.get("di_pos", 25)
    di_neg = row.get("di_neg", 25)

    if adx < ADX_TREND_MIN:
        return 0, f"ADX Weak Trend ({adx:.1f})"
    if di_pos > di_neg:
        return WEIGHTS["adx"], f"ADX Strong Uptrend ({adx:.1f})"
    else:
        return -WEIGHTS["adx"], f"ADX Strong Downtrend ({adx:.1f})"


def score_candle(row: pd.Series) -> tuple[float, str]:
    pat = row.get("candle_pat", 0)
    if pat == 1:
        return WEIGHTS["candle"], "Bullish Candle Pattern"
    elif pat == -1:
        return -WEIGHTS["candle"], "Bearish Candle Pattern"
    return 0, "No Pattern"


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Generate sinyal BUY/SELL dari baris terakhir DataFrame.
    Returns dict dengan score, direction, sl, tp, dan reasons.
    """
    if df.empty:
        return {"direction": "WAIT", "score": 0, "reasons": []}

    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row.get("atr", close * 0.001))

    scores   = []
    reasons  = []

    for fn in [score_rsi, score_macd, score_stoch, score_adx, score_candle]:
        s, r = fn(row)
        scores.append(s)
        reasons.append((s, r))

    s, r = score_ema(row, close)
    scores.append(s)
    reasons.append((s, r))

    s, r = score_bb(row, close)
    scores.append(s)
    reasons.append((s, r))

    total_score = sum(scores)
    max_possible = sum(WEIGHTS.values())

    # Normalize ke -10 .. +10
    normalized = (total_score / max_possible) * 10

    # Tentukan arah
    if normalized >= MIN_SIGNAL_SCORE:
        direction = "BUY"
    elif normalized <= -MIN_SIGNAL_SCORE:
        direction = "SELL"
    else:
        direction = "WAIT"

    # Hitung SL & TP
    sl_dist = atr * ATR_MULTIPLIER_SL
    tp_dist = atr * ATR_MULTIPLIER_TP
    if direction == "BUY":
        sl = round(close - sl_dist, 5)
        tp = round(close + tp_dist, 5)
    elif direction == "SELL":
        sl = round(close + sl_dist, 5)
        tp = round(close - tp_dist, 5)
    else:
        sl = tp = None

    return {
        "direction":  direction,
        "score":      round(normalized, 2),
        "close":      close,
        "atr":        round(atr, 5),
        "sl":         sl,
        "tp":         tp,
        "rr_ratio":   RISK_REWARD_RATIO,
        "reasons":    reasons,
        "raw_score":  round(total_score, 3),
    }
