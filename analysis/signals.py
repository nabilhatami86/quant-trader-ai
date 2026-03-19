import pandas as pd
import numpy as np
from config import *


def _find_swing_levels(df: pd.DataFrame, lookback: int = 100,
                       pivot_window: int = 5) -> tuple[list, list]:
    data  = df.tail(lookback)
    highs = data["High"].values
    lows  = data["Low"].values
    w     = pivot_window

    resistances = []
    supports    = []

    for i in range(w, len(highs) - w):
        if all(highs[i] >= highs[i-j] for j in range(1, w+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, w+1)):
            resistances.append(float(highs[i]))

        if all(lows[i] <= lows[i-j] for j in range(1, w+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, w+1)):
            supports.append(float(lows[i]))

    resistances.sort()
    supports.sort(reverse=True)
    return resistances, supports


def calculate_smart_tp_sl(direction: str, close: float, atr: float,
                           df: pd.DataFrame, final_score: float,
                           decimals: int = 5) -> dict:
    if direction not in ("BUY", "SELL"):
        return {"tp": None, "sl": None, "tp_dist": 0,
                "sl_dist": 0, "rr": 0, "method_tp": "NONE", "method_sl": "NONE"}

    last   = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else last
    buffer = atr * 0.3

    if direction == "BUY":
        candle_low = min(float(last["Low"]), float(prev["Low"]))
        sl         = round(candle_low - buffer, decimals)
        sl_method  = "CandleLow"
        sl_min     = round(close - atr * 0.3, decimals)
        if sl > sl_min:
            sl, sl_method = sl_min, "ATR_min"
    else:
        candle_high = max(float(last["High"]), float(prev["High"]))
        sl          = round(candle_high + buffer, decimals)
        sl_method   = "CandleHigh"
        sl_min      = round(close + atr * 0.3, decimals)
        if sl < sl_min:
            sl, sl_method = sl_min, "ATR_min"

    sl_dist     = abs(close - sl)
    abs_score   = abs(final_score)
    tp_min_dist = max(sl_dist * 1.5, atr * 1.0)
    tp_dist     = 0.0
    tp_method   = "ATR"

    if direction == "BUY":
        resistances, _ = _find_swing_levels(df)
        candidates = [r for r in resistances if r > close + tp_min_dist * 0.8]
        if candidates:
            tp_dist, tp_method = candidates[0] - close, "SwingHigh"

        if tp_dist < tp_min_dist:
            bb_upper = float(df["bb_upper"].iloc[-1]) if "bb_upper" in df.columns else 0
            if bb_upper > close + tp_min_dist:
                tp_dist, tp_method = bb_upper - close, "BB_Upper"

        if tp_dist < tp_min_dist:
            tp_dist   = atr * (1.5 + (abs_score / 10) * 2.0)
            tp_method = "ATR"

        tp = round(close + tp_dist, decimals)

    else:
        _, supports = _find_swing_levels(df)
        candidates = [s for s in supports if s < close - tp_min_dist * 0.8]
        if candidates:
            tp_dist, tp_method = close - candidates[0], "SwingLow"

        if tp_dist < tp_min_dist:
            bb_lower = float(df["bb_lower"].iloc[-1]) if "bb_lower" in df.columns else 0
            if bb_lower < close - tp_min_dist:
                tp_dist, tp_method = close - bb_lower, "BB_Lower"

        if tp_dist < tp_min_dist:
            tp_dist   = atr * (1.5 + (abs_score / 10) * 2.0)
            tp_method = "ATR"

        tp = round(close - tp_dist, decimals)

    rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0

    return {
        "tp":        tp,
        "sl":        sl,
        "tp_dist":   round(tp_dist, decimals),
        "sl_dist":   round(sl_dist, decimals),
        "rr":        rr,
        "method_tp": tp_method,
        "method_sl": sl_method,
    }


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
        return WEIGHTS["macd"], "MACD Bullish"
    elif macd < 0 and hist < 0:
        return -WEIGHTS["macd"], "MACD Bearish"
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
        return WEIGHTS["ema_cross"], "EMA Bullish Cross"
    elif e_fast < e_slow and close < e_trend:
        return -WEIGHTS["ema_cross"], "EMA Bearish Cross"
    elif e_fast > e_slow:
        return WEIGHTS["ema_cross"] * 0.5, "EMA Fast > Slow"
    elif e_fast < e_slow:
        return -WEIGHTS["ema_cross"] * 0.5, "EMA Fast < Slow"
    return 0, "EMA Neutral"


def score_bb(row: pd.Series, close: float) -> tuple[float, str]:
    pct   = row.get("bb_pct", 0.5)
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
        return WEIGHTS["stoch"], "Stoch Oversold + Cross Up"
    elif k >= STOCH_OVERBOUGHT and k < d:
        return -WEIGHTS["stoch"], "Stoch Overbought + Cross Down"
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
        return 0, f"ADX Weak ({adx:.1f})"
    if di_pos > di_neg:
        return WEIGHTS["adx"], f"ADX Uptrend ({adx:.1f})"
    return -WEIGHTS["adx"], f"ADX Downtrend ({adx:.1f})"


def score_candle(row: pd.Series) -> tuple[float, str]:
    pat = row.get("candle_pat", 0)
    if pat == 1:
        return WEIGHTS["candle"], "Bullish Candle Pattern"
    elif pat == -1:
        return -WEIGHTS["candle"], "Bearish Candle Pattern"
    return 0, "No Pattern"


def generate_signal(df: pd.DataFrame, news_bias: dict | None = None) -> dict:
    if df.empty:
        return {"direction": "WAIT", "score": 0, "reasons": []}

    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row.get("atr", close * 0.001))

    scores  = []
    reasons = []

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

    total_score  = sum(scores)
    max_possible = sum(WEIGHTS.values())
    normalized   = (total_score / max_possible) * 10

    news_contribution = 0.0
    if news_bias:
        n_score    = news_bias.get("score", 0.0)
        confidence = news_bias.get("confidence", "LOW")
        weight     = {"HIGH": 0.30, "MEDIUM": 0.20, "LOW": 0.10}.get(confidence, 0.10)
        news_contribution = n_score * weight
        d_word = "↑ BULLISH" if n_score > 0 else "↓ BEARISH" if n_score < 0 else "NEUTRAL"
        reasons.append((news_contribution,
                        f"News Bias {d_word} (score {n_score:+.1f}, conf:{confidence})"))

    final_score = normalized + news_contribution

    if final_score >= MIN_SIGNAL_SCORE:
        direction = "BUY"
    elif final_score <= -MIN_SIGNAL_SCORE:
        direction = "SELL"
    else:
        direction = "WAIT"

    tp_sl = calculate_smart_tp_sl(direction, close, atr, df, final_score)

    return {
        "direction":       direction,
        "score":           round(final_score, 2),
        "score_technical": round(normalized, 2),
        "score_news":      round(news_contribution, 2),
        "close":           close,
        "atr":             round(atr, 5),
        "sl":              tp_sl["sl"],
        "tp":              tp_sl["tp"],
        "tp_dist":         tp_sl["tp_dist"],
        "sl_dist":         tp_sl["sl_dist"],
        "rr_ratio":        tp_sl["rr"],
        "method_tp":       tp_sl["method_tp"],
        "method_sl":       tp_sl["method_sl"],
        "reasons":         reasons,
        "raw_score":       round(total_score, 3),
    }
