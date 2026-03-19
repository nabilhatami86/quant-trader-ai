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

    sl_dist   = atr * ATR_MULTIPLIER_SL
    tp_dist   = atr * ATR_MULTIPLIER_TP
    sl_method = "ATR"
    tp_method = "ATR"

    if direction == "BUY":
        sl = round(close - sl_dist, decimals)
        tp = round(close + tp_dist, decimals)

        resistances, _ = _find_swing_levels(df)
        swing_candidates = [r for r in resistances if r > close + tp_dist]
        if swing_candidates:
            swing_tp = swing_candidates[0]
            tp, tp_method = round(swing_tp, decimals), "SwingHigh"
            tp_dist = swing_tp - close

    else:
        sl = round(close + sl_dist, decimals)
        tp = round(close - tp_dist, decimals)

        _, supports = _find_swing_levels(df)
        swing_candidates = [s for s in supports if s < close - tp_dist]
        if swing_candidates:
            swing_tp = swing_candidates[0]
            tp, tp_method = round(swing_tp, decimals), "SwingLow"
            tp_dist = close - swing_tp

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


def _read_candle_trend(df: pd.DataFrame, lookback: int = 5) -> tuple[str, str]:
    if len(df) < lookback:
        return "MIXED", "Data tidak cukup"

    recent  = df.tail(lookback)
    bullish = int(((recent["Close"] - recent["Open"]) > 0).sum())
    bearish = lookback - bullish
    ratio   = bullish / lookback

    if ratio >= 0.7:
        return "BULLISH", f"{bullish}/{lookback} candle terakhir BULLISH ↑"
    elif ratio <= 0.3:
        return "BEARISH", f"{bearish}/{lookback} candle terakhir BEARISH ↓"
    else:
        return "MIXED", f"Candle MIXED ({bullish} bull / {bearish} bear)"


def _score_indicators(row: pd.Series, close: float,
                      news_bias: dict | None) -> tuple:
    ema20 = row.get(f"ema_{EMA_SLOW}", close)
    ema50 = row.get(f"ema_{EMA_TREND}", close)
    rsi   = row.get("rsi", 50)
    reasons = []

    ema_bull = ema20 > ema50
    ema_bear = ema20 < ema50

    if ema_bull:
        reasons.append((+1, f"EMA20 > EMA50 ({ema20:.2f} > {ema50:.2f}) ↑"))
    elif ema_bear:
        reasons.append((-1, f"EMA20 < EMA50 ({ema20:.2f} < {ema50:.2f}) ↓"))
    else:
        reasons.append((0, "EMA20 = EMA50 (flat)"))

    if rsi < RSI_OVERBOUGHT:
        reasons.append((+0.5, f"RSI {rsi:.1f} < {RSI_OVERBOUGHT} (OK BUY)"))
    else:
        reasons.append((-0.5, f"RSI {rsi:.1f} ≥ {RSI_OVERBOUGHT} (overbought)"))

    if rsi > RSI_OVERSOLD:
        reasons.append((+0.5, f"RSI {rsi:.1f} > {RSI_OVERSOLD} (OK SELL)"))
    else:
        reasons.append((-0.5, f"RSI {rsi:.1f} ≤ {RSI_OVERSOLD} (oversold)"))

    for fn in [score_macd, score_stoch, score_adx, score_candle]:
        s, r = fn(row)
        if s != 0:
            reasons.append((s, r))

    news_contribution = 0.0
    if news_bias:
        n_score    = news_bias.get("score", 0.0)
        confidence = news_bias.get("confidence", "LOW")
        weight     = {"HIGH": 0.30, "MEDIUM": 0.20, "LOW": 0.10}.get(confidence, 0.10)
        news_contribution = n_score * weight
        d_word = "↑ BULLISH" if n_score > 0 else "↓ BEARISH" if n_score < 0 else "NEUTRAL"
        reasons.append((news_contribution,
                        f"News Bias {d_word} (score {n_score:+.1f}, conf:{confidence})"))

    adx      = row.get("adx", 0)
    trending = adx >= ADX_TREND_MIN
    if not trending:
        reasons.append((0, f"ADX {adx:.1f} < {ADX_TREND_MIN} — sideways, skip"))

    return ema_bull, ema_bear, rsi, trending, reasons, news_contribution


def _make_decision(df: pd.DataFrame, row: pd.Series, close: float,
                   news_bias: dict | None) -> tuple[str, list, float]:
    ema_bull, ema_bear, rsi, trending, reasons, news_contribution = \
        _score_indicators(row, close, news_bias)

    candle_trend, candle_reason = _read_candle_trend(df, lookback=5)
    reasons.append((0, f"Candle Trend: {candle_reason}"))

    if ema_bull and rsi < RSI_OVERBOUGHT and trending and candle_trend != "BEARISH":
        direction = "BUY"
    elif ema_bear and rsi > RSI_OVERSOLD and trending and candle_trend != "BULLISH":
        direction = "SELL"
    else:
        direction = "WAIT"
        if ema_bull and candle_trend == "BEARISH":
            reasons.append((0, "⚠ EMA BUY tapi candle bearish — tunggu konfirmasi"))
        elif ema_bear and candle_trend == "BULLISH":
            reasons.append((0, "⚠ EMA SELL tapi candle bullish — tunggu konfirmasi"))

    return direction, reasons, news_contribution


def generate_signal(df: pd.DataFrame, news_bias: dict | None = None) -> dict:
    if df.empty:
        return {"direction": "WAIT", "score": 0, "reasons": []}

    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row.get("atr", close * 0.001))

    direction, reasons, news_contribution = _make_decision(df, row, close, news_bias)

    scores = []
    for fn in [score_rsi, score_macd, score_stoch, score_adx, score_candle]:
        s, _ = fn(row)
        scores.append(s)
    s, _ = score_ema(row, close)
    scores.append(s)
    s, _ = score_bb(row, close)
    scores.append(s)

    total_score  = sum(scores)
    max_possible = sum(WEIGHTS.values())
    normalized   = round((total_score / max_possible) * 10, 2)
    final_score  = round(normalized + news_contribution, 2)

    tp_sl = calculate_smart_tp_sl(direction, close, atr, df, final_score)

    return {
        "direction":       direction,
        "score":           final_score,
        "score_technical": normalized,
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
