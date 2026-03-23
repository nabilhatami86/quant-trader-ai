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


def calculate_auto_tp_sl(direction: str, close: float, atr: float,
                          df: pd.DataFrame, decimals: int = 5) -> dict:
    resistances, supports = _find_swing_levels(df, lookback=150, pivot_window=5)
    buf = atr * 0.3

    if direction == "BUY":
        sl_candidates = [s for s in supports if s < close and (close - s) <= atr * 3]
        if sl_candidates:
            sl        = round(sl_candidates[0] - buf, decimals)
            sl_dist   = close - sl
            sl_method = "SwingLow"
        else:
            sl_dist   = atr * ATR_MULTIPLIER_SL
            sl        = round(close - sl_dist, decimals)
            sl_method = "ATR"

        min_tp_dist   = sl_dist * MIN_RR_RATIO
        tp_candidates = [r for r in resistances if r > close + min_tp_dist]
        if tp_candidates:
            tp        = round(tp_candidates[0], decimals)
            tp_dist   = tp_candidates[0] - close
            tp_method = "SwingHigh"
        else:
            tp_dist   = sl_dist * max(MIN_RR_RATIO, ATR_MULTIPLIER_TP)
            tp        = round(close + tp_dist, decimals)
            tp_method = "ATR"

    else:
        sl_candidates = [r for r in resistances if r > close and (r - close) <= atr * 3]
        if sl_candidates:
            sl        = round(sl_candidates[-1] + buf, decimals)
            sl_dist   = sl - close
            sl_method = "SwingHigh"
        else:
            sl_dist   = atr * ATR_MULTIPLIER_SL
            sl        = round(close + sl_dist, decimals)
            sl_method = "ATR"

        min_tp_dist   = sl_dist * MIN_RR_RATIO
        tp_candidates = [s for s in supports if s < close - min_tp_dist]
        if tp_candidates:
            tp        = round(tp_candidates[0], decimals)
            tp_dist   = close - tp_candidates[0]
            tp_method = "SwingLow"
        else:
            tp_dist   = sl_dist * max(MIN_RR_RATIO, ATR_MULTIPLIER_TP)
            tp        = round(close - tp_dist, decimals)
            tp_method = "ATR"

    rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
    return {
        "tp": tp, "sl": sl,
        "tp_dist": round(tp_dist, decimals),
        "sl_dist": round(sl_dist, decimals),
        "rr": rr,
        "method_tp": tp_method,
        "method_sl": sl_method,
    }


def calculate_smart_tp_sl(direction: str, close: float, atr: float,
                           df: pd.DataFrame, final_score: float,
                           decimals: int = 5) -> dict:
    if direction not in ("BUY", "SELL"):
        return {"tp": None, "sl": None, "tp_dist": 0,
                "sl_dist": 0, "rr": 0, "method_tp": "NONE", "method_sl": "NONE"}

    if AUTO_TP_SL:
        return calculate_auto_tp_sl(direction, close, atr, df, decimals)

    sl_dist   = atr * ATR_MULTIPLIER_SL
    tp_dist   = atr * ATR_MULTIPLIER_TP
    sl_method = "ATR"
    tp_method = "ATR"

    if direction == "BUY":
        sl = round(close - sl_dist, decimals)
        tp = round(close + tp_dist, decimals)
        resistances, _ = _find_swing_levels(df)
        swing_c = [r for r in resistances if r > close + tp_dist]
        if swing_c:
            tp, tp_method = round(swing_c[0], decimals), "SwingHigh"
            tp_dist = swing_c[0] - close
    else:
        sl = round(close + sl_dist, decimals)
        tp = round(close - tp_dist, decimals)
        _, supports = _find_swing_levels(df)
        swing_c = [s for s in supports if s < close - tp_dist]
        if swing_c:
            tp, tp_method = round(swing_c[0], decimals), "SwingLow"
            tp_dist = close - swing_c[0]

    rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
    return {
        "tp": tp, "sl": sl,
        "tp_dist": round(tp_dist, decimals),
        "sl_dist": round(sl_dist, decimals),
        "rr": rr,
        "method_tp": tp_method,
        "method_sl": sl_method,
    }


# ─────────────────────────────────────────────
# FILTER FUNCTIONS
# ─────────────────────────────────────────────

def _check_confirmation(df: pd.DataFrame, row: pd.Series,
                         direction: str) -> tuple[bool, str]:
    close      = float(row["Close"])
    open_      = float(row["Open"])
    high       = float(row["High"])
    low        = float(row["Low"])
    full_range = (high - low) or 0.0001
    wick_up    = high - max(open_, close)
    wick_down  = min(open_, close) - low

    if direction == "BUY":
        if close <= open_:
            return False, "Candle bearish — belum ada konfirmasi BUY"
        long_wick = wick_down / full_range > 0.35
        engulf    = False
        if len(df) >= 2:
            p      = df.iloc[-2]
            engulf = (float(p["Close"]) < float(p["Open"])
                      and close > float(p["Open"])
                      and open_ < float(p["Close"]))
        qual = "bullish engulfing" if engulf else ("lower wick panjang" if long_wick else "bullish")
        return True, f"Konfirmasi candle {qual} ✓"
    else:
        if close >= open_:
            return False, "Candle bullish — belum ada konfirmasi SELL"
        long_wick = wick_up / full_range > 0.35
        engulf    = False
        if len(df) >= 2:
            p      = df.iloc[-2]
            engulf = (float(p["Close"]) > float(p["Open"])
                      and close < float(p["Open"])
                      and open_ > float(p["Close"]))
        qual = "bearish engulfing" if engulf else ("upper wick panjang" if long_wick else "bearish")
        return True, f"Konfirmasi candle {qual} ✓"


def _check_rsi_momentum(df: pd.DataFrame, row: pd.Series,
                         direction: str) -> tuple[bool, str]:
    rsi_now  = float(row.get("rsi", 50))
    rsi_prev = float(df.iloc[-2].get("rsi", 50)) if len(df) >= 2 else rsi_now

    if direction == "BUY":
        if rsi_now >= 50:
            return False, f"RSI {rsi_now:.1f} ≥ 50 — momentum belum bullish"
        if rsi_now <= rsi_prev:
            return False, f"RSI turun {rsi_prev:.1f}→{rsi_now:.1f} — tunggu momentum naik"
        return True, f"RSI {rsi_prev:.1f}→{rsi_now:.1f} naik ✓"
    else:
        if rsi_now <= 50:
            return False, f"RSI {rsi_now:.1f} ≤ 50 — momentum belum bearish"
        if rsi_now >= rsi_prev:
            return False, f"RSI naik {rsi_prev:.1f}→{rsi_now:.1f} — tunggu momentum turun"
        return True, f"RSI {rsi_prev:.1f}→{rsi_now:.1f} turun ✓"


def _check_pullback(row: pd.Series, direction: str) -> tuple[bool, str]:
    close = float(row["Close"])
    ema20 = float(row.get(f"ema_{EMA_SLOW}", close))
    ema50 = float(row.get(f"ema_{EMA_TREND}", close))
    atr   = float(row.get("atr", close * 0.001))

    if direction == "BUY" and ema20 > ema50:
        dist = close - ema20
        if dist > atr * 1.5:
            return False, f"Harga {dist:.2f} di atas EMA20 — tunggu pullback"
    elif direction == "SELL" and ema20 < ema50:
        dist = ema20 - close
        if dist > atr * 1.5:
            return False, f"Harga {dist:.2f} di bawah EMA20 — tunggu pullback"

    return True, "Posisi entry setelah pullback ✓"


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
    return "MIXED", f"MIXED ({bullish} bull / {bearish} bear)"


# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

def _make_decision(df: pd.DataFrame, row: pd.Series, close: float,
                   news_bias: dict | None,
                   news_risk: str = "LOW") -> tuple[str, list, float, dict]:
    reasons  = []
    filters  = {}
    ema20    = float(row.get(f"ema_{EMA_SLOW}", close))
    ema50    = float(row.get(f"ema_{EMA_TREND}", close))
    rsi      = float(row.get("rsi", 50))
    adx      = float(row.get("adx", 0))
    trending = adx >= ADX_TREND_MIN

    ema_bull = ema20 > ema50
    ema_bear = ema20 < ema50

    candle_trend, candle_reason = _read_candle_trend(df, lookback=5)

    trend_str = "UP ↑" if ema_bull else ("DOWN ↓" if ema_bear else "FLAT")
    filters["trend"]    = f"{trend_str}  EMA20={ema20:.2f}  EMA50={ema50:.2f}  ADX={'OK' if trending else 'WEAK'}"
    filters["candle_trend"] = candle_reason

    news_contribution = 0.0
    if news_bias:
        n_score    = news_bias.get("score", 0.0)
        confidence = news_bias.get("confidence", "LOW")
        weight     = {"HIGH": 0.30, "MEDIUM": 0.20, "LOW": 0.10}.get(confidence, 0.10)
        news_contribution = n_score * weight
        d_word = "BULLISH" if n_score > 0 else "BEARISH" if n_score < 0 else "NEUTRAL"
        reasons.append((news_contribution, f"News Bias {d_word} (score {n_score:+.1f}, conf:{confidence})"))

    # 1. Trend + ADX filter
    if not trending:
        filters["momentum"] = f"SKIP — ADX {adx:.1f} sideways"
        filters["pullback"] = "SKIP"
        filters["confirmation"] = "SKIP"
        reasons.append((0, f"ADX {adx:.1f} < {ADX_TREND_MIN} — market sideways"))
        return "WAIT", reasons, news_contribution, filters

    if not (ema_bull or ema_bear):
        filters["momentum"] = "SKIP — EMA flat"
        filters["pullback"] = "SKIP"
        filters["confirmation"] = "SKIP"
        return "WAIT", reasons, news_contribution, filters

    raw_dir = "BUY" if ema_bull else "SELL"
    reasons.append((0, f"Trend {trend_str}  →  arah awal: {raw_dir}"))

    # 2. News HIGH → block sepenuhnya
    if news_risk == "HIGH":
        filters["news"]         = "HIGH RISK — NO TRADE"
        filters["momentum"]     = "SKIP"
        filters["pullback"]     = "SKIP"
        filters["confirmation"] = "SKIP"
        reasons.append((0, "⛔ News HIGH impact — entry dilarang"))
        return "WAIT", reasons, news_contribution, filters
    filters["news"] = f"{news_risk} — OK"

    # 3. RSI momentum
    ok, msg = _check_rsi_momentum(df, row, raw_dir)
    filters["momentum"] = msg
    if not ok:
        reasons.append((0, f"⚠ Momentum: {msg}"))
        return "WAIT", reasons, news_contribution, filters

    # 4. Pullback logic
    ok, msg = _check_pullback(row, raw_dir)
    filters["pullback"] = msg
    if not ok:
        reasons.append((0, f"⚠ Pullback: {msg}"))
        return "WAIT", reasons, news_contribution, filters

    # 5. Candle confirmation
    ok, msg = _check_confirmation(df, row, raw_dir)
    filters["confirmation"] = msg
    if not ok:
        reasons.append((0, f"⚠ Konfirmasi: {msg}"))
        return "WAIT", reasons, news_contribution, filters

    reasons.append((+1 if raw_dir == "BUY" else -1, f"Semua filter passed → {raw_dir}"))
    return raw_dir, reasons, news_contribution, filters


def generate_signal(df: pd.DataFrame,
                    news_bias: dict | None = None,
                    news_risk: str = "LOW",
                    candle_memory: dict | None = None) -> dict:
    if df.empty:
        return {"direction": "WAIT", "score": 0, "reasons": [], "filters": {}}

    row   = df.iloc[-1]
    close = float(row["Close"])
    atr   = float(row.get("atr", close * 0.001))

    direction, reasons, news_contribution, filters = \
        _make_decision(df, row, close, news_bias, news_risk)

    # Candle memory bias
    memory_contribution = 0.0
    if candle_memory and candle_memory.get("total", 0) >= 8:
        mem_bias = candle_memory.get("bias", "NEUTRAL")
        mem_wr   = candle_memory.get("win_rate", 50)
        weight   = (mem_wr - 50) / 50
        if mem_bias == "BUY":
            memory_contribution = +1.5 * weight
        elif mem_bias == "SELL":
            memory_contribution = -1.5 * weight
        reasons.append((memory_contribution,
                        f"Candle Memory {mem_bias} ({mem_wr:.0f}% dari {candle_memory['total']} match)"))
        if direction == "BUY" and mem_bias == "SELL" and mem_wr >= 70:
            direction = "WAIT"
            reasons.append((0, "⚠ Memory bearish 70%+ — sinyal BUY dibatalkan"))
        elif direction == "SELL" and mem_bias == "BUY" and mem_wr >= 70:
            direction = "WAIT"
            reasons.append((0, "⚠ Memory bullish 70%+ — sinyal SELL dibatalkan"))

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
    final_score  = round(normalized + news_contribution + memory_contribution, 2)

    tp_sl = calculate_smart_tp_sl(direction, close, atr, df, final_score)

    if direction in ("BUY", "SELL") and tp_sl["rr"] < MIN_RR_RATIO:
        filters["rr"] = f"REJECT — RR {tp_sl['rr']} < {MIN_RR_RATIO}"
        direction = "WAIT"
    elif direction in ("BUY", "SELL"):
        filters["rr"] = f"OK — RR {tp_sl['rr']}"

    return {
        "direction":       direction,
        "score":           final_score,
        "score_technical": normalized,
        "score_news":      round(news_contribution, 2),
        "score_memory":    round(memory_contribution, 2),
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
        "filters":         filters,
        "raw_score":       round(total_score, 3),
    }


def print_filter_log(filters: dict, direction: str) -> None:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    dc = GREEN if direction == "BUY" else RED if direction == "SELL" else YELLOW
    print(f"\n  {BOLD}[FILTER LOG]{RESET}  Final: {dc}{BOLD}{direction}{RESET}")

    icons = {
        "trend":        "📈",
        "news":         "📰",
        "momentum":     "📊",
        "pullback":     "↩️ ",
        "confirmation": "🕯️ ",
        "candle_trend": "📉",
        "rr":           "⚖️ ",
    }
    for key, val in filters.items():
        icon   = icons.get(key, "  ")
        passed = "SKIP" not in str(val) and "WEAK" not in str(val) \
                 and "REJECT" not in str(val) and "belum" not in str(val) \
                 and "tunggu" not in str(val) and "NO TRADE" not in str(val)
        color  = GREEN if passed else (RED if "NO TRADE" in str(val) or "REJECT" in str(val) else YELLOW)
        print(f"    {icon} {key:<14}: {color}{val}{RESET}")
