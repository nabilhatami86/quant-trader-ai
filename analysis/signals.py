import pandas as pd
import numpy as np
from config import *


# ─────────────────────────────────────────────
# SWING LEVEL DETECTION
# ─────────────────────────────────────────────

def _find_swing_levels(df: pd.DataFrame, lookback: int = 100,
                       pivot_window: int = 5) -> tuple[list, list]:
    data  = df.tail(lookback)
    highs = data["High"].values
    lows  = data["Low"].values
    w     = pivot_window

    resistances, supports = [], []
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


# ─────────────────────────────────────────────
# TP / SL CALCULATION
# ─────────────────────────────────────────────

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
        "rr": rr, "method_tp": tp_method, "method_sl": sl_method,
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
        "rr": rr, "method_tp": tp_method, "method_sl": sl_method,
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

    # Extra patterns override → Three Soldiers / Morning Star tidak perlu tunggu candle bullish
    ex_pat = int(row.get("candle_ex", 0))

    if direction == "BUY":
        if ex_pat >= 1:
            return True, f"Extra pattern bullish ({int(ex_pat)}) override konfirmasi"
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
        return True, f"Konfirmasi candle {qual}"
    else:
        if ex_pat <= -1:
            return True, f"Extra pattern bearish ({int(ex_pat)}) override konfirmasi"
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
        return True, f"Konfirmasi candle {qual}"


def _check_rsi_momentum(df: pd.DataFrame, row: pd.Series,
                         direction: str) -> tuple[bool, str]:
    rsi_now  = float(row.get("rsi", 50))
    rsi_prev = float(df.iloc[-2].get("rsi", 50)) if len(df) >= 2 else rsi_now

    # Liquidity sweep → relaksasi RSI filter (harga sudah reversal)
    liq_bull = int(row.get("liq_bull_sweep", 0))
    liq_bear = int(row.get("liq_bear_sweep", 0))

    if direction == "BUY":
        if liq_bull:
            return True, f"RSI {rsi_now:.1f} — Liquidity sweep override"
        if rsi_now >= 55:
            return False, f"RSI {rsi_now:.1f} >= 55 — momentum belum bullish"
        if rsi_now <= rsi_prev and rsi_now > RSI_OVERSOLD:
            return False, f"RSI turun {rsi_prev:.1f}→{rsi_now:.1f} — tunggu momentum naik"
        return True, f"RSI {rsi_prev:.1f}→{rsi_now:.1f} bullish"
    else:
        if liq_bear:
            return True, f"RSI {rsi_now:.1f} — Liquidity sweep override"
        if rsi_now <= 45:
            return False, f"RSI {rsi_now:.1f} <= 45 — momentum belum bearish"
        if rsi_now >= rsi_prev and rsi_now < RSI_OVERBOUGHT:
            return False, f"RSI naik {rsi_prev:.1f}→{rsi_now:.1f} — tunggu momentum turun"
        return True, f"RSI {rsi_prev:.1f}→{rsi_now:.1f} bearish"


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

    return True, "Posisi entry OK"


def _read_candle_trend(df: pd.DataFrame, lookback: int = 5) -> tuple[str, str]:
    if len(df) < lookback:
        return "MIXED", "Data tidak cukup"
    recent  = df.tail(lookback)
    bullish = int(((recent["Close"] - recent["Open"]) > 0).sum())
    bearish = lookback - bullish
    ratio   = bullish / lookback
    if ratio >= 0.7:
        return "BULLISH", f"{bullish}/{lookback} candle terakhir BULLISH"
    elif ratio <= 0.3:
        return "BEARISH", f"{bearish}/{lookback} candle terakhir BEARISH"
    return "MIXED", f"MIXED ({bullish} bull / {bearish} bear)"


# ─────────────────────────────────────────────
# TRADITIONAL SCORING FUNCTIONS
# ─────────────────────────────────────────────

def score_rsi(row: pd.Series) -> tuple[float, str]:
    rsi = row.get("rsi", 50)
    if rsi <= RSI_OVERSOLD:
        return WEIGHTS["rsi"], f"RSI Oversold {rsi:.1f} (BUY)"
    elif rsi >= RSI_OVERBOUGHT:
        return -WEIGHTS["rsi"], f"RSI Overbought {rsi:.1f} (SELL)"
    elif rsi < 45:
        return WEIGHTS["rsi"] * 0.5, f"RSI Bullish zone {rsi:.1f}"
    elif rsi > 55:
        return -WEIGHTS["rsi"] * 0.5, f"RSI Bearish zone {rsi:.1f}"
    return 0, f"RSI Neutral {rsi:.1f}"


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
        return WEIGHTS["ema_cross"], "EMA Bullish Aligned"
    elif e_fast < e_slow and close < e_trend:
        return -WEIGHTS["ema_cross"], "EMA Bearish Aligned"
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
    return 0, "No Basic Pattern"


# ─────────────────────────────────────────────
# NEW: VOLUME SCORING
# ─────────────────────────────────────────────

def score_obv(row: pd.Series, df: pd.DataFrame) -> tuple[float, str]:
    """OBV trend: apakah volume mendukung arah harga."""
    obv     = df.get("obv", pd.Series(dtype=float))
    obv_ema = df.get("obv_ema", pd.Series(dtype=float))
    if obv.empty or obv_ema.empty or len(df) < 5:
        return 0, "OBV N/A"

    obv_now   = float(obv.iloc[-1])
    obv_ema_n = float(obv_ema.iloc[-1])
    obv_prev  = float(obv.iloc[-5])

    obv_rising = obv_now > obv_prev
    obv_above  = obv_now > obv_ema_n

    if obv_rising and obv_above:
        return WEIGHTS.get("obv", 1.5), "OBV naik & di atas EMA → volume bullish"
    elif not obv_rising and not obv_above:
        return -WEIGHTS.get("obv", 1.5), "OBV turun & di bawah EMA → volume bearish"
    elif obv_rising:
        return WEIGHTS.get("obv", 1.5) * 0.5, "OBV naik (bullish)"
    elif not obv_rising:
        return -WEIGHTS.get("obv", 1.5) * 0.5, "OBV turun (bearish)"
    return 0, "OBV Neutral"


def score_vwap(row: pd.Series, close: float) -> tuple[float, str]:
    """Price vs VWAP — institusi biasa bertransaksi di atas/bawah VWAP."""
    vwap = float(row.get("vwap", close))
    if vwap == 0 or vwap != vwap:   # nan check
        return 0, "VWAP N/A"
    dist_pct = (close - vwap) / vwap * 100
    w = WEIGHTS.get("vwap", 1.5)
    if dist_pct > 0.1:
        return w * min(dist_pct / 0.3, 1.0), f"Price {dist_pct:+.2f}% di atas VWAP (bullish)"
    elif dist_pct < -0.1:
        return -w * min(-dist_pct / 0.3, 1.0), f"Price {dist_pct:+.2f}% di bawah VWAP (bearish)"
    return 0, f"Price dekat VWAP ({dist_pct:+.2f}%)"


def score_williams_r(row: pd.Series) -> tuple[float, str]:
    """Williams %R: -80 s/d -20 dianggap oversold/overbought."""
    wr = float(row.get("williams_r", -50))
    w  = WEIGHTS.get("williams_r", 1.0)
    if wr <= -80:
        return w, f"Williams %R Oversold ({wr:.1f})"
    elif wr >= -20:
        return -w, f"Williams %R Overbought ({wr:.1f})"
    elif wr <= -60:
        return w * 0.5, f"Williams %R Bearish zone ({wr:.1f})"
    elif wr >= -40:
        return -w * 0.5, f"Williams %R Bullish zone ({wr:.1f})"
    return 0, f"Williams %R Neutral ({wr:.1f})"


def score_cci(row: pd.Series) -> tuple[float, str]:
    """CCI: >100 overbought, <-100 oversold."""
    cci = float(row.get("cci", 0))
    w   = WEIGHTS.get("cci", 1.0)
    if cci <= -100:
        return w, f"CCI Oversold ({cci:.0f})"
    elif cci >= 100:
        return -w, f"CCI Overbought ({cci:.0f})"
    elif cci < -50:
        return w * 0.5, f"CCI Bearish zone ({cci:.0f})"
    elif cci > 50:
        return -w * 0.5, f"CCI Bullish zone ({cci:.0f})"
    return 0, f"CCI Neutral ({cci:.0f})"


# ─────────────────────────────────────────────
# NEW: SMART MONEY SCORING
# ─────────────────────────────────────────────

def score_smc(row: pd.Series, close: float) -> tuple[float, str]:
    """
    Smart Money Concepts composite score:
    - FVG (Fair Value Gap): harga di dalam FVG → area magnet, entry kuat
    - Order Block: harga menyentuh OB level → zona institusi
    - BOS/ChoCH: konfirmasi atau perubahan struktur
    - Liquidity Sweep: reversal setelah stop hunt
    """
    w      = WEIGHTS.get("smc", 3.0)
    score  = 0.0
    parts  = []

    # 1. FVG
    fvg_bull = int(row.get("fvg_bull", 0))
    fvg_bear = int(row.get("fvg_bear", 0))
    if fvg_bull:
        score += w * 0.4
        parts.append("Bullish FVG aktif")
    if fvg_bear:
        score -= w * 0.4
        parts.append("Bearish FVG aktif")

    # 2. Order Block
    ob_bull = int(row.get("ob_bull", 0))
    ob_bear = int(row.get("ob_bear", 0))
    if ob_bull:
        score += w * 0.35
        parts.append("Bullish Order Block")
    if ob_bear:
        score -= w * 0.35
        parts.append("Bearish Order Block")

    # 3. BOS / ChoCH
    bos_bull  = int(row.get("bos_bull", 0))
    bos_bear  = int(row.get("bos_bear", 0))
    choch_bull = int(row.get("choch_bull", 0))
    choch_bear = int(row.get("choch_bear", 0))
    if bos_bull:
        score += w * 0.3
        parts.append("BOS Bullish (struktur naik)")
    if bos_bear:
        score -= w * 0.3
        parts.append("BOS Bearish (struktur turun)")
    if choch_bull:
        score += w * 0.5
        parts.append("ChoCH Bullish (reversal besar!)")
    if choch_bear:
        score -= w * 0.5
        parts.append("ChoCH Bearish (reversal besar!)")

    # 4. Liquidity Sweep
    liq_bull = int(row.get("liq_bull_sweep", 0))
    liq_bear = int(row.get("liq_bear_sweep", 0))
    if liq_bull:
        score += w * 0.6
        parts.append("Liquidity Sweep Bull (stop hunt selesai!)")
    if liq_bear:
        score -= w * 0.6
        parts.append("Liquidity Sweep Bear (stop hunt selesai!)")

    desc = " | ".join(parts) if parts else "No SMC signal"
    return round(score, 3), desc


def score_extra_patterns(row: pd.Series) -> tuple[float, str]:
    """Extra candle patterns: Three Soldiers, Crows, Morning/Evening Star, Harami."""
    pat = int(row.get("candle_ex", 0))
    w   = WEIGHTS.get("pattern_ex", 1.5)
    if pat == 2:
        return w, "Three White Soldiers (sangat bullish)"
    elif pat == 1:
        return w * 0.6, "Morning Star / Bullish Harami"
    elif pat == -2:
        return -w, "Three Black Crows (sangat bearish)"
    elif pat == -1:
        return -w * 0.6, "Evening Star / Bearish Harami"
    return 0, "No Extra Pattern"


def score_volume_context(row: pd.Series) -> tuple[float, str]:
    """Volume spike + divergence context."""
    vol_spike = int(row.get("vol_spike", 0))
    vol_div   = int(row.get("vol_div", 0))
    w         = WEIGHTS.get("volume", 1.0)

    if vol_spike and vol_div == 1:
        return w, "Volume spike bullish (divergence dikonfirmasi)"
    elif vol_spike and vol_div == -1:
        return -w, "Volume spike bearish (divergence dikonfirmasi)"
    elif vol_div == 1:
        return w * 0.4, "Volume divergence bullish"
    elif vol_div == -1:
        return -w * 0.4, "Volume divergence bearish"
    return 0, "Volume Normal"


# ─────────────────────────────────────────────
# NEW: SMA SCORING
# ─────────────────────────────────────────────

def score_sma(row: pd.Series, close: float) -> tuple[float, str]:
    """
    SMA cross dan posisi harga terhadap SMA.
    Lebih lambat dari EMA tetapi lebih stabil — konfirmasi trend jangka panjang.
    """
    sma10  = float(row.get("sma10",  close))
    sma20  = float(row.get("sma20",  close))
    sma50  = float(row.get("sma50",  close))
    sma200 = float(row.get("sma200", close))
    w = WEIGHTS.get("sma", 1.5)

    # Golden Cross (SMA50 > SMA200) / Death Cross (SMA50 < SMA200)
    if sma50 > sma200 and close > sma50:
        return w, f"Golden Cross: SMA50>{sma200:.0f} & price atas SMA50 (bullish)"
    elif sma50 < sma200 and close < sma50:
        return -w, f"Death Cross: SMA50<{sma200:.0f} & price bawah SMA50 (bearish)"
    # SMA10/20 cross untuk jangka pendek
    elif sma10 > sma20 and close > sma20:
        return w * 0.5, f"SMA10>{sma20:.0f} (short-term bullish)"
    elif sma10 < sma20 and close < sma20:
        return -w * 0.5, f"SMA10<{sma20:.0f} (short-term bearish)"
    return 0, "SMA Neutral"


# ─────────────────────────────────────────────
# NEW: FIBONACCI SCORING
# ─────────────────────────────────────────────

def score_fibonacci(row: pd.Series, close: float, atr: float) -> tuple[float, str]:
    """
    Price near Fibonacci Retracement levels — area support/resistance kuat.
    Level paling signifikan: 38.2%, 50%, 61.8%
    Scoring berdasarkan proximity: semakin dekat ke level, semakin kuat sinyal.
    """
    w = WEIGHTS.get("fibonacci", 2.0)

    fib_high = float(row.get("fib_swing_high", close))
    fib_low  = float(row.get("fib_swing_low",  close))
    rng      = fib_high - fib_low
    if rng < atr * 0.5:
        return 0, "Fibonacci: range terlalu kecil"

    levels = {
        "23.6%": float(row.get("fib_236", close)),
        "38.2%": float(row.get("fib_382", close)),
        "50.0%": float(row.get("fib_500", close)),
        "61.8%": float(row.get("fib_618", close)),
        "78.6%": float(row.get("fib_786", close)),
    }
    weights = {"23.6%": 0.3, "38.2%": 0.7, "50.0%": 0.6, "61.8%": 0.8, "78.6%": 0.5}

    tolerance = atr * 0.3   # dalam 0.3 ATR dianggap "di level"
    nearest_level = None
    nearest_dist  = float("inf")

    for name, lvl in levels.items():
        dist = abs(close - lvl)
        if dist < nearest_dist:
            nearest_dist  = dist
            nearest_level = (name, lvl, weights[name])

    if nearest_dist > tolerance:
        return 0, f"Fibonacci: jauh dari level (dist={nearest_dist:.2f}, tol={tolerance:.2f})"

    name, lvl, lw = nearest_level
    # Tentukan arah: jika dalam uptrend (close > fib midpoint) → pullback ke level = BUY
    # Jika dalam downtrend → bounce ke level = SELL
    fib_mid = (fib_high + fib_low) / 2
    if close < fib_mid:
        # Harga di area bawah → level fib = support → BUY
        s = w * lw
        return s, f"Fib {name} Support @ {lvl:.2f} (BUY zone)"
    else:
        # Harga di area atas → level fib = resistance → SELL
        s = -w * lw
        return s, f"Fib {name} Resistance @ {lvl:.2f} (SELL zone)"


# ─────────────────────────────────────────────
# NEW: RSI DIVERGENCE SCORING
# ─────────────────────────────────────────────

def score_rsi_divergence(row: pd.Series) -> tuple[float, str]:
    """
    RSI Divergence — sinyal reversal paling kuat.
    Regular divergence: bobot tinggi (4.0)
    Hidden divergence: konfirmasi trend (2.5)
    """
    w = WEIGHTS.get("rsi_div", 4.0)

    bull_div = int(row.get("rsi_bull_div", 0))
    bear_div = int(row.get("rsi_bear_div", 0))
    hid_bull = int(row.get("rsi_hid_bull", 0))
    hid_bear = int(row.get("rsi_hid_bear", 0))

    if bull_div:
        return w, "RSI Bullish Divergence — reversal NAIK kuat!"
    elif bear_div:
        return -w, "RSI Bearish Divergence — reversal TURUN kuat!"
    elif hid_bull:
        return w * 0.6, "RSI Hidden Bullish Div — pullback dalam uptrend, lanjut naik"
    elif hid_bear:
        return -w * 0.6, "RSI Hidden Bearish Div — pullback dalam downtrend, lanjut turun"
    return 0, "No RSI Divergence"


# ─────────────────────────────────────────────
# NEW: MOMENTUM CHAIN SCORING
# ─────────────────────────────────────────────

def score_momentum_chain(row: pd.Series) -> tuple[float, str]:
    """
    Market Structure Chain (HH+HL / LL+LH) — struktur trend.
    Lebih kuat dari ADX karena melihat STRUKTUR, bukan hanya kekuatan.
    bull_chain=6+ dari max 8 → struktur uptrend sangat kuat
    """
    w          = WEIGHTS.get("momentum_chain", 2.0)
    bull_chain = float(row.get("bull_chain", 0))
    bear_chain = float(row.get("bear_chain", 0))
    slope      = float(row.get("close_slope", 0))

    max_chain  = 8.0   # n=4 candles × 2 kondisi (HH+HL)
    bull_norm  = min(bull_chain / max_chain, 1.0)
    bear_norm  = min(bear_chain / max_chain, 1.0)

    if bull_norm > bear_norm:
        # Slope konfirmasi arah
        s = w * bull_norm * (1.2 if slope > 0 else 0.7)
        return round(s, 3), f"Bull Chain {int(bull_chain)}/8 (struktur uptrend)"
    elif bear_norm > bull_norm:
        s = -w * bear_norm * (1.2 if slope < 0 else 0.7)
        return round(s, 3), f"Bear Chain {int(bear_chain)}/8 (struktur downtrend)"
    return 0, "Structure Neutral"


# ─────────────────────────────────────────────
# REGIME-AWARE DECISION ENGINE
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
    regime   = str(row.get("regime", "RANGE"))
    trending = adx >= ADX_TREND_MIN

    ema_bull = ema20 > ema50
    ema_bear = ema20 < ema50

    candle_trend, candle_reason = _read_candle_trend(df, lookback=5)

    trend_str = "UP" if ema_bull else ("DOWN" if ema_bear else "FLAT")
    filters["regime"]  = f"{regime}  ADX={adx:.1f}  Trend={trend_str}"
    filters["candle_trend"] = candle_reason

    # SMC signals — bisa override ADX sideways jika ChoCH/Liquidity sweep
    choch_bull  = int(row.get("choch_bull", 0))
    choch_bear  = int(row.get("choch_bear", 0))
    liq_bull    = int(row.get("liq_bull_sweep", 0))
    liq_bear    = int(row.get("liq_bear_sweep", 0))
    smc_force   = choch_bull or choch_bear or liq_bull or liq_bear
    smc_dir     = "BUY" if (choch_bull or liq_bull) else ("SELL" if (choch_bear or liq_bear) else None)

    news_contribution = 0.0
    if news_bias:
        n_score    = news_bias.get("score", 0.0)
        confidence = news_bias.get("confidence", "LOW")
        weight     = {"HIGH": 0.30, "MEDIUM": 0.20, "LOW": 0.10}.get(confidence, 0.10)
        news_contribution = n_score * weight
        d_word = "BULLISH" if n_score > 0 else "BEARISH" if n_score < 0 else "NEUTRAL"
        reasons.append((news_contribution, f"News Bias {d_word} (score {n_score:+.1f}, conf:{confidence})"))

    # ── 1. Trend + ADX filter (SMC bisa override jika sideways) ──────
    if not trending and not smc_force:
        filters["momentum"] = f"SKIP — ADX {adx:.1f} sideways, no SMC trigger"
        filters["pullback"] = "SKIP"
        filters["confirmation"] = "SKIP"
        reasons.append((0, f"ADX {adx:.1f} < {ADX_TREND_MIN} — market sideways"))
        return "WAIT", reasons, news_contribution, filters

    if not trending and smc_force:
        # ADX lemah tapi ada SMC signal kuat — masih bisa trade
        filters["momentum"] = f"ADX {adx:.1f} weak, tapi SMC override ({smc_dir})"
        raw_dir = smc_dir
        reasons.append((0, f"SMC override ADX lemah → arah {raw_dir}"))
    else:
        if not (ema_bull or ema_bear):
            filters["momentum"] = "SKIP — EMA flat"
            filters["pullback"] = "SKIP"
            filters["confirmation"] = "SKIP"
            return "WAIT", reasons, news_contribution, filters
        raw_dir = "BUY" if ema_bull else "SELL"
        reasons.append((0, f"Trend {trend_str} → arah awal: {raw_dir}"))

    # Jika SMC force direction berbeda dari EMA trend → ikut SMC (ChoCH = trend change)
    if smc_force and smc_dir and smc_dir != raw_dir:
        if choch_bull or choch_bear:
            raw_dir = smc_dir
            reasons.append((0, f"ChoCH override EMA trend → {raw_dir}"))

    # ── 2. News HIGH → block ──────────────────────────────────────────
    if news_risk == "HIGH":
        filters["news"]         = "HIGH RISK — NO TRADE"
        filters["momentum"]     = "SKIP"
        filters["pullback"]     = "SKIP"
        filters["confirmation"] = "SKIP"
        reasons.append((0, "News HIGH impact — entry dilarang"))
        return "WAIT", reasons, news_contribution, filters
    filters["news"] = f"{news_risk} — OK"

    # ── 3. RSI momentum ───────────────────────────────────────────────
    ok, msg = _check_rsi_momentum(df, row, raw_dir)
    filters["momentum"] = msg
    if not ok:
        reasons.append((0, f"Momentum: {msg}"))
        return "WAIT", reasons, news_contribution, filters

    # ── 4. Pullback ───────────────────────────────────────────────────
    ok, msg = _check_pullback(row, raw_dir)
    filters["pullback"] = msg
    if not ok:
        reasons.append((0, f"Pullback: {msg}"))
        return "WAIT", reasons, news_contribution, filters

    # ── 5. Candle confirmation ────────────────────────────────────────
    ok, msg = _check_confirmation(df, row, raw_dir)
    filters["confirmation"] = msg
    if not ok:
        reasons.append((0, f"Konfirmasi: {msg}"))
        return "WAIT", reasons, news_contribution, filters

    reasons.append((+1 if raw_dir == "BUY" else -1, f"Semua filter passed → {raw_dir}"))
    return raw_dir, reasons, news_contribution, filters


# ─────────────────────────────────────────────
# MASTER SIGNAL GENERATOR
# ─────────────────────────────────────────────

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

    # ── Candle memory bias ─────────────────────────────────────────────
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
            reasons.append((0, "Memory bearish 70%+ — sinyal BUY dibatalkan"))
        elif direction == "SELL" and mem_bias == "BUY" and mem_wr >= 70:
            direction = "WAIT"
            reasons.append((0, "Memory bullish 70%+ — sinyal SELL dibatalkan"))

    # ── Traditional scoring ────────────────────────────────────────────
    trad_scores = []
    for fn in [score_rsi, score_macd, score_stoch, score_adx, score_candle]:
        s, _ = fn(row)
        trad_scores.append(s)
    s, _ = score_ema(row, close)
    trad_scores.append(s)
    s, _ = score_bb(row, close)
    trad_scores.append(s)

    # ── New: Volume scoring ────────────────────────────────────────────
    s_obv,  d_obv  = score_obv(row, df)
    s_vwap, d_vwap = score_vwap(row, close)
    s_wr,   d_wr   = score_williams_r(row)
    s_cci,  d_cci  = score_cci(row)
    s_vctx, d_vctx = score_volume_context(row)

    # ── New: SMC + pattern scoring ─────────────────────────────────────
    s_smc,   d_smc   = score_smc(row, close)
    s_expat, d_expat = score_extra_patterns(row)

    # ── New: SMA + Fibonacci scoring ───────────────────────────────────
    s_sma,  d_sma  = score_sma(row, close)
    s_fib,  d_fib  = score_fibonacci(row, close, atr)

    # ── New: RSI Divergence + Momentum Chain ───────────────────────────
    s_rdiv, d_rdiv = score_rsi_divergence(row)
    s_mch,  d_mch  = score_momentum_chain(row)

    # ── Regime multipliers per component ─────────────────────────────
    regime = str(row.get("regime", "RANGE"))
    if regime == "VOLATILE":
        vol_mult  = 1.3
        smc_mult  = 1.3
        trad_mult = 0.7
        str_mult  = 1.2   # momentum structure & divergence
    elif regime == "RANGE":
        vol_mult  = 1.0
        smc_mult  = 1.2
        trad_mult = 0.9
        str_mult  = 0.8
    else:  # TREND
        vol_mult  = 1.1
        smc_mult  = 1.0
        trad_mult = 1.1
        str_mult  = 1.3   # structure matters most in trend

    total_trad  = sum(trad_scores) * trad_mult
    total_vol   = (s_obv + s_vwap + s_wr + s_cci + s_vctx) * vol_mult
    total_smc   = (s_smc + s_expat) * smc_mult
    total_str   = (s_rdiv + s_mch + s_sma + s_fib) * str_mult

    # Normalize traditional score ke -10..+10
    max_trad    = sum(WEIGHTS[k] for k in ["rsi","macd","ema_cross","bb","stoch","adx","candle"])
    normalized_trad = (total_trad / max_trad) * 10 if max_trad > 0 else 0

    # Signal Quality Gate: jika contradicting signals, dampen score
    # RSI divergence opposing EMA direction → reduce confidence
    if s_rdiv > 0 and normalized_trad < -2:
        # Divergence says BUY but trend says SELL — reduce divergence weight
        total_str *= 0.5
    elif s_rdiv < 0 and normalized_trad > 2:
        total_str *= 0.5

    # Volume + SMC + Structure: langsung tambahkan
    final_score = round(
        normalized_trad + total_vol + total_smc + total_str
        + news_contribution + memory_contribution,
        2
    )

    # ── Signal Quality Gate: confidence threshold ──────────────────────
    # Jika sinyal lemah (abu-abu) → WAIT daripada masuk dengan keyakinan rendah
    MIN_CONFIDENCE = 3.0
    if direction in ("BUY", "SELL") and abs(final_score) < MIN_CONFIDENCE:
        direction = "WAIT"
        reasons.append((0, f"Score {final_score:.2f} < MIN_CONFIDENCE {MIN_CONFIDENCE} — terlalu lemah"))

    # Block RANGE regime tanpa SMC atau RSI divergence signal
    if regime == "RANGE" and direction in ("BUY", "SELL"):
        smc_present = abs(s_smc) > 0.5
        div_present = abs(s_rdiv) > 0
        if not smc_present and not div_present:
            direction = "WAIT"
            reasons.append((0, f"RANGE regime tanpa SMC/Divergence — WAIT"))

    # Tambahkan sub-scores ke reasons untuk audit
    if s_obv != 0:
        reasons.append((s_obv, d_obv))
    if s_vwap != 0:
        reasons.append((s_vwap, d_vwap))
    if s_wr != 0:
        reasons.append((s_wr, d_wr))
    if s_cci != 0:
        reasons.append((s_cci, d_cci))
    if s_smc != 0:
        reasons.append((s_smc, d_smc))
    if s_expat != 0:
        reasons.append((s_expat, d_expat))
    if s_vctx != 0:
        reasons.append((s_vctx, d_vctx))
    if s_rdiv != 0:
        reasons.append((s_rdiv, d_rdiv))
    if s_mch != 0:
        reasons.append((s_mch, d_mch))
    if s_sma != 0:
        reasons.append((s_sma, d_sma))
    if s_fib != 0:
        reasons.append((s_fib, d_fib))

    # ── Regime info ────────────────────────────────────────────────────
    filters["regime_score"] = (
        f"Regime={regime} | Trad×{trad_mult} SMC×{smc_mult} Vol×{vol_mult} Str×{str_mult}"
    )

    tp_sl = calculate_smart_tp_sl(direction, close, atr, df, final_score)

    if direction in ("BUY", "SELL") and tp_sl["rr"] < MIN_RR_RATIO:
        filters["rr"] = f"REJECT — RR {tp_sl['rr']} < {MIN_RR_RATIO}"
        direction = "WAIT"
    elif direction in ("BUY", "SELL"):
        filters["rr"] = f"OK — RR {tp_sl['rr']}"

    return {
        "direction":        direction,
        "score":            final_score,
        "score_technical":  round(normalized_trad, 2),
        "score_volume":     round(total_vol, 2),
        "score_smc":        round(total_smc, 2),
        "score_structure":  round(total_str, 2),
        "score_news":       round(news_contribution, 2),
        "score_memory":     round(memory_contribution, 2),
        "close":            close,
        "atr":              round(atr, 5),
        "regime":           regime,
        "sl":               tp_sl["sl"],
        "tp":               tp_sl["tp"],
        "tp_dist":          tp_sl["tp_dist"],
        "sl_dist":          tp_sl["sl_dist"],
        "rr_ratio":         tp_sl["rr"],
        "method_tp":        tp_sl["method_tp"],
        "method_sl":        tp_sl["method_sl"],
        "reasons":          reasons,
        "filters":          filters,
        "raw_score":        round(total_trad + total_vol + total_smc + total_str, 3),
    }


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_filter_log(filters: dict, direction: str) -> None:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    dc = GREEN if direction == "BUY" else RED if direction == "SELL" else YELLOW
    print(f"\n  {BOLD}[FILTER LOG]{RESET}  Final: {dc}{BOLD}{direction}{RESET}")

    icons = {
        "regime":        "  ",
        "regime_score":  "  ",
        "trend":         "  ",
        "news":          "  ",
        "momentum":      "  ",
        "pullback":      "  ",
        "confirmation":  "  ",
        "candle_trend":  "  ",
        "rr":            "  ",
    }
    for key, val in filters.items():
        icon   = icons.get(key, "  ")
        passed = not any(x in str(val) for x in
                         ["SKIP", "WEAK", "REJECT", "belum", "tunggu", "NO TRADE"])
        color  = GREEN if passed else (RED if "NO TRADE" in str(val) or "REJECT" in str(val) else YELLOW)
        print(f"    {icon} {key:<16}: {color}{val}{RESET}")
