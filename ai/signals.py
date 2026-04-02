
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
# MARKET STRUCTURE DETECTION (Upgrade 9)
# ─────────────────────────────────────────────

def _check_market_structure(df: pd.DataFrame, direction: str,
                            lookback: int = 20) -> tuple[bool, str]:
    """
    Deteksi struktur pasar dari pivot highs/lows.
    BUY valid hanya di uptrend structure (HH + HL).
    SELL valid hanya di downtrend structure (LL + LH).
    Hanya blok jika berlawanan penuh (HH+HL saat SELL atau LL+LH saat BUY).
    """
    if len(df) < lookback + 3:
        return True, "structure: data kurang"

    hi = df["High"].iloc[-lookback:].values
    lo = df["Low"].iloc[-lookback:].values

    ph, pl = [], []
    for i in range(1, len(hi) - 1):
        if hi[i] >= hi[i - 1] and hi[i] >= hi[i + 1]:
            ph.append(float(hi[i]))
        if lo[i] <= lo[i - 1] and lo[i] <= lo[i + 1]:
            pl.append(float(lo[i]))

    if len(ph) < 2 or len(pl) < 2:
        return True, "structure: pivot kurang — skip"

    hh = ph[-1] > ph[-2]   # Higher High
    lh = ph[-1] < ph[-2]   # Lower High
    hl = pl[-1] > pl[-2]   # Higher Low
    ll = pl[-1] < pl[-2]   # Lower Low

    if direction == "BUY":
        if ll and lh:
            return False, f"LL+LH downtrend (H:{ph[-2]:.0f}>{ph[-1]:.0f}, L:{pl[-2]:.0f}>{pl[-1]:.0f}) — blok BUY"
        tag = ("HH+HL uptrend" if (hh and hl) else
               ("HH" if hh else ("HL" if hl else "mixed")))
        return True, f"{tag} ✓"
    else:  # SELL
        if hh and hl:
            return False, f"HH+HL uptrend (H:{ph[-1]:.0f}>{ph[-2]:.0f}, L:{pl[-1]:.0f}>{pl[-2]:.0f}) — blok SELL"
        tag = ("LL+LH downtrend" if (ll and lh) else
               ("LL" if ll else ("LH" if lh else "mixed")))
        return True, f"{tag} ✓"


# ─────────────────────────────────────────────
# FAKE BREAKOUT DETECTOR (Upgrade: Anti-FakeBreak)
# ─────────────────────────────────────────────

def _check_fake_breakout(df: pd.DataFrame, direction: str,
                         close: float, atr: float) -> tuple[bool, str]:
    """
    Deteksi fake breakout: harga menembus swing level tapi:
      - candle sebelumnya belum konfirmasi (masih di dalam range), ATAU
      - volume jauh di bawah rata-rata (breakout tanpa partisipasi)
    Hanya aktif saat ada sinyal breakout nyata (harga keluar dari swing range 9 candle).
    Returns (is_fake, message) — is_fake=True berarti fake, jangan entry.
    """
    if len(df) < 15:
        return False, "breakout check: data kurang — skip"

    _prev9 = df.iloc[-10:-1]   # 9 candle sebelum candle terakhir
    _last_vol = float(df["Volume"].iloc[-1])  if "Volume" in df.columns else 0
    _avg_vol  = float(df["Volume"].iloc[-20:].mean()) if "Volume" in df.columns else 0

    if direction == "BUY":
        _swing_hi = float(_prev9["High"].max())
        if close <= _swing_hi:
            return False, "bukan breakout BUY — harga masih dalam range"
        # candle sebelumnya harus sudah close di atas swing (konfirmasi)
        _prev_close = float(df["Close"].iloc[-2])
        if _prev_close < _swing_hi * 0.9985:
            return True, (f"FAKE BREAKOUT BUY — close[−1]={_prev_close:.2f} "
                          f"masih di bawah swing {_swing_hi:.2f}")
        # volume konfirmasi
        if _avg_vol > 1 and _last_vol < _avg_vol * 0.70:
            return True, (f"FAKE BREAKOUT BUY — vol {_last_vol:.0f} "
                          f"< 70% avg {_avg_vol:.0f} (kurang partisipasi)")
        return False, f"breakout BUY valid (swing={_swing_hi:.2f}, vol={_last_vol:.0f})"

    else:  # SELL
        _swing_lo = float(_prev9["Low"].min())
        if close >= _swing_lo:
            return False, "bukan breakout SELL — harga masih dalam range"
        _prev_close = float(df["Close"].iloc[-2])
        if _prev_close > _swing_lo * 1.0015:
            return True, (f"FAKE BREAKOUT SELL — close[−1]={_prev_close:.2f} "
                          f"masih di atas swing {_swing_lo:.2f}")
        if _avg_vol > 1 and _last_vol < _avg_vol * 0.70:
            return True, (f"FAKE BREAKOUT SELL — vol {_last_vol:.0f} "
                          f"< 70% avg {_avg_vol:.0f} (kurang partisipasi)")
        return False, f"breakout SELL valid (swing={_swing_lo:.2f}, vol={_last_vol:.0f})"


# ─────────────────────────────────────────────
# TP / SL CALCULATION
# ─────────────────────────────────────────────

def calculate_auto_tp_sl(direction: str, close: float, atr: float,
                          df: pd.DataFrame, decimals: int = 5,
                          final_score: float = 0.0) -> dict:
    resistances, supports = _find_swing_levels(df, lookback=150, pivot_window=5)
    buf     = atr * 0.3
    tp_mult = _dynamic_rr(final_score)

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
            tp_dist   = sl_dist * tp_mult
            tp        = round(close + tp_dist, decimals)
            tp_method = f"ATR-1:{int(tp_mult)}"
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
            tp_dist   = sl_dist * tp_mult
            tp        = round(close - tp_dist, decimals)
            tp_method = f"ATR-1:{int(tp_mult)}"

    rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
    return {
        "tp": tp, "sl": sl,
        "tp_dist": round(tp_dist, decimals),
        "sl_dist": round(sl_dist, decimals),
        "rr": rr, "method_tp": tp_method, "method_sl": sl_method,
    }


def _dynamic_rr(score: float) -> float:
    # Selalu pakai fixed multiplier dari config — tidak diperlebar meski sinyal kuat
    try:
        from config import ATR_MULTIPLIER_TP
        return ATR_MULTIPLIER_TP
    except Exception:
        return 1.5


def calculate_smart_tp_sl(direction: str, close: float, atr: float,
                           df: pd.DataFrame, final_score: float,
                           decimals: int = 5) -> dict:
    if direction not in ("BUY", "SELL"):
        return {"tp": None, "sl": None, "tp_dist": 0,
                "sl_dist": 0, "rr": 0, "method_tp": "NONE", "method_sl": "NONE"}

    # Selalu pakai ATR fixed — tidak pakai swing levels
    sl_dist   = atr * ATR_MULTIPLIER_SL
    tp_mult   = ATR_MULTIPLIER_TP / ATR_MULTIPLIER_SL  # ratio fixed: TP/SL
    tp_dist   = atr * ATR_MULTIPLIER_TP
    sl_method = "ATR"
    tp_method = f"ATR-1:{int(tp_mult)}"

    if direction == "BUY":
        sl = round(close - sl_dist, decimals)
        tp = round(close + tp_dist, decimals)
    else:
        sl = round(close + sl_dist, decimals)
        tp = round(close - tp_dist, decimals)

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
# NEW: SUPERTREND SCORING
# ─────────────────────────────────────────────

def score_supertrend(row: pd.Series) -> tuple[float, str]:
    """
    Supertrend ATR-based BUY/SELL.
    Flip (sinyal baru) mendapat bobot penuh; sinyal berkelanjutan 0.6x.
    """
    w    = WEIGHTS.get("supertrend", 2.5)
    d    = int(row.get("supertrend_dir",  0))
    flip = int(row.get("supertrend_flip", 0))

    if flip == 1:
        return w,        "Supertrend FLIP → BUY (trend baru naik!)"
    elif flip == -1:
        return -w,       "Supertrend FLIP → SELL (trend baru turun!)"
    elif d == 1:
        return w * 0.6,  "Supertrend BUY (trend naik berlanjut)"
    elif d == -1:
        return -w * 0.6, "Supertrend SELL (trend turun berlanjut)"
    return 0, "Supertrend N/A"


# ─────────────────────────────────────────────
# NEW: MFI SCORING
# ─────────────────────────────────────────────

def score_mfi(row: pd.Series) -> tuple[float, str]:
    """
    Money Flow Index — volume-weighted RSI.
    < 20 oversold (BUY), > 80 overbought (SELL).
    Lebih akurat dari RSI karena mempertimbangkan volume.
    """
    mfi = float(row.get("mfi", 50))
    w   = WEIGHTS.get("mfi", 1.5)
    if mfi <= 20:
        return w,         f"MFI Oversold {mfi:.1f} — volume bearish habis (BUY)"
    elif mfi >= 80:
        return -w,        f"MFI Overbought {mfi:.1f} — volume bullish habis (SELL)"
    elif mfi <= 35:
        return w * 0.5,   f"MFI Near Oversold {mfi:.1f}"
    elif mfi >= 65:
        return -w * 0.5,  f"MFI Near Overbought {mfi:.1f}"
    return 0, f"MFI Neutral {mfi:.1f}"


# ─────────────────────────────────────────────
# NEW: PARABOLIC SAR SCORING
# ─────────────────────────────────────────────

def score_psar(row: pd.Series, close: float) -> tuple[float, str]:
    """
    Parabolic SAR:
    - SAR di bawah harga (psar_dir=1) → BUY
    - SAR di atas  harga (psar_dir=-1) → SELL
    """
    d    = int(row.get("psar_dir", 0))
    psar = float(row.get("psar", close))
    w    = WEIGHTS.get("psar", 1.5)
    dist_pct = abs(close - psar) / close * 100 if close > 0 else 0

    if d == 1:
        return w, f"PSAR BUY — SAR {psar:.2f} di bawah harga ({dist_pct:.2f}% jarak)"
    elif d == -1:
        return -w, f"PSAR SELL — SAR {psar:.2f} di atas harga ({dist_pct:.2f}% jarak)"
    return 0, "PSAR N/A"


# ─────────────────────────────────────────────
# NEW: ICHIMOKU SCORING
# ─────────────────────────────────────────────

def score_ichimoku(row: pd.Series, close: float) -> tuple[float, str]:
    """
    Ichimoku Cloud composite score:
    - Posisi harga vs cloud (atas cloud = bullish, bawah = bearish)
    - Tenkan vs Kijun cross
    - Warna cloud (green = bullish zone, red = bearish zone)
    Sinyal paling kuat: harga di atas green cloud + TK cross bullish
    """
    w         = WEIGHTS.get("ichimoku", 2.5)
    tenkan    = float(row.get("ichi_tenkan",    close))
    kijun     = float(row.get("ichi_kijun",     close))
    cloud_top = float(row.get("ichi_cloud_top", close))
    cloud_bot = float(row.get("ichi_cloud_bot", close))
    cloud_bull = int(row.get("ichi_cloud_bull",  0))
    tk_bull   = int(row.get("ichi_tk_bull",     0))
    tk_bear   = int(row.get("ichi_tk_bear",     0))

    score = 0.0
    parts = []

    # 1. Posisi harga vs cloud
    above_cloud = close > cloud_top
    below_cloud = close < cloud_bot
    in_cloud    = not above_cloud and not below_cloud

    if above_cloud:
        score += w * 0.4
        parts.append(f"Harga ATAS cloud {'hijau' if cloud_bull else 'merah'}")
    elif below_cloud:
        score -= w * 0.4
        parts.append(f"Harga BAWAH cloud {'merah' if not cloud_bull else 'hijau'}")
    else:
        parts.append("Harga di DALAM cloud (sideways)")

    # 2. Cloud color — konfirmasi trend
    if cloud_bull and above_cloud:
        score += w * 0.2
        parts.append("Cloud hijau (bullish trend)")
    elif not cloud_bull and below_cloud:
        score -= w * 0.2
        parts.append("Cloud merah (bearish trend)")

    # 3. Tenkan vs Kijun (posisi saat ini)
    if tenkan > kijun:
        score += w * 0.25
        parts.append(f"Tenkan {tenkan:.2f} > Kijun {kijun:.2f} (bullish)")
    elif tenkan < kijun:
        score -= w * 0.25
        parts.append(f"Tenkan {tenkan:.2f} < Kijun {kijun:.2f} (bearish)")

    # 4. TK Cross (sinyal terkuat — hanya 1 candle)
    if tk_bull:
        score += w * 0.4
        parts.append("Tenkan CROSS UP Kijun — sinyal BUY!")
    elif tk_bear:
        score -= w * 0.4
        parts.append("Tenkan CROSS DOWN Kijun — sinyal SELL!")

    desc = " | ".join(parts) if parts else "Ichimoku Neutral"
    return round(score, 3), desc


# ─────────────────────────────────────────────
# REGIME-AWARE DECISION ENGINE
# ─────────────────────────────────────────────

def _check_strong_candle(df: pd.DataFrame, row: pd.Series,
                          direction: str) -> tuple[bool, str]:
    """
    Validates price-action candle patterns.
    Required: engulfing, pin bar (rejection), or strong momentum candle.
    No valid pattern = NO TRADE.
    """
    close      = float(row.get("Close", 0))
    open_      = float(row.get("Open",  0))
    high       = float(row.get("High",  0))
    low        = float(row.get("Low",   0))
    full_range = (high - low) or 0.0001
    body       = abs(close - open_)
    wick_up    = high - max(open_, close)
    wick_down  = min(open_, close) - low
    body_ratio = body / full_range

    # 1. Extra multi-candle patterns (Three Soldiers / Morning Star etc.)
    ex_pat = int(row.get("candle_ex", 0))
    if direction == "BUY" and ex_pat >= 1:
        return True, f"Multi-candle bullish pattern ✓"
    if direction == "SELL" and ex_pat <= -1:
        return True, f"Multi-candle bearish pattern ✓"

    # 2. Engulfing
    if len(df) >= 2:
        prev = df.iloc[-2]
        pc   = float(prev.get("Close", close))
        po   = float(prev.get("Open",  close))
        if direction == "BUY" and pc < po:          # prev bearish
            if close > po and open_ < pc:           # fully engulf
                return True, "Bullish Engulfing ✓"
        if direction == "SELL" and pc > po:         # prev bullish
            if close < po and open_ > pc:           # fully engulf
                return True, "Bearish Engulfing ✓"

    # 3. Pin Bar (rejection wick ≥ 60% of range, small body ≤ 30%)
    if direction == "BUY" and wick_down / full_range >= 0.60 and body_ratio <= 0.30:
        return True, f"Bullish Pin Bar (wick_dn {wick_down/full_range:.0%}) ✓"
    if direction == "SELL" and wick_up / full_range >= 0.60 and body_ratio <= 0.30:
        return True, f"Bearish Pin Bar (wick_up {wick_up/full_range:.0%}) ✓"

    # 4. Strong momentum candle / breakout candle (body ≥ 60%, closes on correct side)
    if direction == "BUY" and close > open_ and body_ratio >= 0.60:
        return True, f"Strong Bullish Candle (body {body_ratio:.0%}) ✓"
    if direction == "SELL" and close < open_ and body_ratio >= 0.60:
        return True, f"Strong Bearish Candle (body {body_ratio:.0%}) ✓"

    # 5. SMC liquidity sweep / ChoCH — pattern override
    liq_ok = (direction == "BUY"  and int(row.get("liq_bull_sweep", 0))) or \
             (direction == "SELL" and int(row.get("liq_bear_sweep", 0)))
    choch_ok = (direction == "BUY"  and int(row.get("choch_bull", 0))) or \
               (direction == "SELL" and int(row.get("choch_bear", 0)))
    if liq_ok:
        return True, "SMC Liquidity Sweep — pattern override ✓"
    if choch_ok:
        return True, "SMC ChoCH — structure flip override ✓"

    dir_lbl = "bullish" if direction == "BUY" else "bearish"
    return False, f"No valid {dir_lbl} pattern (need engulfing/pin-bar/strong-body)"


def _make_decision(df: pd.DataFrame, row: pd.Series, close: float,
                   news_bias: dict | None,
                   news_risk: str = "LOW") -> tuple[str, list, float, dict]:
    """
    Hard-filter decision pipeline.
    Behaves like a disciplined professional trader:
    - Prefer WAIT over forced trades
    - Never trade without multiple confirmations
    - All filters are mandatory (no soft overrides)
    """
    reasons           = []
    filters           = {}
    news_contribution = 0.0

    # ── Core indicator values ─────────────────────────────────────────
    ema20  = float(row.get(f"ema_{EMA_SLOW}",   close))
    ema50  = float(row.get(f"ema_{EMA_TREND}",  close))
    ema200 = float(row.get(f"ema_{EMA_LONG}",   close))
    rsi    = float(row.get("rsi", 50))
    adx    = float(row.get("adx",  0))
    atr    = float(row.get("atr",  close * 0.001))

    # SMC structural events (strongest override)
    choch_bull = int(row.get("choch_bull",      0))
    choch_bear = int(row.get("choch_bear",      0))
    liq_bull   = int(row.get("liq_bull_sweep",  0))
    liq_bear   = int(row.get("liq_bear_sweep",  0))
    smc_force  = choch_bull or choch_bear or liq_bull or liq_bear
    smc_dir    = ("BUY"  if (choch_bull or liq_bull)  else
                  "SELL" if (choch_bear or liq_bear)  else None)

    # News pre-compute
    _nb_bias  = (news_bias or {}).get("bias", "NEUTRAL")
    _nb_score = abs((news_bias or {}).get("score", 0.0))
    _nb_conf  = (news_bias or {}).get("confidence", "LOW")
    _news_dir = ("BUY"  if _nb_bias == "BULLISH" else
                 "SELL" if _nb_bias == "BEARISH"  else None)
    if news_bias:
        reasons.append((0, f"News Bias {_nb_bias} ({news_bias.get('score', 0):+.1f}) [hint]"))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 1 — Market State (TREND / UNCLEAR / RANGE)
    # Only trade in clear TREND; RANGE and UNCLEAR = NO TRADE
    # Exception: strong SMC event can still trigger in UNCLEAR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if adx >= ADX_TREND_MIN:          # 25+
        market_state = "TREND"
    elif adx >= 20:
        market_state = "UNCLEAR"
    else:
        market_state = "RANGE"

    filters["market_state"] = f"{market_state}  ADX={adx:.1f}"

    if market_state == "RANGE":
        filters["market_state"] += " → NO TRADE"
        reasons.append((0, f"Market RANGE (ADX {adx:.1f} < 20) — skip"))
        return "WAIT", reasons, news_contribution, filters

    if market_state == "UNCLEAR" and not smc_force:
        filters["market_state"] += " → NO TRADE (no SMC)"
        reasons.append((0, f"Market UNCLEAR (ADX {adx:.1f}) — terlalu lemah, no SMC"))
        return "WAIT", reasons, news_contribution, filters

    filters["market_state"] += " ✓" if market_state == "TREND" else " (SMC override)"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 2 — Trend Direction (EMA50 vs EMA200 = macro trend)
    # Counter-trend trades are FORBIDDEN
    # ChoCH can override (genuine structure flip)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ema50 > ema200 * 1.0001:
        macro_trend = "BUY"
    elif ema50 < ema200 * 0.9999:
        macro_trend = "SELL"
    else:
        macro_trend = None   # EMA50/200 flat

    # ChoCH overrides macro trend (institutional structure flip)
    if choch_bull:
        macro_trend = "BUY"
    elif choch_bear:
        macro_trend = "SELL"

    if macro_trend is None:
        filters["trend"] = f"FLAT — EMA50={ema50:.2f} ≈ EMA200={ema200:.2f} → NO TRADE"
        reasons.append((0, "Macro trend flat (EMA50≈EMA200) — tidak ada arah dominan"))
        return "WAIT", reasons, news_contribution, filters

    # Short-term direction (EMA20 vs EMA50) confirms entry timing
    short_trend = "BUY" if ema20 > ema50 else "SELL" if ema20 < ema50 else macro_trend

    # SMC liquidity sweep can trade against macro when very strong
    if smc_force and smc_dir:
        raw_dir = smc_dir
        if smc_dir != macro_trend:
            filters["trend"] = (f"Counter-macro {smc_dir} vs macro {macro_trend} "
                                f"— SMC override (liq.sweep/ChoCH)")
            reasons.append((0, f"SMC {smc_dir} override macro {macro_trend}"))
        else:
            filters["trend"] = f"Macro {macro_trend} + SMC {smc_dir} aligned ✓"
    else:
        # No SMC: must align with BOTH macro and short trend
        if short_trend != macro_trend:
            filters["trend"] = (f"COUNTER-TREND — short={short_trend} vs macro={macro_trend}"
                                f" → NO TRADE")
            reasons.append((0, f"Short trend {short_trend} berlawanan macro {macro_trend} — blok"))
            return "WAIT", reasons, news_contribution, filters
        raw_dir = macro_trend
        filters["trend"] = f"Trend {raw_dir} — EMA20/50/200 aligned ✓"

    reasons.append((0, f"Trend {raw_dir} → arah dikonfirmasi"))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 3 — News
    # HIGH impact = NO TRADE (unpredictable volatility)
    # News bias opposing trend = NO TRADE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if news_risk == "HIGH" and NEWS_HIGH_BLOCK:
        filters["news"] = "HIGH impact news → NO TRADE"
        reasons.append((-1.0, "News HIGH — entry dilarang (terlalu volatile)"))
        return "WAIT", reasons, news_contribution, filters

    if _nb_bias != "NEUTRAL" and _nb_score >= 2:
        if _news_dir and _news_dir != raw_dir:
            filters["news"] = f"News {_nb_bias} berlawanan trend {raw_dir} → NO TRADE"
            reasons.append((0, f"News {_nb_bias} berlawanan {raw_dir} — blok"))
            return "WAIT", reasons, news_contribution, filters
        if _news_dir == raw_dir:
            boost = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}.get(_nb_conf, 0.5)
            news_contribution = boost
            filters["news"]   = f"News {_nb_bias} searah {raw_dir} → +{boost:.1f} ✓"
            reasons.append((boost, f"News {_nb_bias} searah {raw_dir} → +{boost:.1f}"))
        else:
            filters["news"] = f"{news_risk} neutral — OK"
    else:
        filters["news"] = f"{news_risk} — OK"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 4 — RSI Momentum
    # BUY only if RSI > 50 (upward momentum confirmed)
    # SELL only if RSI < 50 (downward momentum confirmed)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    rsi_aligned = (raw_dir == "BUY" and rsi > 50) or (raw_dir == "SELL" and rsi < 50)
    liq_rsi_override = (raw_dir == "BUY" and liq_bull) or (raw_dir == "SELL" and liq_bear)

    if not rsi_aligned and not liq_rsi_override:
        filters["momentum"] = (f"RSI {rsi:.1f} — "
                               f"{'< 50 (need >50 for BUY)' if raw_dir == 'BUY' else '> 50 (need <50 for SELL)'}"
                               f" → NO TRADE")
        reasons.append((0, f"RSI {rsi:.1f} tidak mendukung {raw_dir}"))
        return "WAIT", reasons, news_contribution, filters

    if liq_rsi_override and not rsi_aligned:
        filters["momentum"] = f"RSI {rsi:.1f} — Liq.Sweep override ✓"
    else:
        filters["momentum"] = f"RSI {rsi:.1f} ({'> 50' if raw_dir == 'BUY' else '< 50'}) ✓"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 5 — Candle Pattern (price action confirmation)
    # Must have: engulfing, pin bar, strong body, or SMC event
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ok_candle, candle_msg = _check_strong_candle(df, row, raw_dir)
    filters["candle_pattern"] = candle_msg
    if not ok_candle:
        reasons.append((0, f"No valid candle: {candle_msg}"))
        return "WAIT", reasons, news_contribution, filters
    reasons.append((+1 if raw_dir == "BUY" else -1, f"Candle: {candle_msg}"))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 6 — No Trade Zone (near key S/R levels)
    # BUY near resistance = bad risk/reward → NO TRADE
    # SELL near support   = bad risk/reward → NO TRADE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _zone_tol = atr * NO_TRADE_ZONE_PCT
    _res, _sup = _find_swing_levels(df, lookback=80, pivot_window=4)
    if raw_dir == "BUY" and _res:
        too_close = [r for r in _res if 0 < r - close <= _zone_tol]
        if too_close:
            filters["no_trade_zone"] = (f"NO TRADE — BUY terlalu dekat resistance "
                                        f"{too_close[0]:.2f} (dist {too_close[0]-close:.2f})")
            reasons.append((0, f"No Trade Zone: BUY dekat resistance {too_close[0]:.2f}"))
            return "WAIT", reasons, news_contribution, filters
    elif raw_dir == "SELL" and _sup:
        too_close = [s for s in _sup if 0 < close - s <= _zone_tol]
        if too_close:
            filters["no_trade_zone"] = (f"NO TRADE — SELL terlalu dekat support "
                                        f"{too_close[0]:.2f} (dist {close-too_close[0]:.2f})")
            reasons.append((0, f"No Trade Zone: SELL dekat support {too_close[0]:.2f}"))
            return "WAIT", reasons, news_contribution, filters
    filters["no_trade_zone"] = "OK — level jelas ✓"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 7 — Volatility Filter (Upgrade 4)
    # ATR terlalu rendah = market sepi, spread lebih berdampak, sinyal palsu
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from config import MIN_ATR
    if MIN_ATR > 0 and atr < MIN_ATR:
        filters["volatility"] = f"LOW ATR={atr:.2f} < {MIN_ATR} — market sepi → NO TRADE"
        reasons.append((0, f"Volatility filter: ATR {atr:.2f} < min {MIN_ATR}"))
        return "WAIT", reasons, news_contribution, filters
    filters["volatility"] = f"ATR={atr:.2f} ✓"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 8 — Session Filter Pro (Upgrade 6)
    # XAUUSD paling aktif dan reliable saat London & New York session
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from config import SESSION_FILTER, LONDON_OPEN_UTC, LONDON_CLOSE_UTC, NY_OPEN_UTC, NY_CLOSE_UTC
    if SESSION_FILTER:
        from datetime import datetime as _dt2, timezone as _tz2
        _utc_h     = _dt2.now(tz=_tz2.utc).hour
        _in_london = LONDON_OPEN_UTC <= _utc_h < LONDON_CLOSE_UTC
        _in_ny     = NY_OPEN_UTC     <= _utc_h < NY_CLOSE_UTC
        if not (_in_london or _in_ny):
            _sess = ("Asian" if 0 <= _utc_h < 7 else "Off-Hours")
            filters["session"] = f"NO TRADE — {_sess} session (UTC {_utc_h:02d}:xx, bukan London/NY)"
            reasons.append((0, f"Session filter: UTC {_utc_h:02d}:xx bukan London/NY — WAIT"))
            return "WAIT", reasons, news_contribution, filters
        _sess_label = ("London+NY Overlap" if (_in_london and _in_ny) else
                       ("London" if _in_london else "New York"))
        filters["session"] = f"{_sess_label} session (UTC {_utc_h:02d}:xx) ✓"
    else:
        filters["session"] = "session filter OFF"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 9 — Entry Zone / Sniper Entry (Upgrade 2)
    # BUY hanya jika dekat support, SELL hanya jika dekat resistance
    # Mencegah entry di tengah nowhere (RR buruk)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from config import ENTRY_ZONE_PCT
    if ENTRY_ZONE_PCT > 0:
        _entry_tol = atr * ENTRY_ZONE_PCT
        if raw_dir == "BUY" and _sup:
            _near_sup = [s for s in _sup if 0 < close - s <= _entry_tol]
            if not _near_sup:
                _nearest = min(_sup, key=lambda s: abs(close - s))
                _dist    = abs(close - _nearest)
                filters["entry_zone"] = (f"NO ENTRY — BUY jauh dari support "
                                         f"(nearest={_nearest:.2f}, dist={_dist:.2f}, tol={_entry_tol:.2f})")
                reasons.append((0, f"Entry Zone: support {_nearest:.2f} terlalu jauh ({_dist:.2f} > {_entry_tol:.2f})"))
                return "WAIT", reasons, news_contribution, filters
            filters["entry_zone"] = (f"BUY near support {_near_sup[0]:.2f} "
                                     f"(dist={close-_near_sup[0]:.2f} <= {_entry_tol:.2f}) ✓")
        elif raw_dir == "SELL" and _res:
            _above_res = [r for r in _res if r > close]
            _near_res  = [r for r in _above_res if r - close <= _entry_tol]
            if not _near_res:
                _nearest = min(_above_res, key=lambda r: r - close) if _above_res else None
                _dist    = (_nearest - close) if _nearest else 9999
                filters["entry_zone"] = (f"NO ENTRY — SELL jauh dari resistance "
                                         f"(nearest={_nearest:.2f}, dist={_dist:.2f}, tol={_entry_tol:.2f})"
                                         if _nearest else "NO ENTRY — tidak ada resistance di atas")
                reasons.append((0, f"Entry Zone: resistance terlalu jauh ({_dist:.2f} > {_entry_tol:.2f})"))
                return "WAIT", reasons, news_contribution, filters
            filters["entry_zone"] = (f"SELL near resistance {_near_res[0]:.2f} "
                                     f"(dist={_near_res[0]-close:.2f} <= {_entry_tol:.2f}) ✓")
        else:
            filters["entry_zone"] = "entry zone: no levels found — skip check"
    else:
        filters["entry_zone"] = "entry zone filter OFF"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 10 — Market Structure Real (Upgrade 9)
    # Verifikasi HH/HL untuk BUY, LL/LH untuk SELL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _ok_struct, _struct_msg = _check_market_structure(df, raw_dir, lookback=20)
    filters["market_structure"] = _struct_msg
    if not _ok_struct:
        reasons.append((0, f"Market Structure: {_struct_msg}"))
        return "WAIT", reasons, news_contribution, filters

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HARD FILTER 11 — Fake Breakout Detector (Upgrade)
    # Breakout tanpa volume = stop hunt / liquidity grab → WAIT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _is_fake, _fake_msg = _check_fake_breakout(df, raw_dir, close, atr)
    filters["fake_breakout"] = _fake_msg
    if _is_fake:
        reasons.append((0, f"Fake Breakout: {_fake_msg}"))
        return "WAIT", reasons, news_contribution, filters

    reasons.append((+1 if raw_dir == "BUY" else -1, f"Semua hard-filter passed → {raw_dir}"))
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

    confidence   = 0   # will be filled in confidence scoring block below
    conf_notes   = []

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

    # ── New: Supertrend + MFI + PSAR + Ichimoku ───────────────────────
    s_st,   d_st   = score_supertrend(row)
    s_mfi,  d_mfi  = score_mfi(row)
    s_psar, d_psar = score_psar(row, close)
    s_ichi, d_ichi = score_ichimoku(row, close)

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

    total_vol   = (s_obv + s_vwap + s_wr + s_cci + s_vctx + s_mfi) * vol_mult
    total_smc   = (s_smc + s_expat) * smc_mult
    total_str   = (s_rdiv + s_mch + s_sma + s_fib + s_st + s_psar + s_ichi) * str_mult
    max_trad    = sum(WEIGHTS[k] for k in ["rsi","macd","ema_cross","bb","stoch","adx","candle"])

    # ── Session Behavior Adaptive (Upgrade) ──────────────────────────
    # London  → boost trend-following (supertrend/ichimoku/structure)
    # New York → boost breakout (SMC/volume/momentum)
    # Overlap  → boost keduanya
    from datetime import datetime as _dt_sb, timezone as _tz_sb
    from config import (LONDON_OPEN_UTC as _LON_O, LONDON_CLOSE_UTC as _LON_C,
                        NY_OPEN_UTC     as _NY_O,  NY_CLOSE_UTC      as _NY_C)
    _utc_h_sb  = _dt_sb.now(tz=_tz_sb.utc).hour
    _in_lon_sb = _LON_O <= _utc_h_sb < _LON_C
    _in_ny_sb  = _NY_O  <= _utc_h_sb < _NY_C

    if _in_lon_sb and _in_ny_sb:
        _session_name_sb = "London+NY Overlap"
        total_str *= 1.15   # kedua strategi aktif
        total_smc *= 1.15
        total_vol *= 1.10
    elif _in_lon_sb:
        _session_name_sb = "London"
        total_str *= 1.25   # trend-follow: supertrend, ichimoku, structure
        total_vol *= 1.10
    elif _in_ny_sb:
        _session_name_sb = "New York"
        total_smc *= 1.25   # breakout: SMC, order block, liquidity
        total_vol *= 1.20   # volume + momentum lebih penting di NY
    else:
        _session_name_sb = "Other"   # outside London/NY → session filter sudah blok di Hard Filter 8
    filters["session_strategy"] = f"Strategy: {_session_name_sb}"

    # ── EMA short-trend alignment dampening ──────────────────────────
    # Jika EMA20 > EMA50 (short uptrend): MACD & RSI sering SELL palsu
    # saat koreksi kecil → dampen kontribusi mereka
    _ema20_now = float(row.get(f"ema_{EMA_SLOW}",  close))
    _ema50_now = float(row.get(f"ema_{EMA_TREND}", close))
    _ema_short_up   = _ema20_now > _ema50_now   # True = short uptrend
    _ema_short_down = _ema20_now < _ema50_now   # True = short downtrend

    # Hitung normalized_trad awal sebelum dampening (untuk Quality Gate di bawah)
    total_trad_raw  = sum(trad_scores) * trad_mult
    normalized_trad = (total_trad_raw / max_trad) * 10 if max_trad > 0 else 0

    # MACD histogram score (dari score_macd)
    _macd_score = trad_scores[1]   # index 1 = score_macd
    _rsi_score  = trad_scores[0]   # index 0 = score_rsi

    # Kalau EMA bilang naik tapi MACD/RSI bilang turun = pullback noise
    # Kurangi bobotnya HANYA jika sinyal kita BERLAWANAN dengan indikator tsb
    # (bug lama: dampening berlaku untuk semua sinyal, termasuk SMC counter-trend
    #  yang justru BUTUH MACD/RSI bearish sebagai konfirmasi SELL-nya)
    _dampen_trad = 1.0
    if _ema_short_up and _macd_score < 0 and direction == "BUY":
        # MACD bearish tapi kita BUY di uptrend → itu pullback noise, bukan reversal
        trad_scores[1] = _macd_score * 0.3   # dampen 70%
        _dampen_trad = 0.9
        filters["trad_dampen"] = "MACD bearish di EMA uptrend (BUY) — dikurangi 70%"
    if _ema_short_up and _rsi_score < 0 and direction == "BUY":
        trad_scores[0] = _rsi_score * 0.4    # dampen 60%
        _dampen_trad = min(_dampen_trad, 0.9)
        filters["trad_dampen"] = filters.get("trad_dampen","") + " | RSI bearish di EMA uptrend (BUY) — dikurangi 60%"
    if _ema_short_down and _macd_score > 0 and direction == "SELL":
        # MACD bullish tapi kita SELL di downtrend → itu pullback noise
        trad_scores[1] = _macd_score * 0.3
        _dampen_trad = 0.9
        filters["trad_dampen"] = "MACD bullish di EMA downtrend (SELL) — dikurangi 70%"
    if _ema_short_down and _rsi_score > 0 and direction == "SELL":
        trad_scores[0] = _rsi_score * 0.4
        _dampen_trad = min(_dampen_trad, 0.9)

    # Recalculate trad score setelah dampen
    total_trad = sum(trad_scores) * trad_mult * _dampen_trad

    # Signal Quality Gate: jika contradicting signals, dampen score
    # RSI divergence opposing EMA direction → reduce confidence
    if s_rdiv > 0 and normalized_trad < -2:
        total_str *= 0.5
    elif s_rdiv < 0 and normalized_trad > 2:
        total_str *= 0.5

    # Recalculate normalized trad setelah dampen
    max_trad        = sum(WEIGHTS[k] for k in ["rsi","macd","ema_cross","bb","stoch","adx","candle"])
    normalized_trad = (total_trad / max_trad) * 10 if max_trad > 0 else 0

    # Score murni teknikal — news tidak masuk score
    final_score = round(
        normalized_trad + total_vol + total_smc + total_str + memory_contribution,
        2
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ANTI-CONFLICT ENGINE
    # Hitung vote bullish vs bearish dari semua indikator.
    # Jika > 40% indikator berlawanan dengan arah sinyal → WAIT
    # Bot tidak boleh "memilih" ketika pasar sendiri bingung.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if direction in ("BUY", "SELL"):
        _all_scores = [
            # Traditional
            trad_scores[0],   # rsi
            trad_scores[1],   # macd
            trad_scores[2],   # stoch
            trad_scores[4],   # candle
            trad_scores[5],   # ema
            # Volume + structure
            s_obv, s_vwap, s_mfi,
            # Trend
            s_st, s_ichi, s_psar,
            # Structure
            s_mch, s_rdiv,
            # SMC
            s_smc,
        ]
        _nonzero = [s for s in _all_scores if s != 0]
        if _nonzero:
            _bull_v = sum(1 for s in _nonzero if s > 0)
            _bear_v = sum(1 for s in _nonzero if s < 0)
            _total_v = _bull_v + _bear_v
            _minority = min(_bull_v, _bear_v)
            _conflict_ratio = _minority / _total_v if _total_v > 0 else 0

            filters["conflict"] = (
                f"Bull:{_bull_v} Bear:{_bear_v} "
                f"conflict={_conflict_ratio:.0%}"
            )

            # Blok jika > 40% indikator berlawanan dengan arah
            if direction == "BUY" and _bear_v / _total_v > 0.40:
                direction = "WAIT"
                reasons.append((0, f"Anti-Conflict: {_bear_v}/{_total_v} indikator BEARISH "
                                   f"— terlalu banyak konflik untuk BUY"))
                filters["conflict"] += " → BLOK"
            elif direction == "SELL" and _bull_v / _total_v > 0.40:
                direction = "WAIT"
                reasons.append((0, f"Anti-Conflict: {_bull_v}/{_total_v} indikator BULLISH "
                                   f"— terlalu banyak konflik untuk SELL"))
                filters["conflict"] += " → BLOK"
            else:
                filters["conflict"] += " ✓"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SIMPLIFIED CONFIDENCE SCORING (max 6 here, +1 ML in bot.py = 7)
    # +2 trend aligned  |  +2 valid candle  |  +1 key level
    # +1 strong momentum (RSI > 55 BUY / < 45 SELL)  |  (ML: +1 bot.py)
    # Threshold: MIN_QUALITY_SCORE (default 4)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    confidence = 0
    conf_notes = []

    if direction in ("BUY", "SELL"):
        _e20  = float(row.get(f"ema_{EMA_SLOW}",  close))
        _e50  = float(row.get(f"ema_{EMA_TREND}", close))
        _e200 = float(row.get(f"ema_{EMA_LONG}",  close))

        # +2: Trend aligned (macro + short)
        _macro_ok  = (direction == "BUY"  and _e50 > _e200) or \
                     (direction == "SELL" and _e50 < _e200)
        _short_ok  = (direction == "BUY"  and _e20 > _e50)  or \
                     (direction == "SELL" and _e20 < _e50)
        if _macro_ok and _short_ok:
            confidence += 2
            conf_notes.append("trend✓")
        elif _macro_ok or _short_ok:
            confidence += 1
            conf_notes.append("trend~")

        # +2: Valid candle pattern passed the hard filter (already validated above)
        # Candle confirmed = _make_decision would have returned WAIT if not
        confidence += 2
        conf_notes.append("candle✓")

        # +1: Near key level — Fib support/resistance or SMC zone aligned
        _smc_ok  = (s_smc > 0.5 and direction == "BUY")  or \
                   (s_smc < -0.5 and direction == "SELL")
        _fib_ok  = (s_fib > 0   and direction == "BUY")  or \
                   (s_fib < 0   and direction == "SELL")
        _rdiv_ok = (s_rdiv > 0  and direction == "BUY")  or \
                   (s_rdiv < 0  and direction == "SELL")
        if _smc_ok or _fib_ok or _rdiv_ok:
            confidence += 1
            conf_notes.append("level✓")

        # +1: Strong momentum (RSI > 55 for BUY, < 45 for SELL)
        _rsi_now = float(row.get("rsi", 50))
        if (direction == "BUY" and _rsi_now > 55) or (direction == "SELL" and _rsi_now < 45):
            confidence += 1
            conf_notes.append("momentum✓")

        filters["confidence"] = f"{confidence}/6 ({' '.join(conf_notes)})"

        if confidence < MIN_QUALITY_SCORE:
            direction = "WAIT"
            reasons.append((0, f"Confidence {confidence}/6 < min {MIN_QUALITY_SCORE} "
                              f"— {' '.join(conf_notes) or 'insufficient confirms'}"))
        else:
            reasons.append((0, f"Confidence {confidence}/6 ✓ ({' '.join(conf_notes)})"))

    # Score floor — pakai adaptive threshold jika tersedia
    # Cap: adaptive tidak boleh melebihi MIN_SIGNAL_SCORE + 1.5
    # (mencegah adaptive memblok semua sinyal saat win rate rendah)
    _min_score = MIN_SIGNAL_SCORE
    try:
        from ai.adaptive import get_learner
        _adaptive_score = get_learner().min_score
        _min_score = min(_adaptive_score, MIN_SIGNAL_SCORE + 1.5)
    except Exception:
        pass
    if direction in ("BUY", "SELL") and abs(final_score) < _min_score:
        direction = "WAIT"
        reasons.append((0, f"Score {final_score:.2f} < {_min_score:.1f} — signal terlalu lemah"))

    # ── Counter-Trend Gate: butuh 2x score jika melawan EMA short trend ──────
    # Contoh: EMA20 > EMA50 (naik) tapi signal SELL → itu koreksi, bukan reversal
    # Butuh bukti kuat (2x min score) sebelum melawan trend
    if direction in ("BUY", "SELL"):
        _e20_ct = float(row.get(f"ema_{EMA_SLOW}",  close))
        _e50_ct = float(row.get(f"ema_{EMA_TREND}", close))
        _short_tr = "BUY" if _e20_ct > _e50_ct else ("SELL" if _e20_ct < _e50_ct else None)

        if _short_tr and direction != _short_tr:
            _counter_min = _min_score * 2.0   # 5.0 → 10.0 untuk counter-trend
            if abs(final_score) < _counter_min:
                reasons.append((0, f"Counter short-trend ({direction} vs EMA {_short_tr}) "
                                   f"— score {final_score:.1f} < {_counter_min:.1f} (2x min) → WAIT"))
                filters["counter_trend"] = (f"BLOK — {direction} vs EMA {_short_tr}, "
                                            f"score {final_score:.1f} perlu {_counter_min:.1f}")
                direction = "WAIT"
            else:
                reasons.append((0, f"Counter short-trend ({direction} vs EMA {_short_tr}) "
                                   f"— score {final_score:.1f} >= {_counter_min:.1f} OK (reversal kuat)"))
                filters["counter_trend"] = (f"LOLOS — counter-trend tapi score kuat "
                                            f"{final_score:.1f} >= {_counter_min:.1f}")

    # ── Session Bias Filter ───────────────────────────────────────────────────
    # Bias dari analisis jam tutup sesi: hanya izinkan arah yang searah bias HTF
    try:
        from data.session_bias import get_current_bias
        _bias = get_current_bias()
        if _bias and _bias.get("direction") in ("BUY", "SELL"):
            _bias_dir = _bias["direction"]
            _bias_str = _bias.get("strength", "")
            _bias_sco = _bias.get("score", 0)
            if direction in ("BUY", "SELL") and direction != _bias_dir:
                # Hanya block jika bias kuat (score > 3) dan signal lemah (score < 7)
                if abs(_bias_sco) >= 3.0 and abs(final_score) < 7.0:
                    reasons.append((0, f"Session Bias: {_bias_dir} ({_bias_str}) "
                                       f"berlawanan dengan signal — WAIT"))
                    direction = "WAIT"
            elif direction in ("BUY", "SELL") and direction == _bias_dir:
                reasons.append((0, f"Session Bias: {_bias_dir} ({_bias_str}) "
                                   f"SEARAH — boost konfirmasi"))
    except Exception:
        pass

    # Block RANGE regime tanpa SMC atau RSI divergence
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
    if s_mfi != 0:
        reasons.append((s_mfi, d_mfi))
    if s_st != 0:
        reasons.append((s_st, d_st))
    if s_psar != 0:
        reasons.append((s_psar, d_psar))
    if s_ichi != 0:
        reasons.append((s_ichi, d_ichi))

    # ── Regime info ────────────────────────────────────────────────────
    filters["regime_score"] = (
        f"Regime={regime} | Trad×{trad_mult} SMC×{smc_mult} Vol×{vol_mult} Str×{str_mult} "
        f"[ST:{s_st:+.1f} PSAR:{s_psar:+.1f} Ichi:{s_ichi:+.1f} MFI:{s_mfi:+.1f}]"
    )

    tp_sl = calculate_smart_tp_sl(direction, close, atr, df, final_score)

    if direction in ("BUY", "SELL") and tp_sl["rr"] < MIN_RR_RATIO:
        filters["rr"] = f"REJECT — RR {tp_sl['rr']} < {MIN_RR_RATIO}"
        direction = "WAIT"
    elif direction in ("BUY", "SELL"):
        filters["rr"] = f"OK — RR {tp_sl['rr']}"

    # ── Market state for output ──────────────────────────────────────────
    _adx_val     = float(row.get("adx", 0))
    _market_state = ("TREND" if _adx_val >= ADX_TREND_MIN else
                     "UNCLEAR" if _adx_val >= 20 else "RANGE")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SIGNAL STRENGTH SCORING (Upgrade)
    # Mengukur kekuatan setup: 8 poin maks
    #   STRONG (6+) → entry langsung (no delay)
    #   MEDIUM (4-5) → tunggu 1 candle konfirmasi
    #   WEAK   (<4) → skip entry
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _spts = 0
    if direction in ("BUY", "SELL"):
        # +2: EMA fully stacked + ADX >= 30 (tren kuat)
        _e9_s  = float(row.get(f"ema_{EMA_FAST}",  close))
        _e20_s = float(row.get(f"ema_{EMA_SLOW}",  close))
        _e50_s = float(row.get(f"ema_{EMA_TREND}", close))
        _ema_stack = ((direction == "BUY"  and _e9_s > _e20_s > _e50_s) or
                      (direction == "SELL" and _e9_s < _e20_s < _e50_s))
        if _adx_val >= 30 and _ema_stack:
            _spts += 2
        elif _adx_val >= 25 or _ema_stack:
            _spts += 1

        # +2: Liquidity sweep / SMC event kuat
        if (s_smc > 1.5 and direction == "BUY") or (s_smc < -1.5 and direction == "SELL"):
            _spts += 2
        elif abs(s_smc) > 0.5:
            _spts += 1

        # +2: Confidence (candle + struktur sudah konfirmasi)
        if confidence >= 5:
            _spts += 2
        elif confidence >= 4:
            _spts += 1

        # +1: Near key level (Fibonacci atau RSI Divergence)
        _fib_ok_s  = (s_fib  > 0 and direction == "BUY") or (s_fib  < 0 and direction == "SELL")
        _rdiv_ok_s = (s_rdiv > 0 and direction == "BUY") or (s_rdiv < 0 and direction == "SELL")
        if _fib_ok_s or _rdiv_ok_s:
            _spts += 1

        # +1: RSI momentum (RSI > 55 BUY / < 45 SELL)
        _rsi_s = float(row.get("rsi", 50))
        if (direction == "BUY" and _rsi_s > 55) or (direction == "SELL" and _rsi_s < 45):
            _spts += 1

    if _spts >= 6:
        signal_strength = "STRONG"
    elif _spts >= 4:
        signal_strength = "MEDIUM"
    else:
        signal_strength = "WEAK"
    filters["signal_strength"] = f"{signal_strength} ({_spts}/8 pts)"

    return {
        "direction":        direction,
        "score":            final_score,
        "confidence":       confidence,           # 0-6 quality score
        "signal_strength":  signal_strength,      # STRONG / MEDIUM / WEAK
        "strength_pts":     _spts,                # raw points 0-8
        "market_state":     _market_state,        # TREND / UNCLEAR / RANGE
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
        "market_state":    "  ",
        "trend":           "  ",
        "news":            "  ",
        "momentum":        "  ",
        "candle_pattern":  "  ",
        "no_trade_zone":   "  ",
        "confidence":      "  ",
        "rr":              "  ",
        "regime":          "  ",
        "regime_score":    "  ",
        "candle_trend":    "  ",
        "signal_strength": "⚡ ",
        "session_strategy":"  ",
        "fake_breakout":   "  ",
    }
    for key, val in filters.items():
        icon   = icons.get(key, "  ")
        passed = not any(x in str(val) for x in
                         ["SKIP", "WEAK", "REJECT", "NO TRADE", "CANCEL", "COUNTER",
                          "FAKE", "belum", "tunggu", "berlawanan", "FLAT"])
        color  = GREEN if passed else (RED if "NO TRADE" in str(val) or "REJECT" in str(val)
                                       or "FAKE" in str(val) else YELLOW)
        print(f"    {icon} {key:<18}: {color}{val}{RESET}")
