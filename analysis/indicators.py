import pandas as pd
import numpy as np
from config import *


# ─────────────────────────────────────────────
# TRADITIONAL INDICATORS
# ─────────────────────────────────────────────

def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(close: pd.Series) -> pd.DataFrame:
    ema_fast  = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow  = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal    = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal
    return pd.DataFrame({"macd": macd_line, "signal": signal, "histogram": histogram})


def calculate_bollinger_bands(close: pd.Series) -> pd.DataFrame:
    sma   = close.rolling(BB_PERIOD).mean()
    std   = close.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    bw    = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_upper": upper, "bb_mid": sma,
        "bb_lower": lower, "bb_bw": bw, "bb_pct": pct_b,
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


def calculate_smas(close: pd.Series) -> pd.DataFrame:
    """Simple Moving Averages — 10, 20, 50, 200 period."""
    return pd.DataFrame({
        "sma10":  close.rolling(10).mean(),
        "sma20":  close.rolling(20).mean(),
        "sma50":  close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
    })


def calculate_fibonacci_levels(high: pd.Series, low: pd.Series,
                                lookback: int = 50) -> pd.DataFrame:
    """
    Dynamic Fibonacci Retracement levels berdasarkan swing high/low N candle terakhir.
    Level klasik: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%

    Bullish retracement: pullback dari high ke low → area beli di level fib support
    Bearish retracement: bounce dari low ke high → area jual di level fib resistance
    """
    swing_high = high.rolling(lookback, min_periods=10).max()
    swing_low  = low.rolling(lookback, min_periods=10).min()
    rng        = (swing_high - swing_low).replace(0, np.nan)

    fib_236 = swing_high - rng * 0.236
    fib_382 = swing_high - rng * 0.382
    fib_500 = swing_high - rng * 0.500
    fib_618 = swing_high - rng * 0.618
    fib_786 = swing_high - rng * 0.786

    # Deteksi apakah harga mendekati level fib (dalam 0.5% ATR)
    # Akan digunakan signals.py untuk scoring
    return pd.DataFrame({
        "fib_swing_high": swing_high,
        "fib_swing_low":  swing_low,
        "fib_236":        fib_236,
        "fib_382":        fib_382,
        "fib_500":        fib_500,
        "fib_618":        fib_618,
        "fib_786":        fib_786,
    })


# ─────────────────────────────────────────────
# VOLUME INDICATORS
# ─────────────────────────────────────────────

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume — volume terakumulasi sesuai arah harga."""
    direction    = np.sign(close.diff()).fillna(0)
    obv          = (direction * volume).cumsum()
    return obv


def calculate_vwap(high, low, close, volume, period: int = 20) -> pd.Series:
    """Rolling VWAP — harga rata-rata tertimbang volume (20 candle)."""
    tp   = (high + low + close) / 3
    vwap = (tp * volume).rolling(period).sum() / volume.rolling(period).sum()
    return vwap


def calculate_williams_r(high, low, close, period: int = 14) -> pd.Series:
    """Williams %R — momentum reversal oscillator."""
    hh = high.rolling(period).max()
    ll  = low.rolling(period).min()
    wr  = -100 * (hh - close) / (hh - ll).replace(0, np.nan)
    return wr.fillna(-50)


def calculate_cci(high, low, close, period: int = 20) -> pd.Series:
    """Commodity Channel Index — identifikasi extreme harga."""
    tp  = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci.fillna(0)


# ─────────────────────────────────────────────
# SMART MONEY CONCEPTS (SMC)
# ─────────────────────────────────────────────

def detect_fair_value_gap(open_: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Fair Value Gap (FVG) / Imbalance — SMC:
    Bullish FVG: candle[-3].high < candle[-1].low → gap belum terisi → area support
    Bearish FVG: candle[-3].low  > candle[-1].high → gap belum terisi → area resistance
    """
    bull_fvg = high.shift(2) < low      # imbalance bullish
    bear_fvg = low.shift(2)  > high     # imbalance bearish

    # Apakah FVG masih "terbuka" (harga belum fill kembali)
    # Bullish FVG terbuka selama close masih di atas fvg_bull_bottom
    fvg_bull_bottom = high.shift(2)
    fvg_bear_top    = low.shift(2)

    return pd.DataFrame({
        "fvg_bull":        bull_fvg.astype(int),
        "fvg_bear":        bear_fvg.astype(int),
        "fvg_bull_top":    low.where(bull_fvg, np.nan),
        "fvg_bull_bottom": fvg_bull_bottom.where(bull_fvg, np.nan),
        "fvg_bear_top":    fvg_bear_top.where(bear_fvg, np.nan),
        "fvg_bear_bottom": high.where(bear_fvg, np.nan),
    })


def detect_order_block(open_: pd.Series, high: pd.Series,
                        low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Order Block (OB) — SMC:
    Bullish OB: last bearish candle sebelum minimum 2 bullish berturut-turut naik kuat
    Bearish OB: last bullish candle sebelum minimum 2 bearish berturut-turut turun kuat

    Order block = zona entry institusi — harga sering balik ke sini.
    """
    bull_c = (close > open_).astype(int)
    bear_c = (close < open_).astype(int)

    # Impulse ke depan: minimal 2 dari 3 candle berikutnya searah
    fwd_bull = bull_c.shift(-1).fillna(0) + bull_c.shift(-2).fillna(0) + bull_c.shift(-3).fillna(0)
    fwd_bear = bear_c.shift(-1).fillna(0) + bear_c.shift(-2).fillna(0) + bear_c.shift(-3).fillna(0)

    ob_bull = bear_c.astype(bool) & (fwd_bull >= 2)  # bearish candle sebelum impulse naik
    ob_bear = bull_c.astype(bool) & (fwd_bear >= 2)  # bullish candle sebelum impulse turun

    return pd.DataFrame({
        "ob_bull":  ob_bull.fillna(False).astype(int),
        "ob_bear":  ob_bear.fillna(False).astype(int),
        "ob_high":  high.where(ob_bull | ob_bear, np.nan),
        "ob_low":   low.where(ob_bull | ob_bear, np.nan),
        "ob_mid":   ((high + low) / 2).where(ob_bull | ob_bear, np.nan),
    })


def detect_bos_choch(high: pd.Series, low: pd.Series,
                      close: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """
    Break of Structure (BOS) / Change of Character (ChoCH) — SMC:

    BOS Bull  : close melewati swing high N candle lalu → konfirmasi uptrend berlanjut
    BOS Bear  : close melewati swing low N candle lalu → konfirmasi downtrend berlanjut
    ChoCH Bull: BOS Bull yang terjadi dalam konteks downtrend → kemungkinan reversal
    ChoCH Bear: BOS Bear yang terjadi dalam konteks uptrend → kemungkinan reversal
    """
    prev_high = high.shift(1).rolling(lookback).max()
    prev_low  = low.shift(1).rolling(lookback).min()

    bos_bull = (close > prev_high).astype(int)
    bos_bear = (close < prev_low).astype(int)

    # Konteks trend jangka menengah
    ema_mid   = close.ewm(span=50, adjust=False).mean()
    in_down   = close < ema_mid      # dalam downtrend
    in_up     = close > ema_mid      # dalam uptrend

    choch_bull = bos_bull.astype(bool) & in_down   # BOS naik tapi masih dalam downtrend → ChoCH
    choch_bear = bos_bear.astype(bool) & in_up     # BOS turun tapi masih dalam uptrend → ChoCH

    return pd.DataFrame({
        "bos_bull":   bos_bull,
        "bos_bear":   bos_bear,
        "choch_bull": choch_bull.astype(int),
        "choch_bear": choch_bear.astype(int),
        "swing_high": prev_high,
        "swing_low":  prev_low,
    })


def detect_liquidity_sweep(high: pd.Series, low: pd.Series,
                            close: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """
    Liquidity Sweep / Stop Hunt — SMC:
    Harga spike melewati swing high/low tapi langsung balik → institusi ambil likuiditas
    Ini adalah sinyal reversal yang sangat kuat.
    """
    prev_high = high.shift(1).rolling(lookback).max()
    prev_low  = low.shift(1).rolling(lookback).min()

    # Spike atas swing high tapi close di bawahnya → bearish sweep
    bear_sweep = (high > prev_high) & (close < prev_high)
    # Spike bawah swing low tapi close di atasnya → bullish sweep
    bull_sweep = (low < prev_low) & (close > prev_low)

    return pd.DataFrame({
        "liq_bull_sweep": bull_sweep.astype(int),  # reversal naik setelah sweep
        "liq_bear_sweep": bear_sweep.astype(int),  # reversal turun setelah sweep
    })


# ─────────────────────────────────────────────
# MARKET REGIME DETECTION
# ─────────────────────────────────────────────

def detect_market_regime(close: pd.Series, atr: pd.Series,
                          adx: pd.Series, bb_bw: pd.Series) -> pd.Series:
    """
    Deteksi kondisi pasar: TREND / RANGE / VOLATILE
    - TREND   : ADX > 25, EMA aligned, momentum konsisten
    - RANGE   : ADX < 20, BB narrow, harga bolak-balik
    - VOLATILE: ATR spike, BB melebar tiba-tiba
    """
    atr_pct   = (atr / close.replace(0, np.nan)).fillna(0)
    atr_mean  = atr_pct.rolling(50, min_periods=10).mean().fillna(atr_pct)
    bb_q30    = bb_bw.rolling(100, min_periods=20).quantile(0.30).fillna(bb_bw.mean())
    bb_q70    = bb_bw.rolling(100, min_periods=20).quantile(0.70).fillna(bb_bw.mean())

    adx_filled = adx.fillna(20)
    bb_bw_filled = bb_bw.fillna(0.02)

    trending  = adx_filled > 25
    ranging   = (adx_filled < 20) & (bb_bw_filled < bb_q30)
    volatile  = (atr_pct > atr_mean * 1.8) & (bb_bw_filled > bb_q70)

    # Build regime string Series dengan .where() chain (aman dari NaN)
    regime = pd.Series("RANGE", index=close.index, dtype=object)
    regime = regime.where(~trending,  "TREND")
    regime = regime.where(~volatile,  "VOLATILE")
    regime = regime.where(~ranging,   "RANGE")
    return regime.fillna("RANGE")


# ─────────────────────────────────────────────
# CANDLE PATTERN DETECTION
# ─────────────────────────────────────────────

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
    body         = close - open_
    range_       = (high - low).replace(0, np.nan)
    body_ratio   = body.abs() / range_
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
    tweezer_bot   = (body.abs() / range_ > 0.3) & (low.round(2) == low.shift().round(2)) & (body > 0)
    tweezer_top   = (body.abs() / range_ > 0.3) & (high.round(2) == high.shift().round(2)) & (body < 0)

    name = pd.Series("None", index=close.index)
    name = name.where(~spinning_top,   "Spinning Top")
    name = name.where(~doji,           "Doji")
    name = name.where(~hanging_man,    "Hanging Man")
    name = name.where(~inv_hammer,     "Inverted Hammer")
    name = name.where(~shooting_star,  "Shooting Star")
    name = name.where(~hammer,         "Hammer")
    name = name.where(~marubozu_bear,  "Marubozu Bear")
    name = name.where(~marubozu_bull,  "Marubozu Bull")
    name = name.where(~tweezer_top,    "Tweezer Top")
    name = name.where(~tweezer_bot,    "Tweezer Bottom")
    name = name.where(~bearish_engulf, "Bearish Engulfing")
    name = name.where(~bullish_engulf, "Bullish Engulfing")
    return name


def detect_extra_patterns(open_: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Pola multi-candle advanced:
    +2 = Three White Soldiers (sangat bullish)
    +1 = Morning Star / Bullish Harami / Inside Bar Bullish
    -1 = Evening Star / Bearish Harami / Inside Bar Bearish
    -2 = Three Black Crows (sangat bearish)
     0 = No significant pattern
    """
    body = close - open_

    # Three White Soldiers: 3 candle bullish berurutan, setiap close > close sebelumnya
    three_soldiers = (
        (close > open_) &
        (close.shift(1) > open_.shift(1)) &
        (close.shift(2) > open_.shift(2)) &
        (close > close.shift(1)) &
        (close.shift(1) > close.shift(2)) &
        (body > body.shift(1) * 0.7)          # body tidak mengecil drastis
    )

    # Three Black Crows: 3 candle bearish berurutan, setiap close < close sebelumnya
    three_crows = (
        (close < open_) &
        (close.shift(1) < open_.shift(1)) &
        (close.shift(2) < open_.shift(2)) &
        (close < close.shift(1)) &
        (close.shift(1) < close.shift(2)) &
        (body.abs() > body.shift(1).abs() * 0.7)
    )

    # Morning Star: bearish besar | small body (bintang) | bullish yang menutup > 50% bearish
    morning_star = (
        (body.shift(2) < 0) &
        (body.shift(1).abs() < body.shift(2).abs() * 0.4) &
        (body > 0) &
        (close > (open_.shift(2) + close.shift(2)) / 2)
    )

    # Evening Star: bullish besar | small body | bearish yang menutup > 50% bullish
    evening_star = (
        (body.shift(2) > 0) &
        (body.shift(1).abs() < body.shift(2).abs() * 0.4) &
        (body < 0) &
        (close < (open_.shift(2) + close.shift(2)) / 2)
    )

    # Bullish Harami: candle bearish besar lalu candle kecil di dalamnya, bullish
    bull_harami = (
        (body.shift(1) < 0) &
        (body > 0) &
        (open_ > close.shift(1)) &
        (close < open_.shift(1)) &
        (body < body.shift(1).abs() * 0.6)
    )

    # Bearish Harami: candle bullish besar lalu candle kecil di dalamnya, bearish
    bear_harami = (
        (body.shift(1) > 0) &
        (body < 0) &
        (open_ < close.shift(1)) &
        (close > open_.shift(1)) &
        (body.abs() < body.shift(1) * 0.6)
    )

    pat = pd.Series(0, index=close.index)
    pat = pat.where(~bull_harami,    1)
    pat = pat.where(~morning_star,   1)
    pat = pat.where(~bear_harami,   -1)
    pat = pat.where(~evening_star,  -1)
    pat = pat.where(~three_soldiers, 2)
    pat = pat.where(~three_crows,   -2)
    return pat


# ─────────────────────────────────────────────
# RSI DIVERGENCE
# ─────────────────────────────────────────────

def detect_rsi_divergence(close: pd.Series, rsi: pd.Series,
                           lookback: int = 15) -> pd.DataFrame:
    """
    RSI Divergence — sinyal reversal paling kuat:

    Bullish Regular Divergence:
      Price  → lower low  (turun lebih jauh)
      RSI    → higher low (tidak mengkonfirmasi penurunan)
      Arti   → selling pressure melemah → kemungkinan reversal NAIK

    Bearish Regular Divergence:
      Price  → higher high (naik lebih tinggi)
      RSI    → lower high  (tidak mengkonfirmasi kenaikan)
      Arti   → buying pressure melemah → kemungkinan reversal TURUN

    Hidden Bullish Divergence (trend continuation):
      Price  → higher low
      RSI    → lower low
      Arti   → pullback dalam uptrend, lanjut NAIK

    Hidden Bearish Divergence:
      Price  → lower high
      RSI    → higher high
      Arti   → pullback dalam downtrend, lanjut TURUN
    """
    # Rolling extremes untuk lookback window
    roll_close_low  = close.rolling(lookback, min_periods=5).min()
    roll_close_high = close.rolling(lookback, min_periods=5).max()
    roll_rsi_low    = rsi.rolling(lookback, min_periods=5).min()
    roll_rsi_high   = rsi.rolling(lookback, min_periods=5).max()

    # Prev extremes (N candles sebelumnya) — untuk mendeteksi divergence
    prev_close_low  = close.shift(1).rolling(lookback, min_periods=5).min()
    prev_close_high = close.shift(1).rolling(lookback, min_periods=5).max()
    prev_rsi_low    = rsi.shift(1).rolling(lookback, min_periods=5).min()
    prev_rsi_high   = rsi.shift(1).rolling(lookback, min_periods=5).max()

    # Bullish regular: close near low & lower than prev low, RSI higher than prev low
    bull_reg_div = (
        (close <= roll_close_low * 1.002) &        # price di near recent low
        (close < prev_close_low) &                 # price lebih rendah dari low sebelumnya
        (rsi > prev_rsi_low + 3)                   # RSI lebih tinggi dari low sebelumnya
    )

    # Bearish regular: close near high & higher than prev high, RSI lower
    bear_reg_div = (
        (close >= roll_close_high * 0.998) &
        (close > prev_close_high) &
        (rsi < prev_rsi_high - 3)
    )

    # Hidden bullish: pullback dalam uptrend — price higher low, RSI lower low
    bull_hid_div = (
        (close > prev_close_low) &                 # price higher low (pullback)
        (rsi < prev_rsi_low - 3) &                 # RSI lower low
        (rsi < 50)                                 # masih di zona bearish RSI
    )

    # Hidden bearish: pullback dalam downtrend
    bear_hid_div = (
        (close < prev_close_high) &
        (rsi > prev_rsi_high + 3) &
        (rsi > 50)
    )

    return pd.DataFrame({
        "rsi_bull_div":  bull_reg_div.fillna(False).astype(int),   # reversal kuat naik
        "rsi_bear_div":  bear_reg_div.fillna(False).astype(int),   # reversal kuat turun
        "rsi_hid_bull":  bull_hid_div.fillna(False).astype(int),   # lanjut naik
        "rsi_hid_bear":  bear_hid_div.fillna(False).astype(int),   # lanjut turun
    })


def detect_momentum_chain(high: pd.Series, low: pd.Series,
                           close: pd.Series, n: int = 4) -> pd.DataFrame:
    """
    Market Structure Chain:
    Bullish: Higher High (HH) + Higher Low (HL) secara berurutan → uptrend terstruktur
    Bearish: Lower Low (LL) + Lower High (LH) secara berurutan → downtrend terstruktur

    Lebih kuat dari ADX karena melihat STRUKTUR, bukan hanya kekuatan trend.
    """
    hh = (high > high.shift(1)).astype(int)   # 1 = higher high
    hl = (low  > low.shift(1)).astype(int)    # 1 = higher low
    ll = (low  < low.shift(1)).astype(int)    # 1 = lower low
    lh = (high < high.shift(1)).astype(int)   # 1 = lower high

    # Berapa banyak HH+HL dalam N candle terakhir
    bull_chain = (hh + hl).rolling(n, min_periods=2).sum()
    bear_chain = (ll + lh).rolling(n, min_periods=2).sum()

    # Slope close untuk arah momentum
    close_slope = (close - close.shift(n)) / close.shift(n).replace(0, np.nan)

    return pd.DataFrame({
        "hh": hh, "hl": hl, "ll": ll, "lh": lh,
        "bull_chain": bull_chain.fillna(0),
        "bear_chain": bear_chain.fillna(0),
        "close_slope": close_slope.fillna(0),
    })


# ─────────────────────────────────────────────
# VOLUME DIVERGENCE
# ─────────────────────────────────────────────

def detect_volume_divergence(close: pd.Series, volume: pd.Series,
                              lookback: int = 10) -> pd.Series:
    """
    Bullish divergence: harga turun tapi volume turun → selling pressure melemah → +1
    Bearish divergence: harga naik tapi volume turun → buying pressure melemah → -1
    Convergence/Normal: volume mendukung arah harga → 0
    """
    price_up  = close > close.shift(lookback)
    price_dn  = close < close.shift(lookback)
    vol_avg   = volume.rolling(lookback).mean()
    vol_up    = volume > vol_avg
    vol_dn    = volume < vol_avg

    bull_div  = price_dn & vol_dn   # harga turun + volume turun → divergence bullish
    bear_div  = price_up & vol_dn   # harga naik + volume turun → divergence bearish

    result = pd.Series(0, index=close.index)
    result = result.where(~bull_div,  1)
    result = result.where(~bear_div, -1)
    return result


# ─────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """
    Hitung semua indikator.
    max_rows: cap jumlah baris yang diproses — indikator seperti EMA200/SMA200
    butuh ~200 baris warmup, setelah itu 1000 baris sudah sangat cukup.
    Pass max_rows=0 untuk proses semua (dipakai saat training ML).
    """
    if max_rows > 0 and len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()
    else:
        df = df.copy()

    # Buang kolom indikator lama jika ada (re-compute dari scratch)
    ohlcv = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[ohlcv]

    # Hapus duplikat timestamp — bisa terjadi saat merge data multi-sumber
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df.get("Volume", pd.Series(1, index=df.index))

    # ── Traditional ──────────────────────────────────────
    df["rsi"]          = calculate_rsi(close)
    macd_df            = calculate_macd(close)
    df                 = pd.concat([df, macd_df], axis=1)
    bb_df              = calculate_bollinger_bands(close)
    df                 = pd.concat([df, bb_df], axis=1)
    stoch_df           = calculate_stochastic(high, low, close)
    df                 = pd.concat([df, stoch_df], axis=1)
    df["atr"]          = calculate_atr(high, low, close)
    adx_df             = calculate_adx(high, low, close)
    df                 = pd.concat([df, adx_df], axis=1)
    ema_df             = calculate_emas(close)
    df                 = pd.concat([df, ema_df], axis=1)
    sma_df             = calculate_smas(close)
    df                 = pd.concat([df, sma_df], axis=1)
    fib_df             = calculate_fibonacci_levels(high, low)
    df                 = pd.concat([df, fib_df], axis=1)

    # ── Volume ───────────────────────────────────────────
    df["obv"]          = calculate_obv(close, volume)
    df["obv_ema"]      = df["obv"].ewm(span=20, adjust=False).mean()
    df["vwap"]         = calculate_vwap(high, low, close, volume)
    df["williams_r"]   = calculate_williams_r(high, low, close)
    df["cci"]          = calculate_cci(high, low, close)
    df["vol_ratio"]    = (volume / volume.rolling(20).mean()).fillna(1.0)
    df["vol_spike"]    = (df["vol_ratio"] > 2.0).astype(int)
    df["vol_div"]      = detect_volume_divergence(close, volume)

    # ── Candle Patterns ───────────────────────────────────
    df["candle_pat"]   = detect_candle_pattern(open_, high, low, close)
    df["candle_name"]  = detect_candle_name(open_, high, low, close)
    df["candle_ex"]    = detect_extra_patterns(open_, high, low, close)

    # ── Smart Money Concepts ──────────────────────────────
    fvg_df             = detect_fair_value_gap(open_, high, low, close)
    df                 = pd.concat([df, fvg_df], axis=1)
    ob_df              = detect_order_block(open_, high, low, close)
    df                 = pd.concat([df, ob_df], axis=1)
    bos_df             = detect_bos_choch(high, low, close)
    df                 = pd.concat([df, bos_df], axis=1)
    liq_df             = detect_liquidity_sweep(high, low, close)
    df                 = pd.concat([df, liq_df], axis=1)

    # ── Market Regime ─────────────────────────────────────
    df["regime"]       = detect_market_regime(close, df["atr"], df["adx"], df["bb_bw"])

    # ── RSI Divergence ────────────────────────────────────
    rsi_div_df         = detect_rsi_divergence(close, df["rsi"])
    df                 = pd.concat([df, rsi_div_df], axis=1)

    # ── Momentum Chain ────────────────────────────────────
    mom_df             = detect_momentum_chain(high, low, close)
    df                 = pd.concat([df, mom_df], axis=1)

    # ── Momentum & Price Context ──────────────────────────
    df["price_change"]  = close.pct_change()
    df["volatility"]    = close.rolling(20).std() / close.rolling(20).mean()
    df["momentum"]      = close - close.shift(10)
    df["higher_high"]   = (high > high.shift(1)).astype(int)
    df["lower_low"]     = (low < low.shift(1)).astype(int)

    # Slope EMA untuk trend strength
    ema20 = df.get(f"ema_{EMA_SLOW}", close)
    ema50 = df.get(f"ema_{EMA_TREND}", close)
    df["ema20_slope"]  = (ema20 - ema20.shift(5)) / ema20.shift(5)
    df["ema50_slope"]  = (ema50 - ema50.shift(5)) / ema50.shift(5)

    # Price position relative to key levels
    df["price_vs_vwap"]   = (close - df["vwap"]) / df["vwap"]
    df["price_vs_ema200"] = (close - df[f"ema_{EMA_LONG}"]) / df[f"ema_{EMA_LONG}"]

    # ── Fill NaN untuk kolom opsional (jangan drop row karena ini) ────
    # SMC kolom binary → 0 kalau tidak ada sinyal
    smc_int_cols = [
        "fvg_bull", "fvg_bear", "ob_bull", "ob_bear",
        "bos_bull", "bos_bear", "choch_bull", "choch_bear",
        "liq_bull_sweep", "liq_bear_sweep", "vol_spike", "vol_div",
        "candle_ex",
        "rsi_bull_div", "rsi_bear_div", "rsi_hid_bull", "rsi_hid_bear",
        "hh", "hl", "ll", "lh",
    ]
    for col in smc_int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Momentum chain float cols
    for col in ["bull_chain", "bear_chain", "close_slope"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # SMA cols — fill forward to avoid NaN on early rows
    for col in ["sma10", "sma20", "sma50", "sma200"]:
        if col in df.columns:
            df[col] = df[col].fillna(close)

    # Fibonacci cols — fill with close as neutral default
    for col in ["fib_swing_high", "fib_swing_low", "fib_236", "fib_382",
                "fib_500", "fib_618", "fib_786"]:
        if col in df.columns:
            df[col] = df[col].fillna(close)

    # SMC float kolom (level harga opsional) → fill 0
    smc_float_cols = [
        "fvg_bull_top", "fvg_bull_bottom", "fvg_bear_top", "fvg_bear_bottom",
        "ob_high", "ob_low", "ob_mid", "swing_high", "swing_low",
    ]
    for col in smc_float_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Volume kolom → fill 0 / 1
    if "vol_ratio" in df.columns:
        df["vol_ratio"] = df["vol_ratio"].fillna(1.0)
    if "obv" in df.columns:
        df["obv"] = df["obv"].fillna(0)
    if "obv_ema" in df.columns:
        df["obv_ema"] = df["obv_ema"].fillna(0)
    if "vwap" in df.columns:
        df["vwap"] = df["vwap"].fillna(close)
    if "price_vs_vwap" in df.columns:
        df["price_vs_vwap"] = df["price_vs_vwap"].fillna(0)

    # Drop hanya berdasarkan kolom indikator utama (bukan semua kolom)
    core_cols = [
        "Open", "High", "Low", "Close", "rsi", "macd", "histogram",
        "bb_upper", "bb_lower", "stoch_k", "atr", "adx",
        f"ema_{EMA_FAST}", f"ema_{EMA_SLOW}", f"ema_{EMA_TREND}",
    ]
    existing_core = [c for c in core_cols if c in df.columns]
    return df.dropna(subset=existing_core)
