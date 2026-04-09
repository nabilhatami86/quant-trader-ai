import numpy as np
import pandas as pd
from pathlib import Path


def load_mt5(path) -> pd.DataFrame:
    """Load MT5 exported CSV (tab-separated, <DATE> <TIME> <OPEN> ... format)."""
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip().replace('<', '').replace('>', '').lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(
        df['date'].str.replace('.', '-', regex=False) + ' ' + df['time']
    )
    df = df.sort_values('datetime').reset_index(drop=True).set_index('datetime')
    df = df.drop(columns=['date', 'time'], errors='ignore')
    df = df.rename(columns={'spread': 'spread_pt', 'tickvol': 'volume', 'vol': 'vol_real'})
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df


def add_m5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute M5 trend/indicator features (columns prefixed with m5_)."""
    d = df.copy()
    c = d['close']; h = d['high']; l = d['low']

    for s in [20, 50, 200]:
        d[f'm5_ema{s}'] = c.ewm(span=s, adjust=False).mean()

    d['m5_trend20']  = (c > d['m5_ema20']).astype(int)
    d['m5_trend50']  = (c > d['m5_ema50']).astype(int)
    d['m5_trend200'] = (c > d['m5_ema200']).astype(int)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['m5_rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    d['m5_macd']   = ema12 - ema26
    d['m5_macd_s'] = d['m5_macd'].ewm(span=9).mean()
    d['m5_macd_h'] = d['m5_macd'] - d['m5_macd_s']

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d['m5_atr'] = tr.ewm(span=14).mean()

    dm_p = (h - h.shift()).clip(lower=0)
    dm_n = (l.shift() - l).clip(lower=0)
    di_p = 100 * dm_p.ewm(span=14).mean() / (d['m5_atr'] + 1e-9)
    di_n = 100 * dm_n.ewm(span=14).mean() / (d['m5_atr'] + 1e-9)
    dx   = 100 * (di_p - di_n).abs() / (di_p + di_n + 1e-9)
    d['m5_adx'] = dx.ewm(span=14).mean()

    bb_m = c.rolling(20).mean(); bb_s = c.rolling(20).std()
    d['m5_bb_pct']   = (c - (bb_m - 2 * bb_s)) / (4 * bb_s + 1e-9)
    d['m5_bb_width'] = 4 * bb_s / (bb_m + 1e-9)

    return d[[col for col in d.columns if col.startswith('m5_')]]


def merge_m1_m5(m1: pd.DataFrame, m5_feat: pd.DataFrame) -> pd.DataFrame:
    """Left-join M1 OHLCV with M5 features using backward asof merge."""
    m1r = m1.reset_index()
    m5r = m5_feat.reset_index()
    dt_col = 'datetime' if 'datetime' in m1r.columns else m1r.columns[0]
    return pd.merge_asof(
        m1r.sort_values(dt_col),
        m5r.sort_values(dt_col),
        on=dt_col, direction='backward'
    ).set_index(dt_col)


def add_m1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all M1 features: candle structure, indicators, session, patterns."""
    d = df.copy()
    c = d['close']; h = d['high']; l = d['low']; o = d['open']; v = d['volume']

    # Candle structure
    d['body']         = c - o
    d['body_abs']     = (c - o).abs()
    d['upper_wick']   = h - d[['open', 'close']].max(axis=1)
    d['lower_wick']   = d[['open', 'close']].min(axis=1) - l
    d['candle_range'] = h - l
    d['body_pct']     = d['body_abs'] / (d['candle_range'] + 1e-9)
    d['is_bullish']   = (c > o).astype(int)

    # Returns
    for n in [1, 2, 3, 5, 10, 15, 20]:
        d[f'ret_{n}'] = c.pct_change(n)

    # EMAs
    for s in [5, 9, 14, 21, 50]:
        d[f'ema{s}'] = c.ewm(span=s, adjust=False).mean()
    d['ema5_9']      = d['ema5']  - d['ema9']
    d['ema9_21']     = d['ema9']  - d['ema21']
    d['price_ema9']  = (c - d['ema9'])  / (d['ema9']  + 1e-9)
    d['price_ema21'] = (c - d['ema21']) / (d['ema21'] + 1e-9)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi']       = 100 - (100 / (1 + gain / (loss + 1e-9)))
    d['rsi_lag1']  = d['rsi'].shift(1)
    d['rsi_delta'] = d['rsi'] - d['rsi_lag1']
    d['rsi_ob']    = (d['rsi'] > 70).astype(int)
    d['rsi_os']    = (d['rsi'] < 30).astype(int)

    # MACD
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    d['macd']           = e12 - e26
    d['macd_sig']       = d['macd'].ewm(span=9).mean()
    d['macd_hist']      = d['macd'] - d['macd_sig']
    d['macd_hist_lag1'] = d['macd_hist'].shift(1)

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d['atr']       = tr.ewm(span=14).mean()
    d['atr5']      = tr.ewm(span=5).mean()
    d['range_atr'] = d['candle_range'] / (d['atr'] + 1e-9)
    d['body_atr']  = d['body_abs']     / (d['atr'] + 1e-9)

    # Bollinger Bands
    bb_m = c.rolling(20).mean(); bb_s = c.rolling(20).std()
    d['bb_pct']   = (c - (bb_m - 2 * bb_s)) / (4 * bb_s + 1e-9)
    d['bb_width'] = 4 * bb_s / (bb_m + 1e-9)

    # Stochastic
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    d['stoch_k'] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    # Momentum
    for n in [3, 5, 10]:
        d[f'mom{n}'] = c - c.shift(n)

    # Volume ratios
    d['vol_r5']  = v / (v.rolling(5).mean()  + 1e-9)
    d['vol_r20'] = v / (v.rolling(20).mean() + 1e-9)

    # Session / time
    d['hour']       = d.index.hour
    d['dow']        = d.index.dayofweek
    d['is_london']  = ((d['hour'] >= 7)  & (d['hour'] < 12)).astype(int)
    d['is_ny']      = ((d['hour'] >= 12) & (d['hour'] < 20)).astype(int)
    d['is_overlap'] = ((d['hour'] >= 12) & (d['hour'] < 16)).astype(int)

    # Candle patterns
    d['hammer']   = ((d['lower_wick'] > 2 * d['body_abs']) & (d['body_pct'] < 0.4)).astype(int)
    d['star']     = ((d['upper_wick'] > 2 * d['body_abs']) & (d['body_pct'] < 0.4)).astype(int)
    d['bull_eng'] = ((c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    d['bear_eng'] = ((c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))).astype(int)

    # Spread
    d['spread_usd'] = d['spread_pt'] * 0.001
    d['spread_atr'] = d['spread_usd'] / (d['atr'] + 1e-9)

    # Trend alignment M1 vs M5
    m1_up = (c > d['ema21']).astype(int)
    if 'm5_trend50' in d.columns:
        d['trend_align'] = (m1_up == d['m5_trend50']).astype(int)

    return d
