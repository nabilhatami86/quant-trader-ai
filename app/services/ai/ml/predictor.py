import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

HERE = Path(__file__).parent

# Project root: app/services/ai/ml/ -> app/services/ai/ -> app/services/ -> app/ -> root
_ROOT = HERE.parent.parent.parent.parent

# Lokasi history CSV (shared dengan backend)
_HIST_DIR = _ROOT / 'data' / 'history'


class EnsembleModel:
    """Soft-vote ensemble — harus ada di sini agar joblib bisa unpickle model dari notebook."""
    def __init__(self, estimators):
        self.estimators = estimators
        self.classes_   = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        for _, e in self.estimators:
            try:
                e.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                e.fit(X, y)   # fallback jika estimator tidak support sample_weight
        return self

    def _safe_proba(self, e, X):
        p = e.predict_proba(X)
        if p.ndim == 1 or p.shape[1] == 1:
            p = p.reshape(-1, 1)
            p = np.hstack([1 - p, p])
        return p[:, :2]

    def predict_proba(self, X):
        ps = [self._safe_proba(e, X) for _, e in self.estimators]
        return np.mean(ps, axis=0)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering identik dengan notebook xauusd_scalping_ml.ipynb."""
    ZONE_PERIOD = 50
    ZONE_PCT    = 0.003

    d = df.copy()
    d.columns = [col.lower() for col in d.columns]
    c = d['close']; h = d['high']; l = d['low']; o = d['open']

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['atr']      = tr.ewm(span=14, adjust=False).mean()
    d['atr5']     = tr.ewm(span=5,  adjust=False).mean()
    d['atr_norm'] = d['atr5'] / (d['atr'] + 1e-9)

    # ── Volatility regime features ────────────────────────────────────
    d['atr_pct']    = d['atr'].pct_change(1)
    d['atr_rank50'] = d['atr'].rolling(50).rank(pct=True)   # 0-1, 1=ATR tertinggi
    d['atr_rank20'] = d['atr'].rolling(20).rank(pct=True)
    d['vol_regime'] = (d['atr_rank50'] > 0.8).astype(int)   # 1 = pasar sangat volatile

    # Candle structure
    d['body']        = (c - o).abs()
    d['rng']         = h - l
    d['upper_wick']  = h - pd.concat([c, o], axis=1).max(axis=1)
    d['lower_wick']  = pd.concat([c, o], axis=1).min(axis=1) - l
    d['body_pct']    = d['body'] / (d['rng'] + 1e-9)
    d['is_bull']     = (c > o).astype(int)
    d['range_atr']   = d['rng'] / (d['atr'] + 1e-9)

    # Candle patterns
    d['bullish_engulf'] = ((c > o) & (c.shift(1) < o.shift(1)) &
                           (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    d['bearish_engulf'] = ((c < o) & (c.shift(1) > o.shift(1)) &
                           (c < o.shift(1)) & (o > c.shift(1))).astype(int)
    d['doji']           = (d['body'] < d['rng'] * 0.1).astype(int)
    d['hammer']         = ((d['lower_wick'] > 2*d['body']) & (d['upper_wick'] < d['body'])).astype(int)
    d['shooting_star']  = ((d['upper_wick'] > 2*d['body']) & (d['lower_wick'] < d['body'])).astype(int)
    d['pin_bar_bull']   = (d['lower_wick'] > d['rng'] * 0.6).astype(int)
    d['pin_bar_bear']   = (d['upper_wick'] > d['rng'] * 0.6).astype(int)
    d['inside_bar']     = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    # EMAs
    for sp in [9, 20, 50, 200]:
        d[f'ema{sp}'] = c.ewm(span=sp, adjust=False).mean()
    d['price_ema9']   = (c - d['ema9'])   / (d['atr'] + 1e-9)
    d['price_ema20']  = (c - d['ema20'])  / (d['atr'] + 1e-9)
    d['price_ema50']  = (c - d['ema50'])  / (d['atr'] + 1e-9)
    d['ema9_20']      = (d['ema9']  - d['ema20']) / (d['atr'] + 1e-9)
    d['ema20_50']     = (d['ema20'] - d['ema50']) / (d['atr'] + 1e-9)
    d['trend_ema20']  = (c > d['ema20']).astype(int)
    d['trend_ema50']  = (c > d['ema50']).astype(int)
    d['trend_ema200'] = (c > d['ema200']).astype(int)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi']       = 100 - 100 / (1 + gain / (loss + 1e-9))
    d['rsi_lag1']  = d['rsi'].shift(1)
    d['rsi_slope'] = d['rsi'] - d['rsi_lag1']
    d['rsi_ob']    = (d['rsi'] > 70).astype(int)
    d['rsi_os']    = (d['rsi'] < 30).astype(int)

    # MACD
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    d['macd']          = (ema12 - ema26) / (d['atr'] + 1e-9)
    d['macd_sig']      = d['macd'].ewm(span=9).mean()
    d['macd_hist']     = d['macd'] - d['macd_sig']
    d['macd_cross_up'] = ((d['macd_hist'] > 0) & (d['macd_hist'].shift(1) <= 0)).astype(int)
    d['macd_cross_dn'] = ((d['macd_hist'] < 0) & (d['macd_hist'].shift(1) >= 0)).astype(int)

    # ADX
    dm_p = (h - h.shift()).clip(lower=0)
    dm_n = (l.shift() - l).clip(lower=0)
    di_p = 100 * dm_p.ewm(span=14).mean() / (d['atr'] + 1e-9)
    di_n = 100 * dm_n.ewm(span=14).mean() / (d['atr'] + 1e-9)
    dx   = 100 * (di_p - di_n).abs() / (di_p + di_n + 1e-9)
    d['adx']     = dx.ewm(span=14).mean()
    d['di_diff'] = (di_p - di_n) / (di_p + di_n + 1e-9)
    d['trending'] = (d['adx'] > 25).astype(int)

    # Bollinger Bands
    bb_m = c.rolling(20).mean(); bb_s = c.rolling(20).std()
    d['bb_width']   = 4 * bb_s / (bb_m + 1e-9)
    d['bb_pos']     = (c - (bb_m - 2*bb_s)) / (4*bb_s + 1e-9)
    d['bb_squeeze'] = (4*bb_s < (4*bb_s).rolling(20).mean()).astype(int)

    # Stochastic
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    d['stoch_k'] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    # Volume
    vol = d.get('volume', pd.Series(1, index=d.index)).replace(0, np.nan)
    d['vol_ma20'] = vol.rolling(20).mean()
    d['vol_ratio'] = vol / (d['vol_ma20'] + 1e-9)
    d['vol_spike'] = (d['vol_ratio'] > 1.5).astype(int)

    # MFI
    tp_m = (h + l + c) / 3
    mf   = tp_m * vol.fillna(0)
    pm   = mf.where(tp_m > tp_m.shift(1), 0).rolling(14).sum()
    nm   = mf.where(tp_m < tp_m.shift(1), 0).rolling(14).sum()
    d['mfi'] = 100 - 100 / (1 + pm / (nm + 1e-9))

    # Returns
    for lag in [1, 2, 3, 5]:
        d[f'ret{lag}'] = c.pct_change(lag)
    d['mom5']  = c / c.shift(5)  - 1
    d['mom10'] = c / c.shift(10) - 1

    # Market structure
    d['hh'] = (h > h.shift(1)).astype(int)
    d['ll'] = (l < l.shift(1)).astype(int)
    d['hl'] = (l > l.shift(1)).astype(int)
    d['lh'] = (h < h.shift(1)).astype(int)
    d['hh2'] = ((h > h.shift(1)) & (h.shift(1) > h.shift(2))).astype(int)
    d['ll2'] = ((l < l.shift(1)) & (l.shift(1) < l.shift(2))).astype(int)
    d['swing_high'] = ((h > h.shift(1)) & (h > h.shift(-1))).astype(int)
    d['swing_low']  = ((l < l.shift(1)) & (l < l.shift(-1))).astype(int)
    rh = h.rolling(10).max(); rl = l.rolling(10).min()
    d['bos_bull']    = ((c > rh.shift(1)) & (c.shift(1) <= rh.shift(1))).astype(int)
    d['bos_bear']    = ((c < rl.shift(1)) & (c.shift(1) >= rl.shift(1))).astype(int)
    d['struct_bull'] = d['hh'] + d['hl'] - d['ll'] - d['lh']

    # Liquidity
    eq_thr = d['atr'] * 0.1
    d['equal_high']     = (abs(h - h.shift(1)) < eq_thr).astype(int)
    d['equal_low']      = (abs(l - l.shift(1)) < eq_thr).astype(int)
    d['liq_sweep_up']   = ((h > h.shift(1)) & (c < h.shift(1))).astype(int)
    d['liq_sweep_dn']   = ((l < l.shift(1)) & (c > l.shift(1))).astype(int)
    d['stop_hunt_bull'] = ((l < l.rolling(5).min().shift(1)) & (c > l.rolling(5).min().shift(1))).astype(int)
    d['stop_hunt_bear'] = ((h > h.rolling(5).max().shift(1)) & (c < h.rolling(5).max().shift(1))).astype(int)

    # Area context
    d['r50'] = h.rolling(ZONE_PERIOD).max()
    d['s50'] = l.rolling(ZONE_PERIOD).min()
    zthr = c * ZONE_PCT
    d['near_resistance'] = (abs(c - d['r50']) < zthr).astype(int)
    d['near_support']    = (abs(c - d['s50']) < zthr).astype(int)
    d['dist_res']  = (d['r50'] - c) / (d['atr'] + 1e-9)
    d['dist_sup']  = (c - d['s50']) / (d['atr'] + 1e-9)
    d['range_pos'] = (c - d['s50']) / (d['r50'] - d['s50'] + 1e-9)

    # Session
    try:
        hour = pd.to_datetime(d.index).hour
    except Exception:
        hour = pd.Series(0, index=d.index)
    d['hour']       = hour
    d['is_london']  = ((hour >= 7)  & (hour < 16)).astype(int)
    d['is_newyork'] = ((hour >= 12) & (hour < 21)).astype(int)
    d['is_overlap'] = ((hour >= 12) & (hour < 16)).astype(int)
    d['in_session'] = ((d['is_london'] == 1) | (d['is_newyork'] == 1)).astype(int)

    # Lag features
    for col in ['rsi', 'adx', 'macd_hist', 'body_pct', 'struct_bull']:
        d[f'{col}_lag1'] = d[col].shift(1)
        d[f'{col}_lag2'] = d[col].shift(2)

    # H1 features — default 0, will be filled from h1_df if provided
    d['h1_bull']  = 0
    d['h1_rsi']   = 50.0
    d['h1_align'] = 0

    return d


def _add_h1_features(m5_df: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """Merge H1 trend features into M5 DataFrame."""
    d = m5_df.copy()
    h = h1_df.copy()
    h.columns = [c.lower() for c in h.columns]

    # H1 indicators
    h1_c = h['close']
    h1_ema20 = h1_c.ewm(span=20, adjust=False).mean()
    delta = h1_c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    h1_rsi = 100 - 100 / (1 + gain / (loss + 1e-9))

    h['h1_bull']  = (h1_c > h1_ema20).astype(int)
    h['h1_rsi']   = h1_rsi
    h['h1_align'] = ((h['h1_bull'] == 1) & (h1_rsi > 50)).astype(int)

    # Forward-fill H1 values into M5 timestamps
    h1_feats = h[['h1_bull', 'h1_rsi', 'h1_align']].copy()

    def _strip_tz(idx):
        idx = pd.to_datetime(idx)
        if idx.tz is not None:
            return idx.tz_convert('UTC').tz_localize(None)
        return idx

    h1_feats.index = _strip_tz(h1_feats.index)
    h1_feats = h1_feats.sort_index()
    d.index  = _strip_tz(d.index)

    h1_reset = h1_feats.reset_index()
    h1_reset.columns = ['ts'] + [c for c in h1_reset.columns[1:]]
    m5_reset = pd.DataFrame({'ts': d.index})

    merged = pd.merge_asof(m5_reset.sort_values('ts'), h1_reset.sort_values('ts'),
                           on='ts', direction='backward')

    for col in ['h1_bull', 'h1_rsi', 'h1_align']:
        if col in merged.columns:
            d[col] = merged[col].values

    return d


def _load_history_csv(path) -> pd.DataFrame:
    """
    Load CSV — support dua format:
      1. MT5 export  : tab-separated, kolom <DATE> <TIME> <OPEN> ...
      2. Standard    : comma-separated, index=datetime, kolom OHLCV
    """
    path = Path(path)
    # Cek format dengan baca 1 baris header
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        header = fh.readline()

    if '\t' in header or '<DATE>' in header.upper():
        # Format MT5
        df = pd.read_csv(path, sep='\t')
        df.columns = [c.strip('<>').lower() for c in df.columns]
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df.set_index('datetime')
        elif 'date' in df.columns:
            df.index = pd.to_datetime(df['date'])
        if 'tickvol' in df.columns and 'volume' not in df.columns:
            df = df.rename(columns={'tickvol': 'volume'})
    else:
        # Format standard (pandas CSV)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.strip().lower() for c in df.columns]

    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['open','high','low','close'])
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


class ScalpingPredictor:
    """
    Inference wrapper untuk XAUUSD M5 scalping model.
    Load sekali di init, panggil predict() tiap siklus.
    """

    # ── ATR spike threshold — skip trading jika ATR naik tidak wajar ──
    ATR_SPIKE_MULT = 2.5   # ATR > 2.5x median 100 candle = extreme volatility
    ATR_SPIKE_WIN  = 100   # window median ATR

    # ── Momentum filter — blok arah yang melawan tren candle terakhir ──
    MOMENTUM_WINDOW     = 8    # lihat 8 candle terakhir
    MOMENTUM_MIN_RATIO  = 0.75 # 75% candle harus turun → blok BUY (dan sebaliknya)

    # ── Consecutive loss block ──────────────────────────────────────────
    MAX_CONSEC_LOSS     = 5    # setelah 5 loss berturut-turut arah sama → blok

    def __init__(self, model_path: str = None):
        path = Path(model_path) if model_path else _ROOT / 'ai' / 'ml' / 'models' / 'xauusd_scalping_model.joblib'
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        bundle = joblib.load(path)
        self.model          = bundle['model']
        self.scaler         = bundle['scaler']
        self.selector       = bundle['selector']
        self.feature_cols   = bundle['feature_cols']
        self.selected_feats = bundle['selected_feats']
        # Threshold minimum 0.62 — live WR 37% butuh filter lebih ketat
        self.prob_threshold = max(bundle.get('prob_threshold', 0.62), 0.62)
        # TP/SL multiplier terhadap ATR — RR harus >= 1.8
        self.tp_mult        = bundle.get('tp_mult', 2.0)
        self.sl_mult        = bundle.get('sl_mult', 1.0)
        print(f"[ScalpingPredictor] loaded — {len(self.selected_feats)} features  "
              f"threshold={self.prob_threshold:.2f}  "
              f"TP/SL mult={self.tp_mult}/{self.sl_mult}")

    # ─────────────────────────────────────────────────────────────────
    def predict(self, m5_df: pd.DataFrame, h1_df: pd.DataFrame = None) -> dict:
        """
        Args:
            m5_df : M5 DataFrame — OHLCV (kolom bisa uppercase atau lowercase)
            h1_df : H1 DataFrame — untuk h1_bull, h1_rsi, h1_align features.
                    Jika None, coba auto-load dari data/history/XAUUSD_1h.csv

        Returns:
            dict: direction, probability, prob_buy, prob_sell, confidence,
                  sl, tp, rr, close, reason
        """
        try:
            df_feat = _add_features(m5_df)

            # ── Auto-load H1 dari history jika tidak diberikan ────────
            if h1_df is None:
                h1_path = _HIST_DIR / 'XAUUSD_1h.csv'
                if h1_path.exists():
                    try:
                        h1_df = _load_history_csv(h1_path)
                    except Exception:
                        pass

            if h1_df is not None and not h1_df.empty:
                try:
                    df_feat = _add_h1_features(df_feat, h1_df)
                except Exception:
                    pass  # H1 merge gagal, gunakan default 0

            row   = df_feat.tail(1).copy()
            close = float(m5_df['close'].iloc[-1] if 'close' in m5_df.columns
                          else m5_df['Close'].iloc[-1])
            atr   = float(df_feat['atr'].iloc[-1])

            # ── Volatility Regime Filter ──────────────────────────────
            # Hindari trading saat ATR abnormal tinggi (news crash, gap besar)
            recent_atrs = df_feat['atr'].dropna()
            if len(recent_atrs) >= 20:
                win = min(self.ATR_SPIKE_WIN, len(recent_atrs) - 1)
                atr_median = float(recent_atrs.iloc[-win-1:-1].median())
                atr_spike  = atr_median * self.ATR_SPIKE_MULT
                if atr > atr_spike:
                    return {
                        'direction'  : 'WAIT',
                        'probability': 0.0,
                        'prob_buy'   : 0.0,
                        'prob_sell'  : 0.0,
                        'confidence' : 'VOLATILE',
                        'sl'         : 0.0,
                        'tp'         : 0.0,
                        'rr'         : 0.0,
                        'close'      : round(close, 3),
                        'reason'     : (f"ATR_SPIKE {atr:.1f} > "
                                        f"{atr_spike:.1f} (median={atr_median:.1f}) "
                                        f"— skip volatile market"),
                    }

            # ── Hard Filter 1: Trend Filter (EMA50) ──────────────────────
            # Jika harga di bawah EMA50 M5 → blok BUY, di atas EMA50 → blok SELL
            try:
                _ema50 = df_feat['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                _trend_up   = close > _ema50
                _trend_down = close < _ema50
                if _trend_down and close < _ema50 * 0.9985:   # minimal 0.15% di bawah
                    return {
                        'direction': 'WAIT', 'probability': 0.0,
                        'prob_buy': 0.0, 'prob_sell': 0.0,
                        'confidence': 'TREND_DOWN',
                        'sl': 0.0, 'tp': 0.0, 'rr': 0.0,
                        'close': round(close, 3),
                        'reason': f"TREND_FILTER: harga {close:.1f} di bawah EMA50 {_ema50:.1f} — blok BUY",
                        'signal_notes': [f"Harga di bawah EMA50 ({_ema50:.1f}) — tren TURUN, tidak buka BUY"],
                        'signal_warnings': [], 'analysis': {},
                    }
                elif _trend_up and close > _ema50 * 1.0015:   # minimal 0.15% di atas
                    return {
                        'direction': 'WAIT', 'probability': 0.0,
                        'prob_buy': 0.0, 'prob_sell': 0.0,
                        'confidence': 'TREND_UP',
                        'sl': 0.0, 'tp': 0.0, 'rr': 0.0,
                        'close': round(close, 3),
                        'reason': f"TREND_FILTER: harga {close:.1f} di atas EMA50 {_ema50:.1f} — blok SELL",
                        'signal_notes': [f"Harga di atas EMA50 ({_ema50:.1f}) — tren NAIK, tidak buka SELL"],
                        'signal_warnings': [], 'analysis': {},
                    }
            except Exception:
                pass

            # ── Hard Filter 2: Consecutive Loss Block (5x) ────────────
            try:
                import json as _json_cf
                _cf_path = _ROOT / 'ai' / 'adaptive_state.json'
                if _cf_path.exists():
                    _cf_st   = _json_cf.loads(_cf_path.read_text(encoding='utf-8'))
                    _cf_rts  = _cf_st.get('recent_trades', [])[-10:]
                    for _cf_dir in ('BUY', 'SELL'):
                        _cf_s = 0
                        for _cf_t in reversed(_cf_rts):
                            if _cf_t.get('direction') == _cf_dir and _cf_t.get('result') == 'LOSS':
                                _cf_s += 1
                            else:
                                break
                        if _cf_s >= self.MAX_CONSEC_LOSS:
                            return {
                                'direction': 'WAIT', 'probability': 0.0,
                                'prob_buy': 0.0, 'prob_sell': 0.0,
                                'confidence': 'CONSEC_BLOCK',
                                'sl': 0.0, 'tp': 0.0, 'rr': 0.0,
                                'close': round(close, 3),
                                'reason': f"BLOCKED[CONSEC_LOSS_{_cf_dir}({_cf_s}x)] — cooldown",
                                'signal_notes': [f"⛔ {_cf_s}x {_cf_dir} LOSS berturut-turut — trading {_cf_dir} diblok"],
                                'signal_warnings': [], 'analysis': {},
                            }
            except Exception:
                pass

            # ── Hard Filter 3: Circuit Breaker Harian ─────────────────
            # Jika rugi >= 5 trade hari ini → stop trading 2 jam
            try:
                import json as _json_cb
                from datetime import datetime as _dt_cb, timedelta as _td_cb
                _cb_path = _ROOT / 'ai' / 'adaptive_state.json'
                if _cb_path.exists():
                    _cb_st  = _json_cb.loads(_cb_path.read_text(encoding='utf-8'))
                    _cb_rts = _cb_st.get('recent_trades', [])
                    _today  = _dt_cb.now().strftime('%Y-%m-%d')
                    _today_losses = [t for t in _cb_rts
                                     if t.get('time', '').startswith(_today)
                                     and t.get('result') == 'LOSS']
                    if len(_today_losses) >= 5:
                        # Cek waktu loss terakhir — jika < 2 jam lalu, blok
                        _last_loss_time = _today_losses[-1].get('time', '')
                        try:
                            _llt = _dt_cb.strptime(_last_loss_time, '%Y-%m-%d %H:%M:%S')
                            _elapsed = (_dt_cb.now() - _llt).total_seconds() / 3600
                            if _elapsed < 2.0:
                                _remaining = round((2.0 - _elapsed) * 60)
                                return {
                                    'direction': 'WAIT', 'probability': 0.0,
                                    'prob_buy': 0.0, 'prob_sell': 0.0,
                                    'confidence': 'CIRCUIT_BREAK',
                                    'sl': 0.0, 'tp': 0.0, 'rr': 0.0,
                                    'close': round(close, 3),
                                    'reason': f"CIRCUIT_BREAKER: {len(_today_losses)} loss hari ini — cooldown {_remaining}mnt lagi",
                                    'signal_notes': [f"⛔ Circuit breaker: {len(_today_losses)} rugi hari ini, istirahat {_remaining} menit"],
                                    'signal_warnings': [], 'analysis': {},
                                }
                        except Exception:
                            pass
            except Exception:
                pass

            # ── Candle & Context Analysis ─────────────────────────────
            _notes    = []   # list of note strings untuk ditampilkan ke trader
            _warnings = []   # catatan risiko

            # 1. Baca momentum candle (3 dan 8 terakhir)
            try:
                _closes = df_feat['close'].dropna().values
                _opens  = df_feat['open'].dropna().values if 'open' in df_feat.columns else None
                if len(_closes) >= 9:
                    _c8 = _closes[-8:]
                    _up8 = sum(1 for i in range(1, 8) if _c8[i] > _c8[i-1])
                    _dn8 = 7 - _up8
                    _c3  = _closes[-3:]
                    _up3 = sum(1 for i in range(1, 3) if _c3[i] > _c3[i-1])
                    _dn3 = 2 - _up3

                    _mom8 = 'NAIK' if _up8 >= 5 else ('TURUN' if _dn8 >= 5 else 'MIXED')
                    _mom3 = 'NAIK' if _up3 >= 2 else ('TURUN' if _dn3 >= 2 else 'MIXED')
                    _notes.append(f"Momentum 8c: {_up8}↑/{_dn8}↓ ({_mom8})  |  3c: {_up3}↑/{_dn3}↓ ({_mom3})")
            except Exception:
                pass

            # 2. Candle terakhir — kekuatan body
            try:
                _last_close = float(df_feat['close'].iloc[-1])
                _last_open  = float(df_feat['open'].iloc[-1])
                _last_high  = float(df_feat['high'].iloc[-1])
                _last_low   = float(df_feat['low'].iloc[-1])
                _body  = abs(_last_close - _last_open)
                _range = _last_high - _last_low
                _bpct  = _body / (_range + 1e-9)
                _bull  = _last_close > _last_open
                _shape = 'Engulf kuat' if _bpct > 0.7 else ('Doji/Spinning' if _bpct < 0.3 else 'Normal')
                _dir_c = 'HIJAU' if _bull else 'MERAH'
                _notes.append(f"Candle terakhir: {_dir_c} {_shape} (body={_bpct:.0%} range={_range:.1f}pt)")
            except Exception:
                pass

            # 3. Posisi harga di range 20 candle (apakah di atas/bawah midrange)
            try:
                _h20 = float(df_feat['high'].rolling(20).max().iloc[-1])
                _l20 = float(df_feat['low'].rolling(20).min().iloc[-1])
                _mid = (_h20 + _l20) / 2
                _pos = (close - _l20) / (_h20 - _l20 + 1e-9)
                _pos_str = 'ATAS range' if _pos > 0.65 else ('BAWAH range' if _pos < 0.35 else 'TENGAH range')
                _notes.append(f"Harga di {_pos_str} 20c ({_pos:.0%} dari low={_l20:.1f} high={_h20:.1f})")
            except Exception:
                pass

            # 4. ATR context
            try:
                _atr_rank = float(df_feat['atr_rank50'].iloc[-1]) if 'atr_rank50' in df_feat.columns else 0.5
                _vol_str  = 'SANGAT TINGGI ⚠' if _atr_rank > 0.85 else \
                            ('TINGGI' if _atr_rank > 0.65 else \
                            ('RENDAH' if _atr_rank < 0.3 else 'NORMAL'))
                _notes.append(f"Volatilitas (ATR rank50): {_vol_str} ({_atr_rank:.0%})")
                if _atr_rank > 0.85:
                    _warnings.append("ATR sangat tinggi — SL mudah kena")
            except Exception:
                pass

            # 5. Consecutive loss streak (warning, tidak blok)
            _consec_streak = 0
            _consec_dir    = ''
            try:
                import json as _json
                _state_path = _ROOT / 'ai' / 'adaptive_state.json'
                if _state_path.exists():
                    _st = _json.loads(_state_path.read_text(encoding='utf-8'))
                    _rts = _st.get('recent_trades', [])[-10:]
                    for _chk_dir in ('BUY', 'SELL'):
                        _s = 0
                        for _t in reversed(_rts):
                            if _t.get('direction') == _chk_dir and _t.get('result') == 'LOSS':
                                _s += 1
                            else:
                                break
                        if _s > _consec_streak:
                            _consec_streak = _s
                            _consec_dir    = _chk_dir
            except Exception:
                pass

            if _consec_streak >= 3:
                _warnings.append(f"⚠ {_consec_streak}x {_consec_dir} LOSS berturut-turut — waspadai")
            elif _consec_streak >= 2:
                _warnings.append(f"⚡ {_consec_streak}x {_consec_dir} LOSS — mulai perhatikan")

            # Pastikan semua kolom fitur ada
            for col in self.feature_cols:
                if col not in row.columns:
                    row[col] = 0.0

            X     = row[self.feature_cols].ffill().fillna(0).replace([np.inf, -np.inf], 0)
            X_s   = self.scaler.transform(X)
            X_sel = self.selector.transform(X_s)

            # Binary: 1=BUY, 0=SELL
            proba = self.model.predict_proba(X_sel)[0]
            # Cari index class 1 (BUY) dan 0 (SELL) — aman untuk model apapun
            classes = list(self.model.classes_) if hasattr(self.model, 'classes_') else [0, 1]
            buy_idx  = classes.index(1) if 1 in classes else 1
            sell_idx = classes.index(0) if 0 in classes else 0
            prob_buy  = float(proba[buy_idx])
            prob_sell = float(proba[sell_idx])

            if prob_buy >= self.prob_threshold:
                direction, prob = 'BUY', prob_buy
            elif prob_sell >= self.prob_threshold:
                direction, prob = 'SELL', prob_sell
            else:
                direction, prob = 'WAIT', max(prob_buy, prob_sell)

            dominant   = max(prob_buy, prob_sell)
            confidence = ('HIGH'   if dominant >= 0.75 else
                          'MEDIUM' if dominant >= 0.68 else 'LOW')

            # 6. Alignment check: ML vs candle momentum
            try:
                if direction in ('BUY', 'SELL') and len(_closes) >= 9:
                    _ml_buy = direction == 'BUY'
                    _candle_bull = _up8 >= 5
                    _candle_bear = _dn8 >= 5
                    if _ml_buy and _candle_bull:
                        _notes.append("Alignment: ML BUY ✓ sejalan candle NAIK")
                    elif not _ml_buy and _candle_bear:
                        _notes.append("Alignment: ML SELL ✓ sejalan candle TURUN")
                    elif _ml_buy and _candle_bear:
                        _warnings.append("KONFLIK: ML BUY tapi candle 8c mayoritas TURUN")
                    elif not _ml_buy and _candle_bull:
                        _warnings.append("KONFLIK: ML SELL tapi candle 8c mayoritas NAIK")
                    else:
                        _notes.append("Alignment: candle mixed — andalkan ML")
            except Exception:
                pass

            # 7. TP Score — hitung dulu sebelum tentukan jarak TP/SL
            _tp_verdict   = ''
            _tp_score     = 0
            try:
                if direction in ('BUY', 'SELL'):
                    # a) Momentum 8c mendukung arah?
                    if len(_closes) >= 9:
                        _mom_ok = (_up8 >= 4 and direction == 'BUY') or \
                                  (_dn8 >= 4 and direction == 'SELL')
                        if _mom_ok:
                            _tp_score += 1

                    # b) Candle terakhir searah trade?
                    try:
                        _last_bull2 = float(df_feat['close'].iloc[-1]) > float(df_feat['open'].iloc[-1])
                        _last_ok    = (_last_bull2 and direction == 'BUY') or \
                                      (not _last_bull2 and direction == 'SELL')
                        if _last_ok:
                            _tp_score += 1
                    except Exception:
                        pass

                    # c) Confidence ML HIGH?
                    if confidence == 'HIGH':
                        _tp_score += 1

                    # d) Tidak ada resistance/support kuat di default TP path (12pt)
                    try:
                        _h20b = float(df_feat['high'].rolling(20).max().iloc[-1])
                        _l20b = float(df_feat['low'].rolling(20).min().iloc[-1])
                        _tp12 = close + 12.0 if direction == 'BUY' else close - 12.0
                        _path_clear = (_tp12 <= _h20b) if direction == 'BUY' else (_tp12 >= _l20b)
                        if _path_clear:
                            _tp_score += 1
                        else:
                            _warnings.append(
                                f"TP12={_tp12:.1f} lewati {'high' if direction=='BUY' else 'low'} 20c "
                                f"({'%.1f' % _h20b if direction=='BUY' else '%.1f' % _l20b}) — resistance/support di depan"
                            )
                    except Exception:
                        pass
            except Exception:
                pass

            # 8. Swing-based SL/TP — tentukan harga spesifik dari struktur candle
            # SL BUY  = di bawah swing low terdekat (8c terakhir)
            # SL SELL = di atas swing high terdekat (8c terakhir)
            # TP BUY  = swing high terdekat di atas harga (20c)
            # TP SELL = swing low terdekat di bawah harga (20c)
            _sl_price = None
            _tp_price = None
            _sl_basis = ''
            _tp_basis = ''

            try:
                _highs = df_feat['high'].dropna()
                _lows  = df_feat['low'].dropna()

                if direction == 'BUY':
                    # SL: di bawah swing low 8 candle terakhir (tidak termasuk candle sekarang)
                    _swing_low = float(_lows.iloc[-9:-1].min())
                    _sl_price  = round(_swing_low - 0.5, 2)   # buffer 0.5pt
                    _sl_basis  = f"swing low {_swing_low:.1f} (8c)"

                    # TP: swing high terdekat di atas harga saat ini dalam 20c
                    _cand_highs = _highs.iloc[-20:][_highs.iloc[-20:] > close]
                    if len(_cand_highs) >= 2:
                        # ambil level yang paling sering disentuh (nearest significant)
                        _swing_high = float(_cand_highs.nsmallest(3).max())
                        _tp_price   = round(_swing_high - 0.2, 2)  # sedikit sebelum high
                        _tp_basis   = f"swing high {_swing_high:.1f} (20c)"
                    else:
                        # Tidak ada swing high jelas → fallback ke tp_score-based
                        _tp_price = None

                elif direction == 'SELL':
                    # SL: di atas swing high 8 candle terakhir
                    _swing_high = float(_highs.iloc[-9:-1].max())
                    _sl_price   = round(_swing_high + 0.5, 2)
                    _sl_basis   = f"swing high {_swing_high:.1f} (8c)"

                    # TP: swing low terdekat di bawah harga saat ini dalam 20c
                    _cand_lows = _lows.iloc[-20:][_lows.iloc[-20:] < close]
                    if len(_cand_lows) >= 2:
                        _swing_low2 = float(_cand_lows.nlargest(3).min())
                        _tp_price   = round(_swing_low2 + 0.2, 2)
                        _tp_basis   = f"swing low {_swing_low2:.1f} (20c)"
                    else:
                        _tp_price = None
            except Exception:
                pass

            # Validasi swing SL/TP — pastikan masuk akal
            _use_swing = False
            if _sl_price is not None and _tp_price is not None and direction in ('BUY', 'SELL'):
                _sl_dist_sw = abs(close - _sl_price)
                _tp_dist_sw = abs(close - _tp_price)
                _rr_sw      = _tp_dist_sw / (_sl_dist_sw + 1e-9)

                # Batasan: SL tidak boleh > 15pt, TP tidak boleh > 20pt, RR >= 1.2
                if (0.5 <= _sl_dist_sw <= 15.0 and
                    1.0 <= _tp_dist_sw <= 20.0 and
                    _rr_sw >= 1.2):
                    _use_swing = True
                    _notes.append(
                        f"SL={_sl_price:.1f} ({_sl_basis}, jarak {_sl_dist_sw:.1f}pt)  "
                        f"TP={_tp_price:.1f} ({_tp_basis}, jarak {_tp_dist_sw:.1f}pt)  "
                        f"RR=1:{_rr_sw:.1f}  Score={_tp_score}/4"
                    )

            # Fallback ke score-based jika swing tidak valid
            if not _use_swing:
                if _tp_score >= 3:
                    _tp_pts, _sl_pts = 12.0, 6.0
                    _tp_verdict = f"KUAT ({_tp_score}/4) → TP=12pt SL=6pt"
                elif _tp_score == 2:
                    _tp_pts, _sl_pts = 8.0, 5.0
                    _tp_verdict = f"MEDIUM ({_tp_score}/4) → TP=8pt SL=5pt"
                else:
                    _tp_pts, _sl_pts = 6.0, 5.0
                    _tp_verdict = f"LEMAH ({_tp_score}/4) → TP=6pt SL=5pt"
                _tp_price = round(close + _tp_pts if direction == 'BUY' else close - _tp_pts, 2)
                _sl_price = round(close - _sl_pts if direction == 'BUY' else close + _sl_pts, 2)
                if direction in ('BUY', 'SELL'):
                    _notes.append(f"SL={_sl_price:.1f}  TP={_tp_price:.1f}: {_tp_verdict} (fallback fixed)")

            # Hitung tp_dist / sl_dist dari harga final
            if direction in ('BUY', 'SELL') and _sl_price and _tp_price:
                tp_dist = abs(close - _tp_price)
                sl_dist = abs(close - _sl_price)
            else:
                tp_dist = atr * self.tp_mult
                sl_dist = atr * self.sl_mult
            rr = round(tp_dist / (sl_dist + 1e-9), 1)

            # ── Tolak jika RR < 1.2 ──────────────────────────────────
            if direction in ('BUY', 'SELL') and rr < 1.2:
                direction = 'WAIT'
                _warnings.append(f"RR terlalu kecil ({rr}) — skip")

            # Gabungkan notes & warnings
            _all_notes = _notes + (_warnings if _warnings else [])

            _reason = (f"ScalpML buy={prob_buy:.3f} sell={prob_sell:.3f} "
                       f"thr={self.prob_threshold} atr={atr:.2f}")

            # Kumpulkan data terstruktur untuk logging
            _analysis = {
                'momentum_8c_up'   : int(_up8)   if 'up8' in dir() else None,
                'momentum_8c_dn'   : int(_dn8)   if 'dn8' in dir() else None,
                'momentum_8c'      : _mom8        if '_mom8' in dir() else None,
                'momentum_3c'      : _mom3        if '_mom3' in dir() else None,
                'last_candle_dir'  : _dir_c       if '_dir_c' in dir() else None,
                'last_body_pct'    : round(_bpct, 3) if '_bpct' in dir() else None,
                'last_candle_shape': _shape       if '_shape' in dir() else None,
                'price_pos_pct'    : round(_pos, 3)  if '_pos' in dir() else None,
                'price_pos_label'  : _pos_str     if '_pos_str' in dir() else None,
                'atr_rank50'       : round(_atr_rank, 3) if '_atr_rank' in dir() else None,
                'vol_regime'       : _vol_str     if '_vol_str' in dir() else None,
                'consec_streak'    : _consec_streak,
                'consec_dir'       : _consec_dir,
                'tp_score'         : _tp_score    if '_tp_score' in dir() else None,
                'tp_verdict'       : _tp_verdict  if '_tp_verdict' in dir() else None,
                'alignment'        : next((n for n in _notes if 'Alignment' in n or 'KONFLIK' in n), ''),
            }

            # Harga SL/TP final (swing-based atau fallback)
            _final_sl = _sl_price if _sl_price is not None else round(
                close - sl_dist if direction == 'BUY' else close + sl_dist, 3)
            _final_tp = _tp_price if _tp_price is not None else round(
                close + tp_dist if direction == 'BUY' else close - tp_dist, 3)

            return {
                'direction'      : direction,
                'probability'    : round(prob, 4),
                'prob_buy'       : round(prob_buy, 4),
                'prob_sell'      : round(prob_sell, 4),
                'confidence'     : confidence,
                'sl'             : _final_sl,
                'tp'             : _final_tp,
                'rr'             : rr,
                'close'          : round(close, 3),
                'atr'            : round(atr, 3),
                'reason'         : _reason,
                'signal_notes'   : _notes,
                'signal_warnings': _warnings,
                'analysis'       : _analysis,
                'sl_basis'       : _sl_basis,
                'tp_basis'       : _tp_basis,
                'swing_based'    : _use_swing,
            }

        except Exception as e:
            return {'direction': 'WAIT', 'probability': 0.5,
                    'confidence': 'ERROR', 'error': str(e)}

    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def train_from_history(
        cls,
        m5_csv  : str  = None,
        h1_csv  : str  = None,
        out_path: str  = None,
        tp_pips : float = 12.0,
        sl_pips : float = 6.0,
        lookahead: int  = 12,
        prob_threshold: float = 0.62,
        n_trials: int  = 15,
        k_features: int = 30,
        verbose : bool  = True,
    ) -> dict:
        """
        Retrain model dari history CSV.

        Args:
            m5_csv        : path ke XAUUSD_5m.csv  (default: data/history/XAUUSD_5m.csv)
            h1_csv        : path ke XAUUSD_1h.csv  (default: data/history/XAUUSD_1h.csv)
            out_path      : path simpan model .joblib
            tp_pips       : TP dalam price-points XAUUSD (default 12)
            sl_pips       : SL dalam price-points XAUUSD (default 6)
            lookahead     : jumlah candle M5 ke depan untuk cek TP/SL
            prob_threshold: threshold minimum predict() (simpan ke bundle)
            n_trials      : Optuna trials untuk LightGBM tuning
            k_features    : jumlah fitur setelah SelectKBest
            verbose       : print progress

        Returns:
            dict hasil training (auc, winrate, n_train, n_test, ...)
        """
        import warnings
        warnings.filterwarnings('ignore')

        try:
            import lightgbm as lgb
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(f"Butuh lightgbm + xgboost: pip install lightgbm xgboost — {e}")

        from sklearn.preprocessing import RobustScaler
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ── 1. Load data ──────────────────────────────────────────────
        # Prioritas: ai/ml/data/ (MT5 export, lebih banyak) → data/history/ (fallback)
        _mt5_dir  = _ROOT / 'data'
        _m5_mt5   = _mt5_dir / 'XAUUSDm_M5.csv'
        _h1_mt5   = _mt5_dir / 'XAUUSDm_H1.csv'

        if m5_csv:
            m5_path = Path(m5_csv)
        elif _m5_mt5.exists():
            m5_path = _m5_mt5      # pakai MT5 data (72K candle)
        else:
            m5_path = _HIST_DIR / 'XAUUSD_5m.csv'

        if h1_csv:
            h1_path = Path(h1_csv)
        elif _h1_mt5.exists():
            h1_path = _h1_mt5
        else:
            h1_path = _HIST_DIR / 'XAUUSD_1h.csv'

        model_path = Path(out_path) if out_path else (_ROOT / 'ai' / 'ml' / 'models' / 'xauusd_scalping_model.joblib')

        if verbose:
            print(f"[Train] Load M5: {m5_path}")
        m5 = _load_history_csv(m5_path)
        if verbose:
            print(f"[Train] M5: {len(m5):,} candles  {m5.index[0]} → {m5.index[-1]}")

        h1 = None
        if h1_path.exists():
            if verbose:
                print(f"[Train] Load H1: {h1_path}")
            h1 = _load_history_csv(h1_path)
            if verbose:
                print(f"[Train] H1: {len(h1):,} candles  {h1.index[0]} → {h1.index[-1]}")

        # ── 2. Feature engineering ────────────────────────────────────
        if verbose:
            print("[Train] Computing features...")
        df_feat = _add_features(m5)
        if h1 is not None and not h1.empty:
            try:
                df_feat = _add_h1_features(df_feat, h1)
            except Exception as _e:
                if verbose:
                    print(f"[Train] H1 merge skip: {_e}")

        # ── 3. Label creation (binary BUY=1 / SELL=0) ────────────────
        # Label 1 (BUY)  : harga naik TP sebelum kena SL dalam N candle
        # Label 0 (SELL) : harga turun TP sebelum kena SL dalam N candle
        # Baris ambiguous (keduanya kena) → dibuang
        if verbose:
            print(f"[Train] Creating labels  TP={tp_pips}pt SL={sl_pips}pt look={lookahead}c...")

        c_arr = df_feat['close'].values
        h_arr = df_feat['high'].values
        l_arr = df_feat['low'].values
        n     = len(df_feat)
        labels = np.full(n, np.nan)

        for i in range(n - lookahead):
            entry  = c_arr[i]
            buy_tp = entry + tp_pips
            buy_sl = entry - sl_pips
            sel_tp = entry - tp_pips
            sel_sl = entry + sl_pips

            buy_win = buy_loss = sel_win = sel_loss = False
            for j in range(i + 1, i + 1 + lookahead):
                hh = h_arr[j]; ll = l_arr[j]
                if not buy_win  and not buy_loss:
                    if hh >= buy_tp:  buy_win  = True
                    elif ll <= buy_sl: buy_loss = True
                if not sel_win  and not sel_loss:
                    if ll <= sel_tp:  sel_win  = True
                    elif hh >= sel_sl: sel_loss = True

            # Hanya label jika satu arah jelas
            if buy_win and not buy_loss and not (sel_win and not sel_loss):
                labels[i] = 1   # BUY
            elif sel_win and not sel_loss and not (buy_win and not buy_loss):
                labels[i] = 0   # SELL

        df_feat['label'] = labels
        df_valid = df_feat[df_feat['label'].notna()].copy()

        n_buy  = (df_valid['label'] == 1).sum()
        n_sell = (df_valid['label'] == 0).sum()
        if verbose:
            print(f"[Train] Valid rows: {len(df_valid):,}  "
                  f"BUY={n_buy:,} ({n_buy/len(df_valid)*100:.1f}%)  "
                  f"SELL={n_sell:,} ({n_sell/len(df_valid)*100:.1f}%)")

        if len(df_valid) < 500:
            raise ValueError(f"Data terlalu sedikit untuk training: {len(df_valid)} rows")

        # ── 4. Features & split ───────────────────────────────────────
        EXCLUDE = {'label', 'open', 'high', 'low', 'close', 'volume',
                   'r50', 's50', 'ema9', 'ema20', 'ema50', 'ema200', 'rng', 'body'}
        feat_cols = [
            c for c in df_valid.columns
            if c not in EXCLUDE
            and df_valid[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]
        ]

        X_all = df_valid[feat_cols].ffill().fillna(0).replace([np.inf, -np.inf], 0)
        y_all = df_valid['label'].astype(int)

        # Buang kolom dengan >20% null
        null_pct = df_valid[feat_cols].isnull().mean()
        bad_cols = null_pct[null_pct > 0.2].index.tolist()
        X_all = X_all.drop(columns=bad_cols)
        feat_cols = X_all.columns.tolist()

        N     = len(X_all)
        SPLIT = int(N * 0.80)
        X_train, X_test = X_all.iloc[:SPLIT], X_all.iloc[SPLIT:]
        y_train, y_test = y_all.iloc[:SPLIT], y_all.iloc[SPLIT:]

        if verbose:
            print(f"[Train] Train={len(X_train):,}  Test={len(X_test):,}  "
                  f"Features={len(feat_cols)}")

        # ── 5. Scale + Select ─────────────────────────────────────────
        scaler      = RobustScaler()
        X_train_s   = scaler.fit_transform(X_train)
        X_test_s    = scaler.transform(X_test)

        k = min(k_features, X_train_s.shape[1])
        selector    = SelectKBest(mutual_info_classif, k=k)
        X_train_sel = selector.fit_transform(X_train_s, y_train)
        X_test_sel  = selector.transform(X_test_s)
        sel_feats   = np.array(feat_cols)[selector.get_support()].tolist()

        if verbose:
            scores = selector.scores_[selector.get_support()]
            top5   = sorted(zip(sel_feats, scores), key=lambda x: -x[1])[:5]
            print(f"[Train] Top-5 features: {[f for f,_ in top5]}")

        # ── 6. Optuna tuning LightGBM ─────────────────────────────────
        SEED = 42
        tscv = TimeSeriesSplit(n_splits=5)

        def lgb_obj(trial):
            p = dict(
                n_estimators      = trial.suggest_int('n_est', 300, 1000),
                learning_rate     = trial.suggest_float('lr', 0.01, 0.15, log=True),
                max_depth         = trial.suggest_int('depth', 3, 8),
                num_leaves        = trial.suggest_int('leaves', 15, 63),
                subsample         = trial.suggest_float('sub', 0.6, 1.0),
                colsample_bytree  = trial.suggest_float('col', 0.6, 1.0),
                min_child_samples = trial.suggest_int('min_cs', 10, 80),
                reg_alpha         = trial.suggest_float('reg_a', 1e-8, 2.0, log=True),
                reg_lambda        = trial.suggest_float('reg_l', 1e-8, 2.0, log=True),
                random_state=SEED, verbose=-1, n_jobs=-1
            )
            return cross_val_score(
                lgb.LGBMClassifier(**p), X_train_sel, y_train,
                cv=tscv, scoring='roc_auc', n_jobs=1
            ).mean()

        if verbose:
            print(f"[Train] Optuna LGB ({n_trials} trials)...")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        study.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=False)

        _remap = {'n_est': 'n_estimators', 'lr': 'learning_rate', 'depth': 'max_depth',
                  'leaves': 'num_leaves', 'sub': 'subsample', 'col': 'colsample_bytree',
                  'min_cs': 'min_child_samples', 'reg_a': 'reg_alpha', 'reg_l': 'reg_lambda'}
        lgb_params = {_remap.get(k, k): v for k, v in study.best_params.items()}

        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=SEED, verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train_sel, y_train)

        xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            eval_metric='logloss', use_label_encoder=False,
            random_state=SEED, verbosity=0, n_jobs=-1
        )
        xgb_model.fit(X_train_sel, y_train)

        # ── 7. Soft-vote ensemble (VotingClassifier — pickle-safe) ──
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(
            estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
            voting='soft', weights=[1, 1]
        )
        ensemble.fit(X_train_sel, y_train)

        proba_test = ensemble.predict_proba(X_test_sel)
        pred_test  = ensemble.predict(X_test_sel)
        auc        = roc_auc_score(y_test, proba_test[:, 1])
        acc        = accuracy_score(y_test, pred_test)

        # ── Walk-forward AUC ─────────────────────────────────────────
        wf_aucs = []
        for _, (tr_i, te_i) in enumerate(tscv.split(X_train_sel)):
            _m = lgb.LGBMClassifier(**lgb_params, random_state=SEED, verbose=-1)
            _m.fit(X_train_sel[tr_i], y_train.iloc[tr_i])
            _p = _m.predict_proba(X_train_sel[te_i])[:, 1]
            try:
                wf_aucs.append(roc_auc_score(y_train.iloc[te_i], _p))
            except Exception:
                pass
        wf_auc = float(np.mean(wf_aucs)) if wf_aucs else 0.0

        if verbose:
            print(f"[Train] AUC={auc:.4f}  WF-AUC={wf_auc:.4f}  Acc={acc*100:.1f}%")
            print(f"[Train] Threshold: {prob_threshold}")

        # ── 8. Simpan bundle ─────────────────────────────────────────
        import json
        from datetime import datetime

        bundle = {
            'model'         : ensemble,
            'scaler'        : scaler,
            'selector'      : selector,
            'feature_cols'  : feat_cols,
            'selected_feats': sel_feats,
            'prob_threshold': prob_threshold,
            'tp_mult'       : 2.0,
            'sl_mult'       : 1.0,
        }
        joblib.dump(bundle, model_path)

        meta = {
            'feature_cols'  : feat_cols,
            'selected_feats': sel_feats,
            'prob_threshold': prob_threshold,
            'tp_mult'       : 2.0,
            'sl_mult'       : 1.0,
            'symbol'        : 'XAUUSD',
            'tp_pips'       : tp_pips,
            'sl_pips'       : sl_pips,
            'lookahead'     : lookahead,
            'created_at'    : datetime.now().isoformat(),
            'auc_test'      : auc,
            'wf_auc_mean'   : wf_auc,
            'n_train'       : len(X_train),
            'n_test'        : len(X_test),
            'n_total'       : len(df_valid),
            'acc_test'      : round(acc * 100, 2),
        }
        meta_path = model_path.with_suffix('.json').parent / (model_path.stem + '_meta.json')
        meta_path.write_text(json.dumps(meta, indent=2))

        if verbose:
            print(f"[Train] Model saved → {model_path}")
            print(f"[Train] Meta  saved → {meta_path}")

        return meta


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Retrain model dari history CSV')
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--k', type=int, default=30)
    args = parser.parse_args()

    if args.retrain:
        print("=" * 60)
        print("  RETRAIN ScalpingPredictor dari history CSV")
        print("=" * 60)
        result = ScalpingPredictor.train_from_history(
            n_trials=args.trials,
            k_features=args.k,
            verbose=True,
        )
        print("\n[Hasil Training]")
        for k, v in result.items():
            if not isinstance(v, list):
                print(f"  {k:<20}: {v}")
    else:
        pred   = ScalpingPredictor()
        m5_csv = _HIST_DIR / 'XAUUSD_5m.csv'
        if m5_csv.exists():
            m5 = _load_history_csv(m5_csv)
            result = pred.predict(m5.tail(500))
        else:
            print(f"[!] File tidak ditemukan: {m5_csv}")
            result = {}

        print("\n" + "=" * 50)
        print("  SCALPING ML SIGNAL (M5)")
        print("=" * 50)
        for k, v in result.items():
            print(f"  {k:<15}: {v}")
        print("=" * 50)
