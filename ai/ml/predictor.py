"""
ai/ml/predictor.py — Load trained M5 scalping model and generate BUY/SELL/WAIT signals.

Model dilatih dari notebook xauusd_scalping_m5.ipynb (M5 only).
Tidak lagi butuh M1 data.

Usage:
    from ai.ml import ScalpingPredictor
    pred = ScalpingPredictor()
    result = pred.predict(m5_df)
    # {'direction': 'BUY', 'probability': 0.72, 'confidence': 'HIGH', ...}
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

HERE = Path(__file__).parent


class EnsembleModel:
    """Soft-vote ensemble — harus ada di sini agar joblib bisa unpickle model dari notebook."""
    def __init__(self, estimators):
        self.estimators = estimators
        self.classes_   = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for _, e in self.estimators:
            e.fit(X, y)
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

    return d


class ScalpingPredictor:
    """
    Inference wrapper untuk XAUUSD M5 scalping model.
    Load sekali di init, panggil predict() tiap siklus.
    """

    def __init__(self, model_path: str = None):
        path = Path(model_path) if model_path else HERE / 'models' / 'xauusd_scalping_model.joblib'
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        bundle = joblib.load(path)
        self.model          = bundle['model']
        self.scaler         = bundle['scaler']
        self.selector       = bundle['selector']
        self.feature_cols   = bundle['feature_cols']
        self.selected_feats = bundle['selected_feats']
        self.prob_threshold = bundle.get('prob_threshold', 0.58)
        self.tp_mult        = bundle.get('tp_mult', 0.35)
        self.sl_mult        = bundle.get('sl_mult', 0.55)
        print(f"[ScalpingPredictor] loaded — {len(self.selected_feats)} features  "
              f"threshold={self.prob_threshold:.2f}")

    def predict(self, m5_df: pd.DataFrame, _m1_df=None) -> dict:
        """
        Args:
            m5_df : M5 DataFrame — OHLCV (kolom bisa uppercase atau lowercase)
            _m1_df: diabaikan (backward compat)

        Returns:
            dict: direction, probability, prob_buy, prob_sell, confidence, sl, tp, rr, close, reason
        """
        try:
            df_feat = _add_features(m5_df)
            row     = df_feat.tail(1).copy()

            # Pastikan semua kolom fitur ada
            for col in self.feature_cols:
                if col not in row.columns:
                    row[col] = 0.0

            X     = row[self.feature_cols].ffill().fillna(0).replace([np.inf, -np.inf], 0)
            X_s   = self.scaler.transform(X)
            X_sel = self.selector.transform(X_s)

            # Binary: 1=BUY, 0=SELL
            proba    = self.model.predict_proba(X_sel)[0]
            prob_buy  = float(proba[1])
            prob_sell = float(proba[0])

            close = float(m5_df['close'].iloc[-1] if 'close' in m5_df.columns
                          else m5_df['Close'].iloc[-1])
            atr   = float(df_feat['atr'].iloc[-1])

            if prob_buy >= self.prob_threshold:
                direction, prob = 'BUY', prob_buy
            elif prob_sell >= self.prob_threshold:
                direction, prob = 'SELL', prob_sell
            else:
                direction, prob = 'WAIT', max(prob_buy, prob_sell)

            dominant   = max(prob_buy, prob_sell)
            confidence = 'HIGH' if dominant >= 0.72 else 'MEDIUM' if dominant >= 0.65 else 'LOW'

            tp_dist = atr * self.tp_mult
            sl_dist = atr * self.sl_mult

            return {
                'direction'  : direction,
                'probability': round(prob, 4),
                'prob_buy'   : round(prob_buy, 4),
                'prob_sell'  : round(prob_sell, 4),
                'confidence' : confidence,
                'sl'         : round(close - sl_dist if direction == 'BUY' else close + sl_dist, 3),
                'tp'         : round(close + tp_dist if direction == 'BUY' else close - tp_dist, 3),
                'rr'         : round(tp_dist / (sl_dist + 1e-9), 1),
                'close'      : round(close, 3),
                'reason'     : f"ScalpML buy={prob_buy:.3f} sell={prob_sell:.3f} thr={self.prob_threshold}",
            }

        except Exception as e:
            return {'direction': 'WAIT', 'probability': 0.5, 'confidence': 'ERROR', 'error': str(e)}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    from ai.ml.features import load_mt5
    pred = ScalpingPredictor()
    m5   = load_mt5(HERE / 'data' / 'XAUUSDm_M5.csv')
    result = pred.predict(m5.tail(500))

    print("\n" + "=" * 50)
    print("  SCALPING ML SIGNAL (M5)")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k:<15}: {v}")
    print("=" * 50)
