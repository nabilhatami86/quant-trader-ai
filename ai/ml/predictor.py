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


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering yang sama persis dengan notebook."""
    d = df.copy()
    # Normalisasi nama kolom
    d.columns = [c.lower() for c in d.columns]
    c = d['close']; h = d['high']; l = d['low']; o = d['open']

    # Candle structure
    d['body']         = (c - o).abs()
    d['candle_range'] = h - l
    d['upper_wick']   = h - d[['close','open']].max(axis=1)
    d['lower_wick']   = d[['close','open']].min(axis=1) - l
    d['body_pct']     = d['body'] / (d['candle_range'] + 1e-9)
    d['is_bull']      = (c > o).astype(int)

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['atr']       = tr.ewm(span=14, adjust=False).mean()
    d['atr5']      = tr.ewm(span=5,  adjust=False).mean()
    d['range_atr'] = d['candle_range'] / (d['atr'] + 1e-9)

    # EMAs (normalized, bukan raw price)
    for span in [9, 20, 50, 200]:
        d[f'ema{span}'] = c.ewm(span=span, adjust=False).mean()
    d['price_ema9']   = (c - d['ema9'])  / (d['atr'] + 1e-9)
    d['price_ema20']  = (c - d['ema20']) / (d['atr'] + 1e-9)
    d['price_ema50']  = (c - d['ema50']) / (d['atr'] + 1e-9)
    d['ema9_20']      = (d['ema9'] - d['ema20']) / (d['atr'] + 1e-9)
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

    # MACD (normalized)
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    d['macd']           = (ema12 - ema26) / (d['atr'] + 1e-9)
    d['macd_sig']       = d['macd'].ewm(span=9).mean()
    d['macd_hist']      = d['macd'] - d['macd_sig']
    d['macd_hist_lag1'] = d['macd_hist'].shift(1)
    d['macd_cross']     = (np.sign(d['macd_hist']) != np.sign(d['macd_hist_lag1'])).astype(int)

    # ADX
    dm_p = (h - h.shift()).clip(lower=0)
    dm_n = (l.shift() - l).clip(lower=0)
    di_p = 100 * dm_p.ewm(span=14).mean() / (d['atr'] + 1e-9)
    di_n = 100 * dm_n.ewm(span=14).mean() / (d['atr'] + 1e-9)
    dx   = 100 * (di_p - di_n).abs() / (di_p + di_n + 1e-9)
    d['adx']     = dx.ewm(span=14).mean()
    d['di_diff'] = (di_p - di_n) / (di_p + di_n + 1e-9)

    # Bollinger Bands
    bb_m = c.rolling(20).mean(); bb_s = c.rolling(20).std()
    d['bb_pct']   = (c - (bb_m - 2*bb_s)) / (4*bb_s + 1e-9)
    d['bb_width'] = (4*bb_s) / (bb_m + 1e-9)

    # Stochastic
    ll = l.rolling(14).min(); hh = h.rolling(14).max()
    d['stoch_k'] = 100 * (c - ll) / (hh - ll + 1e-9)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    # Returns
    for n in [1, 2, 3, 5, 10]:
        d[f'ret{n}'] = c.pct_change(n)

    # Momentum ATR-normalized
    d['mom3_atr']  = (c - c.shift(3))  / (d['atr'] + 1e-9)
    d['mom5_atr']  = (c - c.shift(5))  / (d['atr'] + 1e-9)
    d['mom10_atr'] = (c - c.shift(10)) / (d['atr'] + 1e-9)

    # Volume
    vol = d.get('volume', pd.Series(1, index=d.index)).replace(0, np.nan)
    d['vol_r5']  = vol / (vol.rolling(5).mean()  + 1e-9)
    d['vol_r20'] = vol / (vol.rolling(20).mean() + 1e-9)

    # Session
    h_utc = d.index.hour if hasattr(d.index, 'hour') else pd.Series(12, index=d.index)
    d['is_london']  = ((h_utc >= 7)  & (h_utc < 16)).astype(int)
    d['is_ny']      = ((h_utc >= 12) & (h_utc < 21)).astype(int)
    d['is_overlap'] = ((h_utc >= 12) & (h_utc < 16)).astype(int)
    d['hour']       = h_utc
    d['dow']        = d.index.dayofweek if hasattr(d.index, 'dayofweek') else 0

    # Candle patterns
    d['hammer']        = ((d['lower_wick'] > 2*d['body']) & (d['upper_wick'] < d['body'])).astype(int)
    d['shooting_star'] = ((d['upper_wick'] > 2*d['body']) & (d['lower_wick'] < d['body'])).astype(int)
    d['bull_engulf']   = ((d['is_bull']==1) & (d['is_bull'].shift(1)==0) &
                          (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    d['bear_engulf']   = ((d['is_bull']==0) & (d['is_bull'].shift(1)==1) &
                          (o > c.shift(1)) & (c < o.shift(1))).astype(int)

    # Swing high/low
    d['near_high'] = (h > h.rolling(10).max().shift(1) * 0.999).astype(int)
    d['near_low']  = (l < l.rolling(10).min().shift(1) * 1.001).astype(int)

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
