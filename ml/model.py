"""
High-Accuracy ML Model - Candle Direction Predictor
Teknik:
  - Threshold-based labeling (filter noise sideways)
  - 50+ engineered features
  - Voting Ensemble (RandomForest + GradientBoosting + ExtraTrees)
  - Feature importance pruning
  - Confidence filter (hanya predict jika proba > threshold)
Target akurasi: 65-75%
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

from config import *

# Confidence minimum untuk meng-output prediksi (bukan "UNCERTAIN")
# 65% = lebih konservatif, jarang trade tapi akurasi lebih tinggi
CONFIDENCE_THRESHOLD = 65.0

# Label threshold per timeframe — floor minimum (akan di-override ATR jika lebih besar)
LABEL_THRESHOLDS = {
    "1m":  0.0003,
    "5m":  0.0008,   # dinaikkan: GOLD 5m butuh min 0.08% (~$4 di $4900), bukan $1.5
    "15m": 0.0012,
    "1h":  0.0020,
    "4h":  0.0035,
    "1d":  0.006,
}
DEFAULT_LABEL_THRESHOLD = 0.0020

# Berapa candle ke depan untuk prediksi (3 = lebih stabil dari noise vs 1)
LOOKAHEAD_CANDLES = {
    "1m":  3,
    "5m":  3,
    "15m": 3,
    "1h":  2,
    "4h":  2,
    "1d":  1,
}
DEFAULT_LOOKAHEAD = 3


class CandlePredictor:
    def __init__(self, timeframe: str = "1h"):
        self.timeframe     = timeframe
        self.label_thresh  = LABEL_THRESHOLDS.get(timeframe, DEFAULT_LABEL_THRESHOLD)
        self.lookahead     = LOOKAHEAD_CANDLES.get(timeframe, DEFAULT_LOOKAHEAD)
        self.scaler        = RobustScaler()
        self.selector      = None
        self.trained       = False
        self.accuracy      = 0.0
        self.report        = ""
        self.feature_names = []
        self.n_features_selected = 40
        self._build_model()

    # ─── MODEL ENSEMBLE ───────────────────────────────────────
    def _build_model(self):
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
        )
        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_split=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        if ML_MODEL_TYPE == "gb":
            self.model = gb
        elif ML_MODEL_TYPE == "et":
            self.model = et
        elif ML_MODEL_TYPE == "ensemble":
            # Voting ensemble - paling akurat tapi lebih lambat
            self.model = VotingClassifier(
                estimators=[("rf", rf), ("gb", gb), ("et", et)],
                voting="soft",
                n_jobs=-1,
            )
        else:
            # Default: RandomForest (best balance kecepatan/akurasi)
            self.model = rf

    # ─── FEATURE ENGINEERING ──────────────────────────────────
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        open_  = df["Open"]

        X = pd.DataFrame(index=df.index)

        # 1. Indikator dasar (normalized)
        X["rsi"]          = df.get("rsi", 50)
        X["rsi_norm"]     = (df.get("rsi", 50) - 50) / 50  # -1..+1
        X["macd"]         = df.get("macd", 0)
        X["macd_hist"]    = df.get("histogram", 0)
        X["macd_signal"]  = df.get("signal", 0)
        X["adx"]          = df.get("adx", 20)
        X["di_diff"]      = df.get("di_pos", 25) - df.get("di_neg", 25)
        X["stoch_k"]      = df.get("stoch_k", 50)
        X["stoch_d"]      = df.get("stoch_d", 50)
        X["stoch_diff"]   = df.get("stoch_k", 50) - df.get("stoch_d", 50)
        X["atr"]          = df.get("atr", 0.001)
        X["bb_pct"]       = df.get("bb_pct", 0.5)
        X["bb_bw"]        = df.get("bb_bw", 0)
        X["volatility"]   = df.get("volatility", 0)
        X["candle_pat"]   = df.get("candle_pat", 0)

        # 2. Posisi harga relatif terhadap EMA (%)
        for span in [EMA_FAST, EMA_SLOW, EMA_TREND, EMA_LONG]:
            col = f"ema_{span}"
            if col in df.columns:
                X[f"dist_ema{span}"] = (close - df[col]) / df[col]

        # 3. RSI zone (overbought/oversold encoding)
        rsi = df.get("rsi", pd.Series(50, index=df.index))
        X["rsi_oversold"]  = (rsi < RSI_OVERSOLD).astype(int)
        X["rsi_overbought"]= (rsi > RSI_OVERBOUGHT).astype(int)
        X["rsi_bull_zone"] = ((rsi > 40) & (rsi < 60)).astype(int)

        # 4. MACD zero-cross
        macd = df.get("macd", pd.Series(0, index=df.index))
        hist = df.get("histogram", pd.Series(0, index=df.index))
        X["macd_above_zero"] = (macd > 0).astype(int)
        X["hist_rising"]     = (hist > hist.shift(1)).astype(int)
        X["hist_pos"]        = (hist > 0).astype(int)

        # 5. EMA alignment (trend direction strength)
        if all(f"ema_{s}" in df.columns for s in [EMA_FAST, EMA_SLOW, EMA_TREND]):
            ef = df[f"ema_{EMA_FAST}"]
            es = df[f"ema_{EMA_SLOW}"]
            et = df[f"ema_{EMA_TREND}"]
            X["ema_aligned_bull"] = ((ef > es) & (es > et)).astype(int)
            X["ema_aligned_bear"] = ((ef < es) & (es < et)).astype(int)
            X["ema_fast_slope"]   = (ef - ef.shift(3)) / ef.shift(3)
            X["ema_slow_slope"]   = (es - es.shift(3)) / es.shift(3)

        # 6. Price action (candle body & shadow analysis)
        body      = (close - open_).abs()
        full_range= (high - low).replace(0, np.nan)
        X["body_ratio"]      = body / full_range
        X["bull_candle"]     = (close > open_).astype(int)
        X["upper_shadow"]    = (high - close.where(close > open_, open_)) / full_range
        X["lower_shadow"]    = (close.where(close < open_, open_) - low) / full_range
        X["close_vs_open"]   = (close - open_) / full_range

        # 7. Multi-period momentum (rate of change)
        for p in [1, 3, 5, 10, 20]:
            X[f"roc_{p}"] = close.pct_change(p)

        # 8. High/Low channel position
        for p in [10, 20]:
            highest = high.rolling(p).max()
            lowest  = low.rolling(p).min()
            rng     = (highest - lowest).replace(0, np.nan)
            X[f"chan_pos_{p}"] = (close - lowest) / rng  # 0=bottom 1=top

        # 9. Volume ratio (jika ada)
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            vol = df["Volume"].replace(0, np.nan)
            X["vol_ratio"]  = vol / vol.rolling(20).mean()
            X["vol_spike"]  = (X["vol_ratio"] > 2.0).astype(int)

        # 10. Consecutive candle direction
        bull = (close > open_).astype(int)
        X["consec_bull"] = bull.rolling(3).sum()   # 0-3 berapa candle bull berturut
        X["consec_bear"] = (1 - bull).rolling(3).sum()

        # 11. ATR-normalized price change
        atr = df.get("atr", pd.Series(close * 0.001, index=df.index))
        X["atr_norm_chg"] = close.diff() / atr.replace(0, np.nan)

        # 12. Stochastic divergence dari RSI
        X["stoch_rsi_div"] = (X["stoch_k"] / 100) - (rsi / 100)

        # 13. Lagged values (ML_LOOKBACK candles)
        key_cols = ["rsi_norm", "macd_hist", "roc_1", "roc_3",
                    "hist_pos", "ema_fast_slope", "stoch_diff", "bb_pct"]
        for lag in range(1, min(ML_LOOKBACK + 1, 16)):   # max 15 lags
            for col in key_cols:
                if col in X.columns:
                    X[f"{col}_l{lag}"] = X[col].shift(lag)

        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ─── LABELING ─────────────────────────────────────────────
    def _make_labels(self, df: pd.DataFrame, threshold: float = None):
        """
        ATR-adaptive threshold + multi-candle lookahead:
          1 = BUY  (harga naik > threshold dalam N candle ke depan)
          0 = SELL (harga turun > threshold dalam N candle ke depan)
          NaN = sideways / tidak jelas (dibuang dari training)

        Threshold = max(fixed_floor, ATR * 0.4)
        Ini kunci utama: GOLD 5m ATR ~$10, jadi threshold ~$4 bukan $1.5
        """
        n   = self.lookahead
        thr = threshold if threshold is not None else self.label_thresh

        # ATR-adaptive: pakai mana yang lebih besar antara floor dan 40% ATR
        if "atr" in df.columns:
            atr_pct   = df["atr"] / df["Close"] * 0.4  # 40% dari 1-ATR sebagai % harga
            eff_thr   = atr_pct.clip(lower=thr)         # tidak boleh di bawah floor
        else:
            eff_thr   = thr

        # Lihat N candle ke depan: ambil close candle ke-N
        fwd_ret = df["Close"].shift(-n) / df["Close"] - 1

        label = pd.Series(np.nan, index=df.index)
        if isinstance(eff_thr, pd.Series):
            label[fwd_ret >  eff_thr] = 1
            label[fwd_ret < -eff_thr] = 0
        else:
            label[fwd_ret >  eff_thr] = 1
            label[fwd_ret < -eff_thr] = 0
        return label

    # ─── TRAIN ────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        X_raw = self._build_features(df)
        y     = self._make_labels(df)

        # Buang sideways (NaN label) - ini kunci utama peningkatan akurasi
        valid   = y.notna() & X_raw.notna().all(axis=1)
        X_clean = X_raw[valid]
        y_clean = y[valid].astype(int)

        n_sideways = int(valid.shape[0] - valid.sum())

        # Time-series split (jangan shuffle!)
        split       = int(len(X_clean) * ML_TRAIN_SPLIT)
        X_train     = X_clean.iloc[:split]
        X_test      = X_clean.iloc[split:]
        y_train     = y_clean.iloc[:split]
        y_test      = y_clean.iloc[split:]

        # Scale
        self.scaler.fit(X_train)
        Xtr_s = self.scaler.transform(X_train)
        Xte_s = self.scaler.transform(X_test)

        # Feature selection (pilih top-K fitur paling informatif)
        self.selector = SelectKBest(f_classif, k=min(self.n_features_selected, X_train.shape[1]))
        self.selector.fit(Xtr_s, y_train)
        Xtr_sel = self.selector.transform(Xtr_s)
        Xte_sel = self.selector.transform(Xte_s)

        # Train
        self.model.fit(Xtr_sel, y_train)
        y_pred = self.model.predict(Xte_sel)
        y_proba = self.model.predict_proba(Xte_sel)

        # Accuracy HANYA pada prediksi confident (proba > threshold)
        max_proba     = y_proba.max(axis=1)
        conf_mask     = max_proba >= (CONFIDENCE_THRESHOLD / 100)
        conf_accuracy = accuracy_score(y_test[conf_mask], y_pred[conf_mask]) if conf_mask.sum() > 0 else 0
        conf_coverage = conf_mask.mean() * 100

        self.accuracy      = accuracy_score(y_test, y_pred)
        self.conf_accuracy = conf_accuracy
        self.conf_coverage = round(conf_coverage, 1)
        self.report        = classification_report(y_test, y_pred, target_names=["SELL", "BUY"])
        self.trained       = True
        self.feature_names = list(X_clean.columns)

        # Simpan feature names setelah selection
        support = self.selector.get_support()
        self.selected_features = [f for f, s in zip(self.feature_names, support) if s]

        return {
            "accuracy":          round(self.accuracy * 100, 2),
            "conf_accuracy":     round(conf_accuracy * 100, 2),
            "conf_coverage":     self.conf_coverage,
            "precision_buy":     round(precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100, 2),
            "recall_buy":        round(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100, 2),
            "f1":                round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "report":            self.report,
            "n_train":           len(X_train),
            "n_test":            len(X_test),
            "n_sideways_removed":int(n_sideways),
            "n_features":        len(self.selected_features),
        }

    # ─── PREDICT ──────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict:
        if not self.trained:
            return {
                "direction": "UNKNOWN", "confidence": 0.0,
                "proba_buy": 0.5, "proba_sell": 0.5, "uncertain": True,
            }

        X_raw = self._build_features(df)
        last  = X_raw.iloc[[-1]].fillna(0)

        # Pastikan kolom sama
        for col in self.feature_names:
            if col not in last.columns:
                last[col] = 0
        last = last[self.feature_names]

        Xs   = self.scaler.transform(last)
        Xsel = self.selector.transform(Xs)

        pred  = self.model.predict(Xsel)[0]
        proba = self.model.predict_proba(Xsel)[0]

        proba_sell = round(float(proba[0]) * 100, 1)
        proba_buy  = round(float(proba[1]) * 100, 1)
        confidence = max(proba_buy, proba_sell)
        direction  = "BUY" if pred == 1 else "SELL"
        uncertain  = confidence < CONFIDENCE_THRESHOLD

        return {
            "direction":  "WAIT" if uncertain else direction,
            "confidence": confidence,
            "proba_buy":  proba_buy,
            "proba_sell": proba_sell,
            "uncertain":  uncertain,
        }

    # ─── FEATURE IMPORTANCE ───────────────────────────────────
    def feature_importance(self, top_n: int = 10) -> list:
        if not self.trained:
            return []
        try:
            imp  = self.model.feature_importances_
            pairs = sorted(zip(self.selected_features, imp), key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        except AttributeError:
            return []
