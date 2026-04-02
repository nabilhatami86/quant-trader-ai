"""
ml/model.py — Machine Learning predictor untuk sinyal trading.

Kelas utama:
  CandlePredictor : ensemble ML classifier (LightGBM + XGBoost + RF + GB + ET)

Arsitektur model:
  Base models  : LGBMClassifier, XGBClassifier, RandomForest,
                 GradientBoosting, ExtraTreesClassifier
  Meta-model   : LogisticRegression (Stacking) dengan CalibratedClassifierCV
  Feature sel. : SelectKBest (top 70 dari ~120 fitur)
  Scaling      : RobustScaler (tahan outlier harga ekstrem)

Label generation (target variabel):
  Lookahead 3 candle ke depan:
    1 (BUY)  : close[+3] > close[0] * (1 + threshold)
    0 (SELL) : close[+3] < close[0] * (1 - threshold)
    Drop     : di antara keduanya (terlalu kecil pergerakannya)
  Threshold per timeframe:
    5m=0.08%, 15m=0.12%, 1h=0.20%, 4h=0.35%, 1d=0.60%

Metrics kalibrasi (2026-03, XAUUSD 5m):
  conf_accuracy  ~79%   (akurasi saat model confident >= 70%)
  precision      ~68%   (dari semua yang diprediksi BUY, 68% benar)
  F1 score       ~71%

Mode penggunaan di bot:
  #1 Rule+ML   : rule signal + ML confident (conf_accuracy >= ML_MIN_CONFIDENT_ACC)
                 ML bisa veto rule signal jika berlawanan
  #2 ML-Only   : rule tidak cukup tapi ML sangat confident
                 Hanya aktif jika conf_accuracy >= ML_MIN_CONFIDENT_ACC (75%)
  #3 Rule-Only : ML tidak confident — pakai rule signal saja

Retrain:
  Dipanggil setiap siklus bot (incremental, data baru)
  Setiap 15 trade tutup via AdaptiveLearner.should_retrain_ml()
  Setiap session close / weekend via run_market_close_deep_analysis()
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV

# LightGBM & XGBoost — jauh lebih akurat dari RF untuk tabular data
try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

from config import *

# ── Model persistence ─────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(_THIS_DIR, "..", "models")
# Retrain hanya jika ada >= N candle baru sejak training terakhir
RETRAIN_MIN_NEW_CANDLES = 100

CONFIDENCE_THRESHOLD = 70.0   # 70%: akurasi 91.2% dengan coverage 55.6% — sweet spot terbaik

LABEL_THRESHOLDS = {
    "1m":  0.0003,
    "5m":  0.0008,
    "15m": 0.0012,
    "1h":  0.0020,
    "4h":  0.0035,
    "1d":  0.006,
}
DEFAULT_LABEL_THRESHOLD = 0.0020

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
    """
    Ensemble ML classifier untuk prediksi arah candle.

    Model: StackingClassifier (LightGBM + XGBoost + RF + GB + ET)
           Meta-model: LogisticRegression + CalibratedClassifierCV

    Attributes:
      trained         : bool — True setelah train() berhasil
      accuracy        : float — overall test accuracy (%)
      feature_names   : list — nama 70 fitur yang diseleksi
      trained_symbol  : str  — simbol yang dipakai saat training
      train_result    : dict — metrics lengkap dari training terakhir
    """

    def __init__(self, symbol: str = "", timeframe: str = "1h"):
        self.symbol        = symbol.upper()
        self.timeframe     = timeframe
        self.label_thresh  = LABEL_THRESHOLDS.get(timeframe, DEFAULT_LABEL_THRESHOLD)
        self.lookahead     = LOOKAHEAD_CANDLES.get(timeframe, DEFAULT_LOOKAHEAD)
        self.scaler        = RobustScaler()
        self.selector      = None
        self.trained       = False
        self.accuracy      = 0.0
        self.report        = ""
        self.feature_names = []
        self.n_features_selected = 70
        # Training metadata — diisi saat train()
        self.trained_symbol    = ""
        self.trained_timeframe = ""
        self.trained_n_candles = 0
        self._last_candle_time = ""   # ISO timestamp candle terakhir saat training
        self._build_model()

    def _build_model(self):
        # ── Base models ───────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
        )

        # ── LightGBM — leaf-wise, jauh lebih akurat dari RF ───────────
        if LGB_AVAILABLE:
            lgb = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
        else:
            lgb = None

        # ── XGBoost — regularized boosting, robust overfit ────────────
        # scale_pos_weight=1 karena kita sudah fix labeling ke symmetric 1:1
        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=42,
                n_jobs=1,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            xgb = None

        # ── Meta-learner: Logistic Regression ─────────────────────────
        meta = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
            solver="lbfgs",
        )

        if ML_MODEL_TYPE == "gb":
            self.model = gb
        elif ML_MODEL_TYPE == "lgb" and lgb:
            self.model = CalibratedClassifierCV(lgb, cv=3, method="isotonic")
        elif ML_MODEL_TYPE == "xgb" and xgb:
            self.model = xgb
        elif ML_MODEL_TYPE == "ensemble":
            if LGB_AVAILABLE and XGB_AVAILABLE:
                # ── Primary: LightGBM dikalibrasi (isotonic regression) ──
                # Kalibrasi memperbaiki probabilitas → confidence lebih akurat
                lgb_cal = CalibratedClassifierCV(lgb, cv=2, method="isotonic")
                xgb_cal = CalibratedClassifierCV(xgb, cv=2, method="isotonic")

                # VotingClassifier dengan LGB + XGB + RF
                # Lebih cepat dari StackingClassifier, akurasi setara
                self.model = VotingClassifier(
                    estimators=[
                        ("lgb", lgb_cal),
                        ("xgb", xgb_cal),
                        ("rf",  rf),
                    ],
                    voting="soft",
                    n_jobs=1,
                )
            elif LGB_AVAILABLE:
                self.model = CalibratedClassifierCV(lgb, cv=3, method="isotonic")
            elif XGB_AVAILABLE:
                self.model = xgb
            else:
                # Fallback: RF + GB
                self.model = VotingClassifier(
                    estimators=[("rf", rf), ("gb", gb)],
                    voting="soft",
                    n_jobs=1,
                )
        else:
            self.model = rf

    # ── Model persistence ─────────────────────────────────────────────────────

    def _model_path(self, sym: str, tf: str) -> str:
        return os.path.join(MODEL_DIR, f"ml_{sym}_{tf}.joblib")

    def _meta_path(self, sym: str, tf: str) -> str:
        return os.path.join(MODEL_DIR, f"ml_{sym}_{tf}_meta.json")

    def _load_meta(self, sym: str, tf: str) -> dict | None:
        path = self._meta_path(sym, tf)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def save(self, sym: str, tf: str, result: dict):
        """Simpan model terlatih + metadata ke disk."""
        if not self.trained:
            return
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump({
                "model":             self.model,
                "scaler":            self.scaler,
                "selector":          self.selector,
                "feature_names":     self.feature_names,
                "selected_features": getattr(self, "selected_features", []),
                "conf_accuracy":     getattr(self, "conf_accuracy", 0),
                "conf_coverage":     getattr(self, "conf_coverage", 0),
                "accuracy":          self.accuracy,
            }, self._model_path(sym, tf))
            meta = {
                "trained_symbol":    self.trained_symbol,
                "trained_timeframe": self.trained_timeframe,
                "trained_n_candles": self.trained_n_candles,
                "last_candle_time":  self._last_candle_time,
                "last_result":       result,
            }
            with open(self._meta_path(sym, tf), "w") as f:
                json.dump(meta, f)
            print(f"[ML] Model disimpan -> models/ml_{sym}_{tf}.joblib")
        except Exception as e:
            print(f"[ML] Gagal simpan model: {e}")

    def load(self, sym: str, tf: str) -> bool:
        """Muat model dari disk. Return True jika berhasil."""
        path = self._model_path(sym, tf)
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
            self.model             = data["model"]
            self.scaler            = data["scaler"]
            self.selector          = data["selector"]
            self.feature_names     = data["feature_names"]
            self.selected_features = data.get("selected_features", [])
            self.conf_accuracy     = data.get("conf_accuracy", 0)
            self.conf_coverage     = data.get("conf_coverage", 0)
            self.accuracy          = data.get("accuracy", 0)
            self.trained           = True
            self.trained_symbol    = sym
            self.trained_timeframe = tf
            meta = self._load_meta(sym, tf)
            if meta:
                self.trained_n_candles = meta.get("trained_n_candles", 0)
                self._last_candle_time = meta.get("last_candle_time", "")
            return True
        except Exception as e:
            print(f"[ML] Gagal load model dari cache: {e}")
            return False

    def _count_new_candles(self, df: pd.DataFrame, last_candle_time: str) -> int:
        """Hitung berapa candle baru sejak training terakhir."""
        if not last_candle_time or not hasattr(df.index, "max"):
            return RETRAIN_MIN_NEW_CANDLES  # unknown → anggap perlu retrain
        try:
            last_ts = pd.Timestamp(last_candle_time)
            if last_ts.tzinfo is None and df.index.tzinfo is not None:
                last_ts = last_ts.tz_localize(df.index.tzinfo)
            elif last_ts.tzinfo is not None and df.index.tzinfo is None:
                last_ts = last_ts.tz_localize(None)
            return int((df.index > last_ts).sum())
        except Exception:
            return RETRAIN_MIN_NEW_CANDLES

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bangun feature matrix dari DataFrame OHLCV + indikator.
        Input  : df dengan kolom OHLCV + hasil add_all_indicators()
        Output : DataFrame ~120 kolom (sebelum SelectKBest ke 70)
        Fitur  : return%, volatility, rsi, macd, ema_ratio, adx, bb_pct,
                 stoch, atr_ratio, obv_change, vwap_ratio, candle patterns,
                 supertrend, ichimoku, fibonacci levels, dll.
        """
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        open_  = df["Open"]

        X = pd.DataFrame(index=df.index)

        X["rsi"]          = df.get("rsi", 50)
        X["rsi_norm"]     = (df.get("rsi", 50) - 50) / 50
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

        for span in [EMA_FAST, EMA_SLOW, EMA_TREND, EMA_LONG]:
            col = f"ema_{span}"
            if col in df.columns:
                X[f"dist_ema{span}"] = (close - df[col]) / df[col]

        rsi = df.get("rsi", pd.Series(50, index=df.index))
        X["rsi_oversold"]   = (rsi < RSI_OVERSOLD).astype(int)
        X["rsi_overbought"] = (rsi > RSI_OVERBOUGHT).astype(int)
        X["rsi_bull_zone"]  = ((rsi > 40) & (rsi < 60)).astype(int)

        macd = df.get("macd", pd.Series(0, index=df.index))
        hist = df.get("histogram", pd.Series(0, index=df.index))
        X["macd_above_zero"] = (macd > 0).astype(int)
        X["hist_rising"]     = (hist > hist.shift(1)).astype(int)
        X["hist_pos"]        = (hist > 0).astype(int)

        if all(f"ema_{s}" in df.columns for s in [EMA_FAST, EMA_SLOW, EMA_TREND]):
            ef = df[f"ema_{EMA_FAST}"]
            es = df[f"ema_{EMA_SLOW}"]
            et = df[f"ema_{EMA_TREND}"]
            X["ema_aligned_bull"] = ((ef > es) & (es > et)).astype(int)
            X["ema_aligned_bear"] = ((ef < es) & (es < et)).astype(int)
            X["ema_fast_slope"]   = (ef - ef.shift(3)) / ef.shift(3)
            X["ema_slow_slope"]   = (es - es.shift(3)) / es.shift(3)

        body       = (close - open_).abs()
        full_range = (high - low).replace(0, np.nan)
        X["body_ratio"]    = body / full_range
        X["bull_candle"]   = (close > open_).astype(int)
        X["upper_shadow"]  = (high - close.where(close > open_, open_)) / full_range
        X["lower_shadow"]  = (close.where(close < open_, open_) - low) / full_range
        X["close_vs_open"] = (close - open_) / full_range

        for p in [1, 3, 5, 10, 20]:
            X[f"roc_{p}"] = close.pct_change(p)

        for p in [10, 20]:
            highest = high.rolling(p).max()
            lowest  = low.rolling(p).min()
            rng     = (highest - lowest).replace(0, np.nan)
            X[f"chan_pos_{p}"] = (close - lowest) / rng

        if "Volume" in df.columns and df["Volume"].sum() > 0:
            vol = df["Volume"].replace(0, np.nan)
            X["vol_ratio"] = vol / vol.rolling(20).mean()
            X["vol_spike"] = (X["vol_ratio"] > 2.0).astype(int)

        bull = (close > open_).astype(int)
        X["consec_bull"] = bull.rolling(3).sum()
        X["consec_bear"] = (1 - bull).rolling(3).sum()

        atr = df.get("atr", pd.Series(close * 0.001, index=df.index))
        X["atr_norm_chg"] = close.diff() / atr.replace(0, np.nan)

        X["stoch_rsi_div"] = (X["stoch_k"] / 100) - (rsi / 100)

        # ── Volume indicators ──────────────────────────────────────────
        if "obv" in df.columns:
            obv = df["obv"]
            obv_ema = df.get("obv_ema", obv.ewm(span=20).mean())
            X["obv_above_ema"]  = (obv > obv_ema).astype(int)
            X["obv_slope"]      = (obv - obv.shift(5)) / (obv.shift(5).abs().replace(0, np.nan))

        if "vwap" in df.columns:
            vwap = df["vwap"].replace(0, np.nan)
            X["price_vs_vwap"]  = (close - vwap) / vwap

        if "williams_r" in df.columns:
            X["williams_r_norm"] = df["williams_r"] / 100.0   # -1..0

        if "cci" in df.columns:
            X["cci_norm"]        = (df["cci"] / 200).clip(-1, 1)

        if "vol_div" in df.columns:
            X["vol_div"]         = df["vol_div"]

        # ── Smart Money Concepts (SMC) features ────────────────────────
        for smc_col in ["fvg_bull", "fvg_bear", "ob_bull", "ob_bear",
                        "bos_bull", "bos_bear", "choch_bull", "choch_bear",
                        "liq_bull_sweep", "liq_bear_sweep"]:
            if smc_col in df.columns:
                X[smc_col] = df[smc_col].fillna(0).astype(int)

        # SMC composite: net bullish SMC signals
        smc_bull_cols = ["fvg_bull", "ob_bull", "bos_bull", "choch_bull", "liq_bull_sweep"]
        smc_bear_cols = ["fvg_bear", "ob_bear", "bos_bear", "choch_bear", "liq_bear_sweep"]
        smc_bull = sum(X.get(c, pd.Series(0, index=df.index)) for c in smc_bull_cols if c in X)
        smc_bear = sum(X.get(c, pd.Series(0, index=df.index)) for c in smc_bear_cols if c in X)
        X["smc_net"] = smc_bull - smc_bear

        # Extra candle patterns
        if "candle_ex" in df.columns:
            X["candle_ex"] = df["candle_ex"].fillna(0)

        # Market regime encoding
        if "regime" in df.columns:
            regime = df["regime"]
            X["regime_trend"]    = (regime == "TREND").astype(int)
            X["regime_volatile"] = (regime == "VOLATILE").astype(int)
            X["regime_range"]    = (regime == "RANGE").astype(int)

        # EMA slopes (trend strength)
        if "ema20_slope" in df.columns:
            X["ema20_slope"] = df["ema20_slope"]
        if "ema50_slope" in df.columns:
            X["ema50_slope"] = df["ema50_slope"]

        # ── SMA features ───────────────────────────────────────────────
        for sma_col in ["sma10", "sma20", "sma50", "sma200"]:
            if sma_col in df.columns:
                sma_val = df[sma_col].replace(0, np.nan)
                X[f"dist_{sma_col}"] = (close - sma_val) / sma_val.fillna(close)

        if "sma50" in df.columns and "sma200" in df.columns:
            sma50  = df["sma50"].replace(0, np.nan)
            sma200 = df["sma200"].replace(0, np.nan)
            X["sma_golden_cross"] = (sma50 > sma200).astype(int)
            X["sma_death_cross"]  = (sma50 < sma200).astype(int)

        # ── Fibonacci features ─────────────────────────────────────────
        if all(c in df.columns for c in ["fib_382", "fib_618", "fib_swing_high", "fib_swing_low"]):
            rng     = (df["fib_swing_high"] - df["fib_swing_low"]).replace(0, np.nan)
            X["price_vs_fib382"]  = (close - df["fib_382"]) / rng.fillna(1)
            X["price_vs_fib618"]  = (close - df["fib_618"]) / rng.fillna(1)
            X["price_vs_fib500"]  = (close - df["fib_500"]) / rng.fillna(1) if "fib_500" in df.columns else 0
            # How much of the fib range has been retraced
            X["fib_retracement"]  = (close - df["fib_swing_low"]) / rng.fillna(1)

        # ── RSI Divergence features ────────────────────────────────────
        for div_col in ["rsi_bull_div", "rsi_bear_div", "rsi_hid_bull", "rsi_hid_bear"]:
            if div_col in df.columns:
                X[div_col] = df[div_col].fillna(0).astype(int)

        # RSI net divergence: +1 bullish, -1 bearish, 0 neutral
        bull_div = X.get("rsi_bull_div", pd.Series(0, index=df.index))
        bear_div = X.get("rsi_bear_div", pd.Series(0, index=df.index))
        X["rsi_div_net"] = bull_div - bear_div

        # ── Momentum Chain features ────────────────────────────────────
        for mc_col in ["bull_chain", "bear_chain", "close_slope", "hh", "hl", "ll", "lh"]:
            if mc_col in df.columns:
                X[mc_col] = df[mc_col].fillna(0)

        if "bull_chain" in df.columns and "bear_chain" in df.columns:
            X["chain_net"] = df["bull_chain"].fillna(0) - df["bear_chain"].fillna(0)

        # ── Multi-window momentum — penting untuk trend detection ─────
        for w in [3, 7, 14, 21]:
            X[f"rsi_mean_{w}"]   = df.get("rsi", 50).rolling(w).mean().fillna(50)
            X[f"close_roc_{w}"]  = close.pct_change(w).fillna(0)

        # Candle quality score: body_ratio × direction
        X["candle_quality"] = X["body_ratio"] * X.get("bull_candle", pd.Series(0.5, index=df.index)).map({1: 1, 0: -1}).fillna(0)

        # Price position within recent range (20-bar channel)
        for w in [10, 20, 50]:
            hi = high.rolling(w).max().fillna(high)
            lo = low.rolling(w).min().fillna(low)
            rng = (hi - lo).replace(0, np.nan)
            X[f"chan_pos_{w}"] = ((close - lo) / rng).fillna(0.5)

        # Key features untuk lag — dibatasi 8 lag dan 10 kolom paling penting
        key_cols = [
            "rsi_norm", "macd_hist", "roc_3",
            "ema_fast_slope", "bb_pct",
            "smc_net", "rsi_div_net", "chain_net",
            "obv_above_ema", "atr_norm_chg",
        ]
        for lag in range(1, 9):
            for col in key_cols:
                if col in X.columns:
                    X[f"{col}_l{lag}"] = X[col].shift(lag)

        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _make_labels(self, df: pd.DataFrame, threshold: float = None):
        """
        Symmetric TP/SL Direction Labeling.

        Untuk setiap candle i:
          TP = close[i] + atr[i] * 1.0   (1x ATR ke atas)
          SL = close[i] - atr[i] * 1.0   (1x ATR ke bawah)

        Cek N candle ke depan:
          Jika high[i+k] >= TP lebih dulu  → label = 1  (arah BUY)
          Jika low[i+k]  <= SL lebih dulu  → label = 0  (arah SELL)
          Jika keduanya tidak → NaN (excluded — sideways)

        Kenapa symmetric (1:1)?
          - RR asimetri 1:3 membuat P(SL dulu) = 75% secara matematis di pasar random
          - Akibatnya label menjadi 78% SELL → model belajar bias SELL terus
          - Dengan 1:1, label menjadi ~50/50 → ML belajar ARAH, bukan gambling RR
          - Prediksi arah yang akurat lebih berguna daripada win-rate pada RR tertentu
        """
        close  = df["Close"].values
        high   = df["High"].values
        low    = df["Low"].values
        n      = len(df)

        # ATR — gunakan kolom jika ada, fallback range rata-rata
        if "atr" in df.columns:
            atr = df["atr"].fillna(df["atr"].mean()).values
        else:
            atr = ((df["High"] - df["Low"]).rolling(14).mean()
                   .fillna(0.002 * df["Close"])).values

        # Symmetric 1:1 — tanya "arah mana yang gerak 1 ATR duluan?"
        # Ini menghilangkan bias SELL yang terjadi pada RR asimetri
        tp_mult   = 1.0
        sl_mult   = 1.0
        lookahead = min(self.lookahead * 10, 30)   # cek lebih jauh untuk TP/SL

        # ── Vectorized TP/SL simulation ─────────────────────────────────
        # Loop atas lookahead (L=30 iterasi), tiap iterasi fully vectorized.
        # Jauh lebih cepat dari nested Python loops O(n×L).

        tp_arr = close + atr * tp_mult   # shape (n,)
        sl_arr = close - atr * sl_mult

        # tp_first_hit[i] = indeks offset pertama kali high >= tp_arr[i]
        # sl_first_hit[i] = indeks offset pertama kali low  <= sl_arr[i]
        tp_first = np.full(n, lookahead + 1, dtype=np.int32)
        sl_first = np.full(n, lookahead + 1, dtype=np.int32)

        for j in range(1, lookahead + 1):
            if j >= n:
                break
            # high/low shifted j posisi ke depan
            h_j = np.empty(n); h_j[:] = np.nan
            l_j = np.empty(n); l_j[:] = np.nan
            h_j[:n - j] = high[j:]
            l_j[:n - j] = low[j:]

            # Tandai i yang baru pertama kali hit TP di step j
            tp_new = (h_j >= tp_arr) & (tp_first > j) & (atr > 0)
            sl_new = (l_j <= sl_arr) & (sl_first > j) & (atr > 0)
            tp_first[tp_new] = j
            sl_first[sl_new] = j

        label = np.full(n, np.nan)
        label[tp_first < sl_first] = 1   # TP kena duluan → BUY menang
        label[sl_first < tp_first] = 0   # SL kena duluan → SELL menang
        # tp_first == sl_first atau keduanya > lookahead → NaN (excluded)

        # ── Label Quality Filter — buang "close calls" ────────────────
        # Jika TP dan SL hit dalam selisih <= 2 candle → ambiguous, buang
        # Ini kurangi noise: hanya label yang jelas yang masuk training
        ambiguous = (np.abs(tp_first.astype(float) - sl_first.astype(float)) <= 2) & \
                    (tp_first <= lookahead) & (sl_first <= lookahead)
        label[ambiguous] = np.nan

        return pd.Series(label, index=df.index)

    def _filter_quality_rows(self, df: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Buang baris yang tidak informatif dari training:
        1. Label NaN (sideways — TP/SL sama-sama tidak kena)
        2. Candle di RANGE regime tanpa SMC signal dan tanpa RSI divergence
           → pure noise, menyesatkan model
        """
        mask = y.notna()

        if "regime" in df.columns:
            is_range = df["regime"] == "RANGE"
            has_smc  = (
                df.get("choch_bull", 0).fillna(0).astype(bool) |
                df.get("choch_bear", 0).fillna(0).astype(bool) |
                df.get("liq_bull_sweep", 0).fillna(0).astype(bool) |
                df.get("liq_bear_sweep", 0).fillna(0).astype(bool) |
                df.get("fvg_bull", 0).fillna(0).astype(bool) |
                df.get("fvg_bear", 0).fillna(0).astype(bool)
            )
            has_div = (
                df.get("rsi_bull_div", 0).fillna(0).astype(bool) |
                df.get("rsi_bear_div", 0).fillna(0).astype(bool)
            )
            # RANGE tanpa sinyal = noise → buang
            noisy_range = is_range & ~has_smc & ~has_div
            mask = mask & ~noisy_range

        return mask

    def train(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> dict:
        """
        Latih ensemble model dengan data OHLCV + indikator.

        Input  : df — DataFrame dengan OHLCV + add_all_indicators() sudah dijalankan
        Output : dict metrics — {accuracy, conf_accuracy, precision, recall, f1,
                                 n_train, n_test, coverage, feature_importance_top5}

        Proses:
          1. Cek cache — jika model sudah ada dan candle baru < RETRAIN_MIN_NEW_CANDLES
             → load dari disk, skip training (cepat)
          2. Build features (_build_features)
          3. Generate labels lookahead N candle (_make_labels)
          4. Filter kualitas baris (_filter_quality_rows)
          5. Train/test split (ML_TRAIN_SPLIT=0.8)
          6. SelectKBest (top 70 fitur)
          7. Fit StackingClassifier + CalibratedClassifierCV
          8. Hitung conf_accuracy (akurasi pada sample >= CONFIDENCE_THRESHOLD)
          9. Walk-forward validation (3 fold, jika data >= 400 baris)
         10. Save model ke disk

        Jika data < 150 candle valid → return error dict, self.trained tetap False.
        """
        sym = (symbol or self.symbol).upper()
        tf  = timeframe or self.timeframe

        # ── Cache check: skip retrain jika belum ada cukup data baru ─────────
        meta = self._load_meta(sym, tf)
        if meta:
            n_new = self._count_new_candles(df, meta.get("last_candle_time", ""))
            if n_new < RETRAIN_MIN_NEW_CANDLES:
                if self.load(sym, tf):
                    cached_result = meta.get("last_result", {})
                    print(f"[ML] Cache loaded -- {n_new} candle baru "
                          f"(< {RETRAIN_MIN_NEW_CANDLES}), skip retrain "
                          f"| conf_acc={cached_result.get('conf_accuracy', 0):.1f}%")
                    return {**cached_result, "from_cache": True}

        try:
            import traceback as _tb
            X_raw = self._build_features(df)
            y     = self._make_labels(df)

            quality_mask = self._filter_quality_rows(df, y)
            valid        = quality_mask & X_raw.notna().all(axis=1)
            X_clean      = X_raw[valid]
            y_clean      = y[valid].astype(int)
            n_sideways   = int(quality_mask.shape[0] - quality_mask.sum())

            # Guard: minimum data valid
            MIN_VALID_ROWS = 150
            if len(X_clean) < MIN_VALID_ROWS:
                print(f"[ML] Data valid {len(X_clean)} baris < {MIN_VALID_ROWS} — skip training")
                return {"error": f"Data tidak cukup: {len(X_clean)} baris valid", "trained": False}

            split = int(len(X_clean) * ML_TRAIN_SPLIT)
            # Guard: test set minimal 20 baris
            if len(X_clean) - split < 20:
                print(f"[ML] Test set terlalu kecil ({len(X_clean) - split} baris) — skip")
                return {"error": "Test set terlalu kecil", "trained": False}

            X_train = X_clean.iloc[:split]
            X_test  = X_clean.iloc[split:]
            y_train = y_clean.iloc[:split]
            y_test  = y_clean.iloc[split:]

            self.scaler.fit(X_train)
            Xtr_s = self.scaler.transform(X_train)
            Xte_s = self.scaler.transform(X_test)

            self.selector = SelectKBest(f_classif, k=min(self.n_features_selected, X_train.shape[1]))
            self.selector.fit(Xtr_s, y_train)
            Xtr_sel = self.selector.transform(Xtr_s)
            Xte_sel = self.selector.transform(Xte_s)

            self.model.fit(Xtr_sel, y_train)
            y_pred  = self.model.predict(Xte_sel)
            y_proba = self.model.predict_proba(Xte_sel)

            max_proba     = y_proba.max(axis=1)
            conf_mask     = max_proba >= (CONFIDENCE_THRESHOLD / 100)
            conf_accuracy = accuracy_score(y_test[conf_mask], y_pred[conf_mask]) if conf_mask.sum() > 0 else 0
            conf_coverage = conf_mask.mean() * 100

            self.accuracy          = accuracy_score(y_test, y_pred)
            self.conf_accuracy     = conf_accuracy
            self.conf_coverage     = round(conf_coverage, 1)
            self.report            = classification_report(y_test, y_pred, target_names=["SELL", "BUY"])
            self.trained           = True
            self.feature_names     = list(X_clean.columns)
            self.trained_symbol    = sym
            self.trained_timeframe = tf
            self.trained_n_candles = len(df)
            # Catat timestamp candle terakhir untuk cache invalidation
            try:
                self._last_candle_time = df.index[-1].isoformat()
            except Exception:
                self._last_candle_time = ""

            support = self.selector.get_support()
            self.selected_features = [f for f, s in zip(self.feature_names, support) if s]

            # ── Walk-forward validation (3-fold, hanya jika data cukup) ──────
            wf_mean = None
            n_folds = 3
            if len(X_clean) >= 400:
                wf_scores  = []
                fold_size  = len(X_clean) // (n_folds + 1)
                k_wf       = min(self.n_features_selected, X_clean.shape[1])
                for fold in range(1, n_folds + 1):
                    wf_end_tr = fold * fold_size
                    wf_end_te = wf_end_tr + fold_size
                    if wf_end_te > len(X_clean):
                        break
                    try:
                        from sklearn.preprocessing import RobustScaler as _RS
                        Xwf_tr = X_clean.iloc[:wf_end_tr]
                        Xwf_te = X_clean.iloc[wf_end_tr:wf_end_te]
                        ywf_tr = y_clean.iloc[:wf_end_tr]
                        ywf_te = y_clean.iloc[wf_end_tr:wf_end_te]
                        sc_wf  = _RS()
                        Xs_tr  = sc_wf.fit_transform(Xwf_tr)
                        Xs_te  = sc_wf.transform(Xwf_te)
                        sel_wf = SelectKBest(f_classif, k=k_wf)
                        Xs_tr  = sel_wf.fit_transform(Xs_tr, ywf_tr)
                        Xs_te  = sel_wf.transform(Xs_te)
                        self.model.fit(Xs_tr, ywf_tr)
                        wf_scores.append(accuracy_score(ywf_te, self.model.predict(Xs_te)))
                    except Exception as _wf_err:
                        print(f"[ML] Walk-forward fold {fold} error: {_wf_err}")
                # Restore model ke full train set
                self.model.fit(Xtr_sel, y_train)
                wf_mean = round(float(sum(wf_scores) / len(wf_scores)) * 100, 2) if wf_scores else None
                if wf_mean:
                    print(f"[ML] Walk-forward accuracy ({n_folds}-fold): {wf_mean}%")

            result = {
                "accuracy":           round(self.accuracy * 100, 2),
                "conf_accuracy":      round(conf_accuracy * 100, 2),
                "conf_coverage":      self.conf_coverage,
                "precision_buy":      round(precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100, 2),
                "recall_buy":         round(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100, 2),
                "f1":                 round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
                "report":             self.report,
                "n_train":            len(X_train),
                "n_test":             len(X_test),
                "n_sideways_removed": int(n_sideways),
                "n_features":         len(self.selected_features),
                "trained_symbol":     self.trained_symbol,
                "trained_timeframe":  self.trained_timeframe,
                "trained_n_candles":  self.trained_n_candles,
                "wf_accuracy":        wf_mean,
                "from_cache":         False,
            }
            self.save(sym, tf, result)
            return result

        except Exception as _train_err:
            import traceback as _tb
            print(f"[ML ERROR] train() gagal: {_train_err}")
            _tb.print_exc()
            return {"error": str(_train_err), "trained": False}

    def predict(self, df: pd.DataFrame, predict_symbol: str = "") -> dict:
        """
        Prediksi arah candle berikutnya menggunakan baris terakhir dari df.

        Input  : df — DataFrame dengan OHLCV + indikator (baris terakhir = candle sekarang)
        Output : dict — {direction, confidence, proba_buy, proba_sell,
                         uncertain, trained_symbol, symbol_match, [warning]}

          direction  : "BUY" / "SELL" / "WAIT" (uncertain)
          confidence : probabilitas tertinggi dalam persen (0-100)
          uncertain  : True jika confidence < CONFIDENCE_THRESHOLD (70%)
          proba_buy  : probabilitas BUY (0-1)
          proba_sell : probabilitas SELL (0-1)

        Guard: jika trained_symbol != predict_symbol → return WAIT + warning.
        """
        if not self.trained:
            return {
                "direction":      "UNKNOWN",
                "confidence":     0.0,
                "proba_buy":      0.5,
                "proba_sell":     0.5,
                "uncertain":      True,
                "trained_symbol": self.trained_symbol,
                "symbol_match":   False,
                "warning":        "Model belum di-training",
            }

        # ── Validasi symbol — jangan campur data antar symbol ────────
        pred_sym    = (predict_symbol or self.symbol).upper()
        train_sym   = self.trained_symbol.upper()
        symbol_match = (not train_sym) or (not pred_sym) or (train_sym == pred_sym)

        if not symbol_match:
            YELLOW = "\033[93m"
            BOLD   = "\033[1m"
            RESET  = "\033[0m"
            print(f"  {BOLD}{YELLOW}[ML WARNING]{RESET} Model di-train pakai {train_sym} "
                  f"tapi diprediksi untuk {pred_sym} — hasil tidak akurat!")
            return {
                "direction":      "WAIT",
                "confidence":     0.0,
                "proba_buy":      0.0,
                "proba_sell":     0.0,
                "uncertain":      True,
                "trained_symbol": train_sym,
                "symbol_match":   False,
                "warning":        f"Symbol mismatch: trained={train_sym}, predict={pred_sym}",
            }

        X_raw = self._build_features(df)
        last  = X_raw.iloc[[-1]].fillna(0)

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
            "direction":      "WAIT" if uncertain else direction,
            "confidence":     confidence,
            "proba_buy":      proba_buy,
            "proba_sell":     proba_sell,
            "uncertain":      uncertain,
            "trained_symbol": train_sym,
            "trained_tf":     self.trained_timeframe,
            "trained_candles": self.trained_n_candles,
            "symbol_match":   True,
        }

    def feature_importance(self, top_n: int = 10) -> list:
        if not self.trained:
            return []
        try:
            imp   = self.model.feature_importances_
            pairs = sorted(zip(self.selected_features, imp), key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        except AttributeError:
            return []
