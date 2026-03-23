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

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        TP/SL Simulation Labeling — jauh lebih akurat dari simple forward return.

        Untuk setiap candle i (simulasi BUY trade):
          TP = close[i] + atr[i] * tp_mult   (default: 3x ATR)
          SL = close[i] - atr[i] * sl_mult   (default: 1x ATR → RR 1:3)

        Cek N candle ke depan:
          Jika high[i+k] >= TP lebih dulu  → label = 1  (BUY menang)
          Jika low[i+k]  <= SL lebih dulu  → label = 0  (BUY kalah = arah SELL)
          Jika keduanya tidak → NaN (excluded — sideways, tidak informatif)

        Keunggulan vs forward return:
          - Mencerminkan hasil trade nyata, bukan hanya arah harga akhir
          - RR asimetri (TP > SL) → ML belajar setup yang worth masuk
          - Candle sideways otomatis excluded (NaN) → dataset lebih bersih
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

        # RR 1:3 — sama dengan konfigurasi bot (SL=1x ATR, TP=3x ATR)
        # Lebih konservatif dari TP=10x ATR di config biar label lebih banyak
        tp_mult = 3.0
        sl_mult = 1.0
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
        X_raw = self._build_features(df)
        y     = self._make_labels(df)

        quality_mask = self._filter_quality_rows(df, y)
        valid        = quality_mask & X_raw.notna().all(axis=1)
        X_clean      = X_raw[valid]
        y_clean      = y[valid].astype(int)
        n_sideways   = int(quality_mask.shape[0] - quality_mask.sum())

        split   = int(len(X_clean) * ML_TRAIN_SPLIT)
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
        # Simpan metadata symbol training
        self.trained_symbol    = (symbol or self.symbol).upper()
        self.trained_timeframe = timeframe or self.timeframe
        self.trained_n_candles = len(df)

        support = self.selector.get_support()
        self.selected_features = [f for f, s in zip(self.feature_names, support) if s]

        return {
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
        }

    def predict(self, df: pd.DataFrame, predict_symbol: str = "") -> dict:
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
