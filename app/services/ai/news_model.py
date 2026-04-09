"""
NewsDirectionModel — ML yang ditraining dari berita → arah harga nyata.

Flow:
  1. Kumpulkan news cache harian (dari news_cache/)
  2. Match setiap hari ke pergerakan harga nyata (dari price_df)
  3. Ekstrak fitur dari news (skor, jumlah, tipe event, dll)
  4. Train LogisticRegression → predict BUY/SELL/WAIT
  5. Tiap hari baru ada data → model di-retrain otomatis
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.core.paths import CACHE_DIR as _CACHE_DIR

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = str(_CACHE_DIR)
MODEL_MIN_SAMPLES = 3   # minimal hari untuk training (cukup 3 hari)

DIRECTION_KEYWORDS_GOLD = None   # lazy import dari news_filter


def _load_direction_keywords():
    global DIRECTION_KEYWORDS_GOLD
    if DIRECTION_KEYWORDS_GOLD is None:
        try:
            from app.services.news.news_filter import DIRECTION_KEYWORDS_GOLD as dk
            DIRECTION_KEYWORDS_GOLD = dk
        except Exception:
            DIRECTION_KEYWORDS_GOLD = {}
    return DIRECTION_KEYWORDS_GOLD


class NewsDirectionModel:
    """
    Ditraining dari pasangan (fitur_berita_hari_X → arah_harga_hari_X+1).
    Fitur:
      - keyword_score_bull  : total skor bullish dari semua berita hari itu
      - keyword_score_bear  : total skor bearish
      - net_score           : bull - bear
      - high_count          : jumlah berita HIGH impact
      - medium_count        : jumlah berita MEDIUM impact
      - n_sources           : jumlah sumber berbeda
      - has_cal_high        : ada kalender HIGH hari ini (1/0)
      - cal_beat            : kalender actual > forecast (1/0)
      - cal_miss            : kalender actual < forecast (1/0)
    Label:
      - 1 = harga naik (BUY)
      - 0 = harga turun (SELL)
    """

    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol   = symbol.upper()
        self.model    = LogisticRegression(C=1.0, max_iter=500,
                                           class_weight="balanced",
                                           random_state=42)
        self.scaler   = StandardScaler()
        self.trained  = False
        self.accuracy = 0.0
        self.n_train  = 0
        self.feature_names = [
            "keyword_score_bull", "keyword_score_bear", "net_score",
            "high_count", "medium_count", "n_sources",
            "has_cal_high", "cal_beat", "cal_miss",
        ]

    # ──────────────────────────────────────────────────────────────
    def _extract_features(self, day_data: dict) -> np.ndarray:
        """Ekstrak vektor fitur dari satu hari cache news."""
        news     = day_data.get("news", [])
        calendar = day_data.get("calendar", [])
        dir_kw   = _load_direction_keywords()

        bull_score = 0.0
        bear_score = 0.0
        sources    = set()

        for item in news:
            title = item.get("title", "").lower()
            impact = item.get("impact", "LOW")
            src    = item.get("source_api", "unknown")
            sources.add(src)
            imp_w = 1.5 if impact == "HIGH" else 1.0 if impact == "MEDIUM" else 0.6
            for kw, (kw_score, _) in dir_kw.items():
                if kw in title:
                    if kw_score > 0:
                        bull_score += kw_score * imp_w
                    else:
                        bear_score += abs(kw_score) * imp_w
                    break

        high_count   = sum(1 for n in news if n.get("impact") == "HIGH")
        medium_count = sum(1 for n in news if n.get("impact") == "MEDIUM")

        # Calendar
        has_cal_high = 0
        cal_beat     = 0
        cal_miss     = 0
        for ev in calendar:
            if ev.get("impact", "LOW") == "HIGH":
                has_cal_high = 1
                actual   = ev.get("actual", "")
                forecast = ev.get("forecast", "")
                if actual and forecast:
                    try:
                        av = float(actual.replace("K","000").replace("%","").strip())
                        fv = float(forecast.replace("K","000").replace("%","").strip())
                        if av > fv:
                            cal_beat = 1
                        elif av < fv:
                            cal_miss = 1
                    except ValueError:
                        pass

        return np.array([
            bull_score,
            bear_score,
            bull_score - bear_score,
            high_count,
            medium_count,
            len(sources),
            has_cal_high,
            cal_beat,
            cal_miss,
        ], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────
    def train(self, price_df: pd.DataFrame) -> dict:
        """
        Load semua cache news, match ke price_df, train model.
        price_df: DataFrame dengan index datetime dan kolom 'Close'
        """
        if not os.path.exists(CACHE_DIR):
            return {"error": "news_cache tidak ditemukan", "trained": False}

        # Kumpulkan semua file cache
        records = []
        for fname in sorted(os.listdir(CACHE_DIR)):
            if not fname.endswith(".json"):
                continue
            if self.symbol.upper() not in fname.upper() and \
               "GOLD" not in fname.upper() and "XAUUSD" not in fname.upper():
                if self.symbol not in ("XAUUSD", "GOLD"):
                    continue

            fpath = os.path.join(CACHE_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            date_str = data.get("date", "")
            if not date_str:
                continue

            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            feat = self._extract_features(data)
            records.append({"date": date, "features": feat, "data": data})

        if len(records) < MODEL_MIN_SAMPLES:
            return {
                "error": f"Cache berita hanya {len(records)} hari (min {MODEL_MIN_SAMPLES})",
                "trained": False,
                "n_days": len(records),
            }

        # Match ke price — label: apakah harga naik esok hari?
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        # Resample ke harian
        daily_close = price_df["Close"].resample("D").last().dropna()

        X, y, dates_used = [], [], []
        for rec in records:
            date      = rec["date"]
            next_date = date + timedelta(days=1)

            # Cari close hari ini dan besok
            try:
                c_today = daily_close.asof(date)
                c_next  = daily_close.asof(next_date)
            except Exception:
                continue

            if pd.isna(c_today) or pd.isna(c_next) or c_today == 0:
                continue

            label = 1 if c_next > c_today else 0   # 1=BUY, 0=SELL
            X.append(rec["features"])
            y.append(label)
            dates_used.append(date.strftime("%Y-%m-%d"))

        if len(X) < MODEL_MIN_SAMPLES:
            return {
                "error": f"Pasangan news-price hanya {len(X)} hari",
                "trained": False,
            }

        X = np.array(X)
        y = np.array(y)

        # Butuh minimal 2 kelas (BUY dan SELL) untuk LogisticRegression
        if len(np.unique(y)) < 2:
            only = "SELL" if y[0] == 0 else "BUY"
            return {
                "error": f"Semua {len(y)} hari cache hanya kelas {only} — butuh data lebih bervariasi",
                "trained": False,
                "n_days": len(y),
            }

        # Train/test split — kalau data sedikit, pakai semua untuk train
        n = len(X)
        split = max(int(n * 0.8), n - 2) if n >= 5 else n
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        # Pastikan train set juga punya 2 kelas
        if len(np.unique(y_tr)) < 2:
            # Pindahkan satu sampel dari test ke train supaya ada 2 kelas
            X_tr = np.vstack([X_tr, X_te[:1]])
            y_tr = np.append(y_tr, y_te[:1])
            X_te = X_te[1:]
            y_te = y_te[1:]

        self.scaler.fit(X_tr)
        X_tr_s = self.scaler.transform(X_tr)
        X_te_s = self.scaler.transform(X_te) if len(X_te) > 0 else X_tr_s[:0]

        self.model.fit(X_tr_s, y_tr)
        if len(X_te_s) > 0:
            y_pred  = self.model.predict(X_te_s)
            y_proba = self.model.predict_proba(X_te_s)
        else:
            y_pred  = np.array([], dtype=int)
            y_proba = np.empty((0, 2))

        # ── Metrics ───────────────────────────────────────────────────
        has_test = len(y_te) > 0 and len(np.unique(y_te)) > 1

        self.trained   = True
        self.accuracy  = round(accuracy_score(y_te, y_pred) * 100, 1) if len(y_te) > 0 else 0
        self.n_train   = len(X_tr)
        self.precision = round(precision_score(y_te, y_pred, zero_division=0) * 100, 1) if has_test else 0
        self.recall    = round(recall_score(y_te, y_pred, zero_division=0) * 100, 1) if has_test else 0
        self.f1        = round(f1_score(y_te, y_pred, zero_division=0) * 100, 1) if has_test else 0
        self.report    = classification_report(y_te, y_pred,
                             target_names=["SELL","BUY"],
                             zero_division=0) if has_test else ""

        # ── Feature importance (koefisien LogReg) ─────────────────────
        coefs = self.model.coef_[0]
        feat_imp = sorted(
            zip(self.feature_names, coefs),
            key=lambda x: abs(x[1]), reverse=True
        )
        self.feat_importance = feat_imp

        # Distribusi label
        n_buy  = int(np.sum(y == 1))
        n_sell = int(np.sum(y == 0))

        # Confidence coverage pada test set
        max_proba = y_proba.max(axis=1)
        conf_mask = max_proba >= 0.60
        conf_acc  = round(accuracy_score(y_te[conf_mask], y_pred[conf_mask]) * 100, 1) \
                    if conf_mask.sum() > 0 else 0
        conf_cov  = round(conf_mask.mean() * 100, 1)

        return {
            "trained":      True,
            "accuracy":     self.accuracy,
            "precision":    self.precision,
            "recall":       self.recall,
            "f1":           self.f1,
            "conf_accuracy": conf_acc,
            "conf_coverage": conf_cov,
            "n_train":      self.n_train,
            "n_test":       len(X_te),
            "n_days":       len(X),
            "n_buy_days":   n_buy,
            "n_sell_days":  n_sell,
            "feat_importance": feat_imp,
            "dates_range":  f"{dates_used[0]} s/d {dates_used[-1]}" if dates_used else "",
            "report":       self.report,
        }

    # ──────────────────────────────────────────────────────────────
    def predict(self, today_news: list, today_calendar: list) -> dict:
        """
        Prediksi arah dari berita hari ini.
        Returns: {direction, confidence, proba_buy, proba_sell, uncertain}
        """
        if not self.trained:
            return {
                "direction": "WAIT", "confidence": 0.0,
                "proba_buy": 50.0,   "proba_sell": 50.0,
                "uncertain": True,   "warning": "Model belum ditraining",
            }

        day_data = {"news": today_news, "calendar": today_calendar}
        feat     = self._extract_features(day_data).reshape(1, -1)
        feat_s   = self.scaler.transform(feat)

        pred  = self.model.predict(feat_s)[0]
        proba = self.model.predict_proba(feat_s)[0]

        proba_buy  = round(float(proba[1]) * 100, 1)
        proba_sell = round(float(proba[0]) * 100, 1)
        confidence = max(proba_buy, proba_sell)
        direction  = "BUY" if pred == 1 else "SELL"
        uncertain  = confidence < 60.0

        return {
            "direction":  "WAIT" if uncertain else direction,
            "confidence": confidence,
            "proba_buy":  proba_buy,
            "proba_sell": proba_sell,
            "uncertain":  uncertain,
        }
