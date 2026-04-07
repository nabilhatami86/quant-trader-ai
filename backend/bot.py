import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))
import time
import sys

from config import *
from data.candle_db import save_candles, merge_with_db, get_db_stats

# ── AI modules: optional — bisa di-disable dengan BOT_AI_ENABLED=false ──────
# Jika AI tidak tersedia, backend tetap jalan (API, DB, journal, MT5 tetap OK)
try:
    from ai.indicators import add_all_indicators
    from ai.signals import generate_signal
    from ai.model import CandlePredictor
    from ai.deep_model import LSTMPredictor, TF_AVAILABLE
    from data.news_filter import NewsFilter
    AI_AVAILABLE = True
except Exception as _ai_err:
    AI_AVAILABLE = False
    TF_AVAILABLE = False
    add_all_indicators = None
    generate_signal    = None
    CandlePredictor    = None
    LSTMPredictor      = None
    NewsFilter         = None
    import logging as _log
    _log.getLogger("trader_ai.bot").warning(
        f"AI modules tidak tersedia — bot jalan tanpa AI: {_ai_err}"
    )

# ── Scalping ML model (M1+M5, hasil training dari ai/ml/trainer.py) ────────
try:
    from ai.ml.predictor import ScalpingPredictor
    _scalping_pred = ScalpingPredictor()
    SCALPING_ML_AVAILABLE = True
except Exception as _e:
    _scalping_pred        = None
    SCALPING_ML_AVAILABLE = False


def fetch_data(symbol_key: str = DEFAULT_SYMBOL,
               timeframe: str  = DEFAULT_TIMEFRAME,
               period: str     = None) -> pd.DataFrame:
    ticker = SYMBOLS.get(symbol_key, symbol_key)
    period = period or DATA_PERIOD.get(timeframe, "60d")

    interval_map = {"4h": "1h"}  # Yahoo tidak punya 4h, pakai 1h
    interval = interval_map.get(timeframe, timeframe)

    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"[!] Tidak ada data untuk {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns jika ada
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        print(f"[ERROR] fetch_data: {e}")
        return pd.DataFrame()


def get_market_session() -> str:
    """Deteksi sesi berdasarkan UTC (standar pasar forex)."""
    from datetime import timezone
    hour = datetime.now(tz=timezone.utc).hour
    if 22 <= hour or hour < 7:
        return "Sydney/Tokyo"
    elif 7 <= hour < 12:
        return "London"
    elif 12 <= hour < 17:
        return "London+New York"
    elif 17 <= hour < 22:
        return "New York"
    return "Overlap"


class TradingBot:

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME,
                 use_lstm: bool = False, use_news: bool = True,
                 mt5_connector=None, use_tv: bool = False,
                 tv_user: str = None, tv_pass: str = None):
        self.symbol        = symbol
        self.timeframe     = timeframe
        self.use_lstm      = use_lstm and TF_AVAILABLE and AI_AVAILABLE
        self.use_news      = use_news
        self.mt5_conn      = mt5_connector
        self.use_tv        = use_tv
        self.tv_user       = tv_user
        self.tv_pass       = tv_pass
        self.predictor     = CandlePredictor(symbol=symbol, timeframe=timeframe) if AI_AVAILABLE else None
        self.lstm          = LSTMPredictor(timeframe=timeframe) if self.use_lstm else None
        self.news_filter   = NewsFilter(symbol=symbol) if (AI_AVAILABLE and use_news) else None
        self.df            = pd.DataFrame()
        self.df_hist       = pd.DataFrame()
        self.df_ind        = pd.DataFrame()
        self.df_ind_hist   = pd.DataFrame()
        self.df_m1         = pd.DataFrame()   # M1 candles untuk ScalpingPredictor
        self.trained       = False
        self.lstm_trained  = False
        self.train_result  = {}
        self.lstm_result   = {}
        self.news_sentiment= {}

    def load_data(self) -> bool:
        if self.mt5_conn:
            # Reconnect sekali jika koneksi putus
            if not self.mt5_conn.connected:
                print("[!] MT5 terputus — mencoba reconnect...")
                self.mt5_conn.connect()

            if self.mt5_conn.connected:
                print(f"[~] Fetching {self.symbol} data dari MT5 ({self.timeframe})...")
                raw = self.mt5_conn.get_ohlcv(self.symbol, self.timeframe,
                                              count=1500, include_forming=True)
                # Validasi candle forming: buang jika flat (High==Low) atau Volume==0
                if not raw.empty:
                    last = raw.iloc[-1]
                    if float(last["High"]) == float(last["Low"]) or float(last["Volume"]) == 0:
                        print("[!] Candle forming tidak valid (flat/zero volume) — dibuang")
                        raw = raw.iloc[:-1]

                if raw.empty:
                    print("[!] MT5 data kosong setelah validasi, fallback ke Yahoo Finance...")
                    raw = fetch_data(self.symbol, self.timeframe)
                else:
                    last_ts = raw.index[-1].strftime("%H:%M:%S")
                    print(f"[MT5] {len(raw)} candles  |  candle[-1]: "
                          f"{float(raw['Close'].iloc[-1]):.5f}  @ {last_ts} (LIVE forming)")
                    # Fetch M1 data untuk ScalpingPredictor
                    if SCALPING_ML_AVAILABLE and self.timeframe != "1m":
                        try:
                            raw_m1 = self.mt5_conn.get_ohlcv(self.symbol, "1m", count=500)
                            if not raw_m1.empty:
                                self.df_m1 = raw_m1
                        except Exception:
                            pass
            else:
                print("[!] MT5 gagal reconnect — fallback ke Yahoo Finance...")
                raw = fetch_data(self.symbol, self.timeframe)
        elif self.use_tv:
            from data.tv_feed import get_tv_ohlcv, TV_AVAILABLE
            if TV_AVAILABLE:
                raw = get_tv_ohlcv(self.symbol, self.timeframe, count=1500,
                                   username=self.tv_user, password=self.tv_pass)
                if raw.empty:
                    print("[!] TradingView data kosong, fallback ke Yahoo Finance...")
                    raw = fetch_data(self.symbol, self.timeframe)
                else:
                    print(f"[TV] close terakhir: {float(raw['Close'].iloc[-1]):.5f}")
            else:
                print("[!] tvdatafeed belum terinstall. Jalankan: pip install tvdatafeed")
                raw = fetch_data(self.symbol, self.timeframe)
        else:
            print(f"[~] Fetching {self.symbol} data dari Yahoo Finance ({self.timeframe})...")
            raw = fetch_data(self.symbol, self.timeframe)
        if raw.empty:
            return False

        new_count = save_candles(raw, self.symbol, self.timeframe)
        stats     = get_db_stats(self.symbol, self.timeframe)
        if new_count > 0:
            print(f"[DB] +{new_count} candle baru disimpan  "
                  f"| Total DB: {stats['count']} candles "
                  f"({stats['from']} s/d {stats['to']})")
        else:
            print(f"[DB] Database: {stats['count']} candles "
                  f"({stats['from']} s/d {stats['to']})  [no update]")

        merged = merge_with_db(raw, self.symbol, self.timeframe)

        self.df      = raw      # candle terbaru (untuk sinyal)
        self.df_hist = merged   # seluruh historis (untuk training)
        # max_rows=1000: analisis cukup 1000 candle terakhir (lebih cepat)
        self.df_ind  = add_all_indicators(raw, max_rows=1000)
        # Training butuh semua data
        self.df_ind_hist = add_all_indicators(merged, max_rows=0) if len(merged) > len(raw) else self.df_ind

        # ── Simpan ke PostgreSQL untuk analisa ────────────────────────
        try:
            from services.db_logger import save_candles_batch, save_candle_logs_batch
            n_c = save_candles_batch(merged, self.symbol, self.timeframe)
            n_l = save_candle_logs_batch(self.df_ind_hist, self.symbol, self.timeframe)
            if n_c > 0 or n_l > 0:
                print(f"[PG]  +{n_c} candles | +{n_l} candle_logs → PostgreSQL")
        except Exception as _pg_err:
            pass  # PostgreSQL opsional, jangan block bot

        print(f"[OK] {len(self.df_ind)} candles aktif | "
              f"{len(self.df_ind_hist)} candles untuk training ML")

        return True

    def train_model(self):
        if self.df_ind.empty:
            print("[!] Data belum di-load")
            return

        # Pakai df_ind_hist (gabungan DB) jika tersedia dan lebih besar
        train_data = getattr(self, "df_ind_hist", self.df_ind)
        if train_data is None or train_data.empty:
            train_data = self.df_ind

        print(f"[~] Training ML model ({ML_MODEL_TYPE.upper()}) "
              f"symbol={self.symbol} tf={self.timeframe} "
              f"({len(train_data)} candles)...")
        result = self.predictor.train(train_data,
                                      symbol=self.symbol,
                                      timeframe=self.timeframe)
        self.trained     = True
        self.train_result = result
        acc   = result.get('accuracy', 0)
        cacc  = result.get('conf_accuracy', 0)
        grade = "EXCELLENT" if cacc >= 80 else "GOOD" if cacc >= 65 else "FAIR" if cacc >= 55 else "LOW"
        from_cache = result.get("from_cache", False)
        label = "Model loaded (cache)" if from_cache else "Model trained"
        print(f"[OK] {label}!")
        print(f"     ----------------------------------------")
        print(f"     Symbol / Timeframe  : {result.get('trained_symbol', self.symbol)} / {result.get('trained_timeframe', self.timeframe)}")
        print(f"     Overall Accuracy    : {acc}%")
        print(f"     Confident Accuracy  : {cacc}%  [{grade}]")
        if not from_cache:
            print(f"     Precision BUY       : {result.get('precision_buy', 0)}%")
            print(f"     Recall BUY          : {result.get('recall_buy', 0)}%")
            print(f"     F1 Score            : {result.get('f1', 0)}%")
        print(f"     ----------------------------------------")
        print(f"     Features selected   : {result.get('n_features', 0)}")
        if not from_cache:
            print(f"     Sideways removed    : {result.get('n_sideways_removed', 0)} candles (noise dibuang)")
            print(f"     Train / Test        : {result.get('n_train', 0)} / {result.get('n_test', 0)}")
        print(f"     ----------------------------------------")
        if cacc >= 80:
            print(f"     [!] Akurasi tinggi - prediksi hanya pada pergerakan signifikan")
        elif cacc < 55:
            print(f"     [!] Akurasi rendah - coba timeframe lebih besar (1d) untuk akurasi lebih tinggi")

        # Train LSTM jika aktif
        if self.use_lstm and self.lstm and self.lstm.available:
            print(f"[~] Training LSTM (TensorFlow/Keras)...")
            self.lstm_result = self.lstm.train(self.df_ind)
            if "error" not in self.lstm_result:
                self.lstm_trained = True
                lc = self.lstm_result.get("conf_accuracy", 0)
                lg = "EXCELLENT" if lc >= 80 else "GOOD" if lc >= 65 else "FAIR" if lc >= 55 else "LOW"
                print(f"[OK] LSTM trained!")
                print(f"     LSTM Overall Acc  : {self.lstm_result.get('accuracy', 0)}%")
                print(f"     LSTM Confident Acc: {lc}%  [{lg}]")
                print(f"     Model Parameters  : {self.lstm_result.get('model_params', 0):,}")

    def fetch_news(self):
        if self.use_news and self.news_filter:
            self.news_sentiment = self.news_filter.get_sentiment(use_cache=True)
            risk  = self.news_sentiment.get("risk_level", "UNKNOWN")
            bias  = self.news_sentiment.get("direction_bias", {})
            b_str = bias.get("bias", "NEUTRAL")
            b_sc  = bias.get("score", 0)
            cached = "cache_date" in self.news_sentiment
            tag   = "[CACHE]" if cached else "[LIVE]"
            print(f"[OK] News {tag} - Risk: {risk} | "
                  f"Bias Berita: {b_str} ({b_sc:+.1f}) | "
                  f"{self.news_sentiment.get('total_news', 0)} articles")

    def analyze(self, candle_memory: dict = None) -> dict:
        if self.df_ind.empty:
            return {}

        # ── Jika AI tidak tersedia, return WAIT ───────────────────────
        if not AI_AVAILABLE:
            return {
                "direction": "WAIT",
                "score": 0,
                "confidence": 0,
                "reasons": [(0, "AI modules tidak aktif — BOT_AI_ENABLED=false")],
                "filters": {"ai": "DISABLED"},
                "close": float(self.df_ind.iloc[-1]["Close"]),
                "atr": 0,
            }

        # ── Realtime data dari MT5 ────────────────────────────────────
        rt_data = {}
        if self.mt5_conn and self.mt5_conn.connected:
            rt_data = self.mt5_conn.get_realtime_data(self.symbol, tick_seconds=60)

        news_bias   = self.news_sentiment.get("direction_bias") if self.news_sentiment else None
        news_risk   = self.news_sentiment.get("risk_level", "LOW") if self.news_sentiment else "LOW"
        news_bias_  = (news_bias or {}).get("bias", "NEUTRAL")
        news_score_ = (news_bias or {}).get("score", 0)

        # ── Rule-based signal DINONAKTIFKAN — pakai ScalpML saja ─────
        # sig = generate_signal(self.df_ind, news_bias=news_bias,
        #                       news_risk=news_risk, candle_memory=candle_memory)
        sig = {"direction": "WAIT", "score": 0, "confidence": 0,
               "reasons": [], "filters": {"mode": "ScalpML-Only"},
               "signal_strength": "", "market_state": ""}

        ml_pred   = {}
        lstm_pred = {}

        row   = self.df_ind.iloc[-1]
        close = float(row["Close"])

        # ── ScalpingPredictor — satu-satunya sumber sinyal ────────────
        scalping_pred = {}
        if SCALPING_ML_AVAILABLE and _scalping_pred is not None:
            try:
                m5_raw = self.df.copy() if not self.df.empty else self.df_ind[["Open","High","Low","Close","Volume"]].copy()
                m5_raw.columns = [c.lower() for c in m5_raw.columns]
                if 'volume' not in m5_raw.columns:
                    m5_raw['volume'] = 0
                scalping_pred = _scalping_pred.predict(m5_raw)
            except Exception as _e:
                scalping_pred = {}
                print(f"[ScalpML] predict error: {_e}")

        consensus = "NO SIGNAL"

        from ai.signals import calculate_smart_tp_sl

        sig_dir   = "WAIT"
        rule_conf = 0

        # ─── Session Bias dari analisis jam tutup sesi ────────────────────
        _sess_bias_dir   = "NEUTRAL"
        _sess_bias_score = 0.0
        _sess_bias_str   = ""
        try:
            from data.session_bias import get_current_bias
            _sb = get_current_bias()
            if _sb:
                _sess_bias_dir   = _sb.get("direction", "NEUTRAL")
                _sess_bias_score = float(_sb.get("score", 0))
                _sess_bias_str   = _sb.get("strength", "")
        except Exception:
            pass

        # ─── Pre-session plan (dari deep analysis saat market tutup) ──────
        _plan_bias = "NEUTRAL"
        try:
            from data.session_bias import get_session_plan
            _plan = get_session_plan()
            if _plan:
                _plan_bias = _plan.get("htf_bias", "NEUTRAL")
                # Terapkan min_score dari adaptive jika plan lebih ketat
                import config as _cfg
                _plan_score = float(_plan.get("min_score", _cfg.MIN_SIGNAL_SCORE))
                if _plan_score > _cfg.MIN_SIGNAL_SCORE:
                    _cfg.MIN_SIGNAL_SCORE = _plan_score
        except Exception:
            pass

        # Realtime confirmation
        rt_bias  = rt_data.get("realtime_bias", "NEUTRAL")
        rt_score = rt_data.get("realtime_score", 0)
        rt_note  = f" +RT:{rt_bias}" if rt_bias != "NEUTRAL" else ""

        # ─── GUARD: tunggu posisi tutup dulu (SL/TP) ────────────────────
        _open_positions = []
        if self.mt5_conn and self.mt5_conn.connected:
            try:
                # get_all_positions: cek semua arah (BUY dan SELL)
                _open_positions = self.mt5_conn.mt5.get_all_positions(self.symbol)
            except Exception:
                _open_positions = []
        _n_open = len(_open_positions)

        sc_dir  = scalping_pred.get("direction", "WAIT")
        sc_prob = scalping_pred.get("probability", 0.0)
        sc_conf = scalping_pred.get("confidence", "LOW")

        # ── ML-ONLY MODE — rule-based dinonaktifkan sementara ────────────
        # Indikator sudah di-encode ke dalam model, pakai ScalpML langsung.
        # Rule+indicator hanya jadi info log, tidak blok order.
        # Aktifkan kembali: hapus blok ini dan uncomment Rule logic di bawah.
        # ─────────────────────────────────────────────────────────────────
        _sc_thr = _scalping_pred.prob_threshold if _scalping_pred else 0.52

        if _n_open >= MAX_OPEN_POSITIONS:
            exec_direction = "WAIT"
            exec_source    = f"GUARD-MaxPos ({_n_open} open)"
            priority_num   = 0

        elif sc_dir in ("BUY", "SELL") and sc_prob >= _sc_thr:
            from ai.signals import calculate_smart_tp_sl
            exec_direction = sc_dir
            exec_source    = f"ScalpML({sc_conf},{sc_prob:.3f})"
            priority_num   = 1
            tp_sl = calculate_smart_tp_sl(
                exec_direction, close,
                float(row.get("atr", close * 0.001)),
                self.df_ind, 5.0
            )
            sig = dict(sig)
            sig["direction"] = exec_direction
            sig["sl"]        = tp_sl["sl"]
            sig["tp"]        = tp_sl["tp"]
            sig["tp_dist"]   = tp_sl["tp_dist"]
            sig["sl_dist"]   = tp_sl["sl_dist"]
            sig["rr_ratio"]  = tp_sl["rr"]

        else:
            exec_direction = "WAIT"
            exec_source    = f"ScalpML-WAIT(prob={sc_prob:.3f}<{_sc_thr})"
            priority_num   = 5

        # ── Rule-based logic (dinonaktifkan — indikator sudah di model) ──
        # if sig_dir in ("BUY", "SELL") and _n_open >= MAX_OPEN_POSITIONS:
        #     exec_direction = "WAIT"
        #     exec_source    = f"GUARD-MaxPos ({_n_open} open)"
        #     priority_num   = 0
        # elif sig_dir in ("BUY", "SELL"):
        #     exec_direction = sig_dir
        #     if sc_dir == sig_dir:
        #         exec_source = f"Rule+ScalpML({sc_conf},{sc_prob:.2f}) conf={rule_conf}"
        #     elif sc_dir in ("BUY","SELL") and sc_dir != sig_dir and sc_conf=="HIGH" and sc_prob>=0.75:
        #         exec_direction = "WAIT"
        #         exec_source    = f"ScalpML-CANCEL({sc_dir} p={sc_prob:.2f} vs Rule:{sig_dir})"
        #     else:
        #         exec_source = f"Rule conf={rule_conf}"
        #     priority_num = 3
        # elif sc_dir in ("BUY","SELL") and sc_prob >= 0.65 and sc_conf in ("HIGH","MEDIUM"):
        #     exec_direction = sc_dir
        #     exec_source    = f"ScalpML-Only({sc_conf},{sc_prob:.2f})"
        #     priority_num   = 4
        #     ...
        # else:
        #     exec_direction = "WAIT"
        #     exec_source    = "NoSignal"
        #     priority_num   = 5

        # Risk note
        risk_note = ""
        mstate = sig.get("market_state", "")
        if mstate in ("RANGE", "UNCLEAR"):
            risk_note = f" Market {mstate}"
        if news_score_ and abs(news_score_) >= 2:
            risk_note += f" | News:{news_bias_}({news_score_:+.1f})"

        # Final advice
        if exec_direction in ("BUY", "SELL"):
            final_advice = f"{exec_direction} — {exec_source}{rt_note}{risk_note}"
        else:
            final_advice = f"WAIT{risk_note}"

        result = {
            "symbol":       self.symbol,
            "timeframe":    self.timeframe,
            "timestamp":    datetime.now(tz=WIB).strftime("%Y-%m-%d %H:%M:%S WIB"),
            "session":      get_market_session(),
            "close":        close,
            "open":         float(row["Open"]),
            "high":         float(row["High"]),
            "low":          float(row["Low"]),
            "volume":       float(row.get("Volume", 0)),
            "candle_name":  str(row.get("candle_name", "None")),
            # Indikator
            "rsi":          round(float(row.get("rsi", 50)), 2),
            "macd":         round(float(row.get("macd", 0)), 5),
            "histogram":    round(float(row.get("histogram", 0)), 5),
            "adx":          round(float(row.get("adx", 0)), 2),
            "atr":          round(float(row.get("atr", 0)), 5),
            f"ema{EMA_FAST}":   round(float(row.get(f"ema_{EMA_FAST}", close)), 5),
            f"ema{EMA_SLOW}":   round(float(row.get(f"ema_{EMA_SLOW}", close)), 5),
            f"ema{EMA_TREND}":  round(float(row.get(f"ema_{EMA_TREND}", close)), 5),
            "bb_upper":     round(float(row.get("bb_upper", close)), 5),
            "bb_lower":     round(float(row.get("bb_lower", close)), 5),
            "stoch_k":      round(float(row.get("stoch_k", 50)), 2),
            "stoch_d":      round(float(row.get("stoch_d", 50)), 2),
            # Sinyal
            "signal":          sig,
            "ml_pred":         ml_pred,
            "lstm_pred":       lstm_pred,
            "scalping_pred":   scalping_pred,
            "consensus":       consensus,
            "consensus_dir":   consensus,
            "exec_direction":  exec_direction,
            "exec_source":     exec_source,
            "priority_num":    priority_num,
            "final_advice":    final_advice,
            "news_risk":       news_risk,
            "news_bias":       news_bias or {},
            "news_sentiment":  self.news_sentiment,
            "realtime_data":   rt_data,
            # Sumber data — untuk audit di DB
            "data_source":     "MT5" if (self.mt5_conn and self.mt5_conn.connected)
                               else ("TV" if self.use_tv else "Yahoo"),
        }
        return result

    def print_analysis(self, result: dict) -> None:
        if not result:
            print("[!] Tidak ada hasil analisis")
            return

        sig    = result["signal"]
        ml     = result.get("ml_pred", {})
        direct = sig["direction"]

        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        CYAN   = "\033[96m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        DIM    = "\033[2m"

        color    = GREEN if direct == "BUY" else RED if direct == "SELL" else YELLOW
        sep      = "=" * 60
        exec_dir = result.get("exec_direction", "WAIT")
        exec_src = result.get("exec_source", "")
        ex_color = GREEN if exec_dir == "BUY" else RED if exec_dir == "SELL" else YELLOW

        # ── Indicators ringkas ────────────────────────────────────────
        rsi_c  = GREEN if result["rsi"] < RSI_OVERSOLD else RED if result["rsi"] > RSI_OVERBOUGHT else RESET
        adx_c  = GREEN if result["adx"] > ADX_TREND_MIN else YELLOW
        ema20  = result.get(f"ema{EMA_SLOW}", result["close"])
        ema50  = result.get(f"ema{EMA_TREND}", result["close"])
        ema_dir = f"{GREEN}↑BULL{RESET}" if ema20 > ema50 else f"{RED}↓BEAR{RESET}"

        # ── ML ringkas ───────────────────────────────────────────────
        ml_str = ""
        if ml and ml.get("direction") != "UNKNOWN":
            mc = GREEN if ml["direction"] == "BUY" else RED if ml["direction"] == "SELL" else YELLOW
            ml_str = f" | ML:{mc}{ml['direction']}({ml['confidence']}%){RESET}"

        # ── Scalping ML ringkas ──────────────────────────────────────
        sp = result.get("scalping_pred", {})
        sc_str = ""
        if sp and sp.get("direction") != "WAIT":
            sc = GREEN if sp["direction"] == "BUY" else RED if sp["direction"] == "SELL" else YELLOW
            sc_str = f" | ScalpML:{sc}{sp['direction']}({sp.get('probability',0):.2f}/{sp.get('confidence','?')}){RESET}"

        # ── Signal strength ──────────────────────────────────────────
        ss     = sig.get("signal_strength", "")
        spts   = sig.get("strength_pts", 0)
        ss_c   = GREEN if ss == "STRONG" else YELLOW if ss == "MEDIUM" else RED
        ss_str = f"{ss_c}{ss}({spts}/8){RESET}"

        # ── Session + News ───────────────────────────────────────────
        nb       = result.get("news_bias", {})
        nb_bias  = nb.get("bias", "NEUTRAL")
        nb_score = nb.get("score", 0)
        nb_c     = GREEN if nb_bias == "BULLISH" else RED if nb_bias == "BEARISH" else YELLOW

        session_str = ""
        try:
            from data.session_bias import get_current_bias
            _sb = get_current_bias()
            if _sb and _sb.get("direction") in ("BUY", "SELL", "NEUTRAL"):
                _sc = GREEN if _sb["direction"] == "BUY" else (RED if _sb["direction"] == "SELL" else YELLOW)
                session_str = f"  Session:{_sc}{_sb['direction']}({_sb.get('score',0):+.1f}){RESET}"
        except Exception:
            pass

        # ── Hist accuracy ─────────────────────────────────────────────
        sa     = result.get("sig_accuracy", {})
        sa_str = ""
        if sa and sa.get("total", 0) >= 3:
            sa_c   = GREEN if sa["accuracy"] >= 60 else YELLOW if sa["accuracy"] >= 50 else RED
            sa_str = (f"  HistAcc:{sa_c}{sa['accuracy']}%{RESET}"
                      f"({sa['win']}W/{sa['loss']}L)")

        print(f"\n{BOLD}{sep}{RESET}")
        print(f"  {BOLD}{CYAN}{result['symbol']} {result['timeframe']}{RESET}"
              f"  {result['timestamp']}  {DIM}|{RESET}  Session:{result['session']}")
        print(f"  Close:{result['close']:.2f}  "
              f"RSI:{rsi_c}{result['rsi']}{RESET}  "
              f"ADX:{adx_c}{result['adx']}{RESET}  "
              f"ATR:{result['atr']:.2f}  "
              f"EMA:{ema_dir}  "
              f"MACD:{GREEN if result['histogram']>0 else RED}{result['histogram']:+.3f}{RESET}")

        regime   = sig.get("regime", "?")
        regime_c = GREEN if regime == "TREND" else RED if regime == "VOLATILE" else YELLOW
        conf_val = sig.get("confidence", 0)
        conf_c   = GREEN if conf_val >= 5 else YELLOW if conf_val >= 4 else RED
        print(f"  Score:{sig['score']:+.2f}  {regime_c}{regime}{RESET}  "
              f"Dir:{color}{BOLD}{direct}{RESET}"
              + ml_str
              + sc_str
              + f"  Strength:{ss_str}  Conf:{conf_c}{conf_val}/7{RESET}")

        if sig.get("sl"):
            rr = sig.get("rr_ratio", 0)
            rr_c = GREEN if rr >= 2 else YELLOW if rr >= 1 else RED
            print(f"  SL:{RED}{sig['sl']:.2f}{RESET}  "
                  f"TP:{GREEN}{sig['tp']:.2f}{RESET}  "
                  f"R:R {rr_c}1:{rr}{RESET}")

        if sp and sp.get("direction") != "WAIT":
            sc     = GREEN if sp["direction"] == "BUY" else RED
            sc_rr  = sp.get("rr", 0)
            sc_rr_c= GREEN if sc_rr >= 1.5 else YELLOW
            print(f"  {BOLD}ScalpML{RESET}: {sc}{sp['direction']}{RESET}"
                  f"  prob={sp.get('probability',0):.2f}"
                  f"  conf={sp.get('confidence','?')}"
                  f"  SL:{RED}{sp.get('sl',0):.2f}{RESET}"
                  f"  TP:{GREEN}{sp.get('tp',0):.2f}{RESET}"
                  f"  RR:{sc_rr_c}1:{sc_rr}{RESET}")

        print(f"  News:{nb_c}{nb_bias}({nb_score:+.1f}){RESET}"
              + session_str + sa_str)

        src_color = (GREEN if "★" in exec_src else
                     RED if any(x in exec_src for x in ("CANCEL","VETO","GUARD")) else CYAN)
        print(f"  {BOLD}EKSEKUSI:{ex_color}{exec_dir}{RESET}  {src_color}{exec_src}{RESET}")

        _risk = result.get("final_advice", "")
        if "⚠" in _risk or "News:" in _risk:
            print(f"  {YELLOW}Risk: {_risk}{RESET}")

        print(sep)
        print()
