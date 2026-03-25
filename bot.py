"""
bot.py — Trading Bot: data fetching, indicator pipeline, ML training, analisis sinyal.

Kelas utama:
  TradingBot   : inti bot — load data, train ML, generate signal, analyze, print output

Alur kerja (per siklus):
  1. load_data()     → ambil OHLCV dari MT5 / TradingView / Yahoo Finance
  2. train_model()   → latih CandlePredictor (ensemble ML) + LSTM (opsional)
  3. analyze()       → jalankan indikator → generate_signal → gabungkan ML vote
                       → terapkan session bias (GUARD-HTF-Bias)
                       → tentukan final direction + TP/SL/RR
  4. print_result()  → tampilkan hasil analisis di terminal dengan warna

Sumber data (prioritas):
  1. MT5 live (realtime, candle forming)
  2. TradingView via tvdatafeed (--tv)
  3. Yahoo Finance (fallback, delay ~1 menit)

Session Bias (GUARD-HTF-Bias):
  Analisis HTF (H4+Daily) dijalankan saat session close (Tokyo/London/Daily).
  Jika bias score >= 3.0 dan sinyal M5 berlawanan → WAIT (blok masuk).
  Bias state disimpan di data/session_bias_state.json.

Dependencies:
  yfinance, pandas, numpy, MetaTrader5 (opsional),
  ml.model.CandlePredictor, ml.deep_model.LSTMPredictor,
  data.news_filter.NewsFilter, data.candle_db
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))
import time
import sys

from config import *
from analysis.indicators import add_all_indicators
from analysis.signals import generate_signal
from ml.model import CandlePredictor
from ml.deep_model import LSTMPredictor, TF_AVAILABLE
from data.news_filter import NewsFilter
from data.candle_db import save_candles, merge_with_db, get_db_stats


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
    """
    Inti trading bot — mengelola data, ML, dan analisis sinyal.

    Params:
      symbol        : simbol trading (misal "XAUUSD", "EURUSD")
      timeframe     : timeframe candle ("1m","5m","15m","1h","4h","1d")
      use_lstm      : aktifkan LSTM deep model (butuh TensorFlow)
      use_news      : aktifkan news filter (sentiment + economic calendar)
      mt5_connector : instance MT5Connector yang sudah connected
      use_tv        : gunakan TradingView sebagai sumber data
      tv_user/pass  : kredensial TradingView (opsional, untuk akun premium)

    State penting:
      self.df       : raw OHLCV terbaru
      self.df_ind   : OHLCV + semua indikator (99 kolom)
      self.trained  : True setelah train_model() berhasil
      self.train_result : dict hasil training ML (accuracy, precision, dll)
    """

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME,
                 use_lstm: bool = False, use_news: bool = True,
                 mt5_connector=None, use_tv: bool = False,
                 tv_user: str = None, tv_pass: str = None):
        self.symbol        = symbol
        self.timeframe     = timeframe
        self.use_lstm      = use_lstm and TF_AVAILABLE
        self.use_news      = use_news
        self.mt5_conn      = mt5_connector
        self.use_tv        = use_tv
        self.tv_user       = tv_user
        self.tv_pass       = tv_pass
        self.predictor     = CandlePredictor(symbol=symbol, timeframe=timeframe)
        self.lstm          = LSTMPredictor(timeframe=timeframe) if self.use_lstm else None
        self.news_filter   = NewsFilter(symbol=symbol) if use_news else None
        self.df            = pd.DataFrame()
        self.df_hist       = pd.DataFrame()
        self.df_ind        = pd.DataFrame()
        self.df_ind_hist   = pd.DataFrame()
        self.trained       = False
        self.lstm_trained  = False
        self.train_result  = {}
        self.lstm_result   = {}
        self.news_sentiment= {}

    def load_data(self) -> bool:
        if self.mt5_conn and self.mt5_conn.connected:
            print(f"[~] Fetching {self.symbol} data dari MT5 ({self.timeframe})...")
            # include_forming=True → candle yang sedang terbentuk ikut dibaca (realtime)
            raw = self.mt5_conn.get_ohlcv(self.symbol, self.timeframe,
                                          count=1500, include_forming=True)
            if raw.empty:
                print("[!] MT5 data kosong, fallback ke Yahoo Finance...")
                raw = fetch_data(self.symbol, self.timeframe)
            else:
                last_ts  = raw.index[-1].strftime("%H:%M:%S")
                is_live  = "(LIVE forming)" if self.mt5_conn.connected else ""
                print(f"[MT5] {len(raw)} candles  |  candle[-1]: "
                      f"{float(raw['Close'].iloc[-1]):.5f}  @ {last_ts} {is_live}")
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
        # Training butuh semua data → max_rows=0 (tanpa batas)
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
        acc   = result['accuracy']
        cacc  = result['conf_accuracy']
        grade = "EXCELLENT" if cacc >= 80 else "GOOD" if cacc >= 65 else "FAIR" if cacc >= 55 else "LOW"
        print(f"[OK] Model trained!")
        print(f"     ----------------------------------------")
        print(f"     Symbol / Timeframe  : {result['trained_symbol']} / {result['trained_timeframe']}")
        print(f"     Overall Accuracy    : {acc}%")
        print(f"     Confident Accuracy  : {cacc}%  [{grade}]")
        print(f"     Precision BUY       : {result['precision_buy']}%")
        print(f"     Recall BUY          : {result['recall_buy']}%")
        print(f"     F1 Score            : {result['f1']}%")
        print(f"     ----------------------------------------")
        print(f"     Features selected   : {result['n_features']}")
        print(f"     Sideways removed    : {result['n_sideways_removed']} candles (noise dibuang)")
        print(f"     Train / Test        : {result['n_train']} / {result['n_test']}")
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

        # ── Realtime data dari MT5 (tick + orderbook) ─────────────────
        rt_data = {}
        if self.mt5_conn and self.mt5_conn.connected:
            rt_data = self.mt5_conn.get_realtime_data(self.symbol, tick_seconds=60)

        news_bias = self.news_sentiment.get("direction_bias") if self.news_sentiment else None
        news_risk = self.news_sentiment.get("risk_level", "LOW") if self.news_sentiment else "LOW"
        sig = generate_signal(self.df_ind, news_bias=news_bias,
                              news_risk=news_risk, candle_memory=candle_memory)

        # ML prediction (Random Forest / Gradient Boosting)
        ml_pred = {}
        if ML_ENABLED and self.trained:
            ml_pred = self.predictor.predict(self.df_ind, predict_symbol=self.symbol)

        # LSTM prediction (Deep Learning)
        lstm_pred = {}
        if self.use_lstm and self.lstm_trained:
            lstm_pred = self.lstm.predict(self.df_ind)

        # Gabungkan
        row   = self.df_ind.iloc[-1]
        close = float(row["Close"])

        votes = []
        if sig["direction"] in ("BUY", "SELL"):
            votes.append(sig["direction"])
        if ml_pred and ml_pred.get("direction") in ("BUY", "SELL"):
            votes.append(ml_pred["direction"])
        if lstm_pred and lstm_pred.get("direction") in ("BUY", "SELL"):
            votes.append(lstm_pred["direction"])

        buy_votes  = votes.count("BUY")
        sell_votes = votes.count("SELL")
        if not votes:
            consensus_dir = "WAIT"
            consensus     = "NO SIGNAL"
        elif buy_votes > sell_votes:
            consensus_dir = "BUY"
            consensus     = f"BUY  ({buy_votes}/{len(votes)} votes)"
        elif sell_votes > buy_votes:
            consensus_dir = "SELL"
            consensus     = f"SELL ({sell_votes}/{len(votes)} votes)"
        else:
            consensus_dir = "WAIT"
            consensus     = "SPLIT (tie)"

        # ═══════════════════════════════════════════════════════════════════
        # DECISION ENGINE — Clean voting system, discipline-first
        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY CHAIN:
        # GUARD  : Position already open → WAIT (no overtrading)
        # #1 ★★★ : Rule + ML both confirm (ML prob ≥ 65%) → EXECUTE
        # #2 VETO: ML disagrees at ≥ 70% → CANCEL rule signal
        # #3 ★★  : Rule signal, ML uncertain/unavailable → EXECUTE
        # #4 ★   : ML alone (conf_acc ≥ threshold, structure ok) → EXECUTE
        # #5     : No signal → WAIT
        # ═══════════════════════════════════════════════════════════════════
        from analysis.signals import calculate_smart_tp_sl

        ml_conf    = ml_pred.get("confidence", 0)    if ml_pred else 0
        ml_dir     = ml_pred.get("direction", "WAIT") if ml_pred else "WAIT"
        ml_prob_b  = ml_pred.get("proba_buy",  0)    if ml_pred else 0   # 0-100
        ml_prob_s  = ml_pred.get("proba_sell", 0)    if ml_pred else 0
        ml_certain = (bool(ml_pred) and not ml_pred.get("uncertain")
                      and ml_dir in ("BUY", "SELL"))

        _ml_cacc    = self.train_result.get("conf_accuracy", 0) if self.trained else 0
        ml_reliable = _ml_cacc >= ML_MIN_CONFIDENT_ACC

        sig_dir    = sig["direction"]
        rule_conf  = sig.get("confidence", 0)        # 0-6 quality score from signals.py
        news_bias_  = (news_bias or {}).get("bias", "NEUTRAL")
        news_score_ = (news_bias or {}).get("score", 0)

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

        # ─── GUARD: No new position if one already open ─────────────────
        _open_positions = []
        if self.mt5_conn and self.mt5_conn.connected:
            try:
                _open_positions = self.mt5_conn.mt5.get_positions(self.symbol)
            except Exception:
                _open_positions = []
        _n_open = len(_open_positions)

        if sig_dir in ("BUY", "SELL") and _n_open >= MAX_OPEN_POSITIONS:
            exec_direction = "WAIT"
            exec_source    = f"GUARD-MaxPos ({_n_open} open ≥ {MAX_OPEN_POSITIONS})"
            priority_num   = 0

        # ─── GUARD: HTF Session Bias Counter-Trend Block ────────────────
        # Jika HTF bias kuat berlawanan (score >= 3.0), blok sinyal counter-trend
        elif sig_dir in ("BUY", "SELL") and _sess_bias_dir in ("BUY", "SELL") \
                and sig_dir != _sess_bias_dir and abs(_sess_bias_score) >= 3.0:
            exec_direction = "WAIT"
            exec_source    = (f"GUARD-HTF-Bias Session:{_sess_bias_dir}"
                              f"({_sess_bias_score:+.1f}) vs Signal:{sig_dir} "
                              f"— counter-trend HTF")
            priority_num   = 0

        # ─── #1 RULE + ML CONFIRM ★★★ ────────────────────────────────
        elif sig_dir in ("BUY", "SELL") and ml_certain and ml_dir == sig_dir:
            ml_prob_aligned = ml_prob_b if sig_dir == "BUY" else ml_prob_s
            if ml_prob_aligned >= ML_VOTE_THRESHOLD:
                conf_total     = min(rule_conf + 1, 7)   # ML adds +1
                exec_direction = sig_dir
                exec_source    = f"#1-Rule+ML({ml_conf:.0f}%,prob{ml_prob_aligned:.0f}%) ★★★ conf={conf_total}"
                priority_num   = 1
            else:
                # ML same direction but low probability — treat as rule-only
                exec_direction = sig_dir
                exec_source    = f"#3-Rule+ML-weak({ml_conf:.0f}%) ★★ conf={rule_conf}"
                priority_num   = 3

        # ─── #2 ML DISAGREES → CANCEL (only if ML reliable) ────────────
        elif sig_dir in ("BUY", "SELL") and ml_certain and ml_dir != sig_dir \
                and ml_conf >= 70 and ml_reliable:
            ml_prob_oppose = ml_prob_s if sig_dir == "BUY" else ml_prob_b
            exec_direction = "WAIT"
            exec_source    = f"#2-ML-CANCEL Rule:{sig_dir} ← ML:{ml_dir}({ml_conf:.0f}%,opp:{ml_prob_oppose:.0f}%)"
            priority_num   = 2

        # ─── #3 RULE-ONLY ★★ ─────────────────────────────────────────
        elif sig_dir in ("BUY", "SELL"):
            ml_status = ""
            if ml_certain and ml_dir == sig_dir:
                ml_status = f" [ML:{ml_conf:.0f}%-weak]"
            elif not ml_certain:
                ml_status = " [ML:uncertain]"
            exec_direction = sig_dir
            exec_source    = f"#3-Rule-Only{ml_status} ★★ conf={rule_conf}"
            priority_num   = 3

        # ─── #4 ML-ONLY ★ (Rule=WAIT, ML reliable, no structure conflict) ──
        elif ml_certain and ml_conf >= 70 and ml_reliable:
            rule_score   = sig.get("score", 0)
            struct_conflict = ((ml_dir == "SELL" and rule_score > +4.0) or
                               (ml_dir == "BUY"  and rule_score < -4.0))
            if struct_conflict:
                exec_direction = "WAIT"
                exec_source    = f"#4-ML-SKIP struct conflict score={rule_score:+.1f} vs ML:{ml_dir}"
                priority_num   = 4
            else:
                exec_direction = ml_dir
                exec_source    = f"#4-ML-Only({ml_conf:.0f}%) ★"
                priority_num   = 4
                tp_sl = calculate_smart_tp_sl(
                    exec_direction, close,
                    float(row.get("atr", close * 0.001)),
                    self.df_ind, sig["score"]
                )
                sig = dict(sig)
                sig["direction"] = exec_direction
                sig["sl"]        = tp_sl["sl"]
                sig["tp"]        = tp_sl["tp"]
                sig["tp_dist"]   = tp_sl["tp_dist"]
                sig["sl_dist"]   = tp_sl["sl_dist"]
                sig["rr_ratio"]  = tp_sl["rr"]
                sig["method_tp"] = tp_sl["method_tp"]
                sig["method_sl"] = tp_sl["method_sl"]

        elif ml_certain and ml_conf >= 70 and not ml_reliable:
            exec_direction = "WAIT"
            exec_source    = f"#4-ML-SKIP acc={_ml_cacc:.1f}%<{ML_MIN_CONFIDENT_ACC}%"
            priority_num   = 4

        # ─── #5 NO-SIGNAL ─────────────────────────────────────────────
        else:
            exec_direction = "WAIT"
            exec_source    = "#5-NoSignal"
            priority_num   = 5

        # Risk note
        risk_note = ""
        mstate = sig.get("market_state", "")
        if mstate in ("RANGE", "UNCLEAR"):
            risk_note = f" ⚠ Market {mstate}"
        if news_score_ and abs(news_score_) >= 2:
            risk_note += f" | News:{news_bias_}({news_score_:+.1f})"

        # Final advice
        if "#2-ML-CANCEL" in exec_source:
            final_advice = f"WAIT — ML cancels Rule {sig_dir} (ML:{ml_dir} {ml_conf:.0f}%)"
        elif exec_direction in ("BUY", "SELL"):
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
            "consensus":       consensus,
            "consensus_dir":   consensus_dir,
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

        color = GREEN if direct == "BUY" else RED if direct == "SELL" else YELLOW
        sep   = "=" * 60

        print(f"\n{BOLD}{sep}{RESET}")
        print(f"  {BOLD}{CYAN}TRADING ANALYSIS - {result['symbol']} | {result['timeframe']}{RESET}")
        print(f"  {result['timestamp']}  |  Session: {result['session']}")
        print(sep)

        print(f"  Close : {result['close']:.5f}  |  Open: {result['open']:.5f}")
        print(f"  High  : {result['high']:.5f}  |  Low : {result['low']:.5f}")
        cname = result.get("candle_name", "None")
        if cname and cname != "None":
            is_bull = "↑" in cname
            is_bear = "↓" in cname
            cc = GREEN if is_bull else RED if is_bear else YELLOW
            print(f"  Candle: {BOLD}{cc}{cname}{RESET}")
        print(sep)

        print(f"  {BOLD}--- INDICATORS -------------------------------------------{RESET}")
        rsi_color = GREEN if result["rsi"] < RSI_OVERSOLD else RED if result["rsi"] > RSI_OVERBOUGHT else RESET
        print(f"  RSI({RSI_PERIOD})       : {rsi_color}{result['rsi']}{RESET}")
        macd_color = GREEN if result["histogram"] > 0 else RED
        print(f"  MACD        : {macd_color}{result['macd']:+.5f}  Hist: {result['histogram']:+.5f}{RESET}")
        adx_color = GREEN if result["adx"] > ADX_TREND_MIN else YELLOW
        print(f"  ADX({ADX_PERIOD})       : {adx_color}{result['adx']}{RESET}")
        print(f"  ATR({ATR_PERIOD})       : {result['atr']:.5f}")
        sk_color = GREEN if result["stoch_k"] < STOCH_OVERSOLD else RED if result["stoch_k"] > STOCH_OVERBOUGHT else RESET
        print(f"  Stoch K/D   : {sk_color}{result['stoch_k']} / {result['stoch_d']}{RESET}")
        print(f"  EMA {EMA_FAST}/{EMA_SLOW}/{EMA_TREND} : {result[f'ema{EMA_FAST}']:.5f} / {result[f'ema{EMA_SLOW}']:.5f} / {result[f'ema{EMA_TREND}']:.5f}")
        print(f"  BB Upper/Low: {result['bb_upper']:.5f} / {result['bb_lower']:.5f}")
        print(sep)

        print(f"  {BOLD}--- RULE-BASED SIGNAL -----------------------------------{RESET}")
        tech_sc = sig.get("score_technical",  sig["score"])
        vol_sc  = sig.get("score_volume",    0)
        smc_sc  = sig.get("score_smc",       0)
        str_sc  = sig.get("score_structure", 0)
        news_sc = sig.get("score_news",      0)
        regime  = sig.get("regime",          "?")
        regime_c = GREEN if regime == "TREND" else RED if regime == "VOLATILE" else YELLOW
        print(f"  Score       : {sig['score']:+.2f}  "
              f"| Tech:{tech_sc:+.2f}  Vol:{vol_sc:+.2f}  SMC:{smc_sc:+.2f}  Str:{str_sc:+.2f}  News:{news_sc:+.2f}")
        print(f"  Regime      : {regime_c}{BOLD}{regime}{RESET}")
        print(f"  Direction   : {BOLD}{color}{direct}{RESET}")
        if sig["sl"]:
            m_tp   = sig.get("method_tp", "ATR")
            m_sl   = sig.get("method_sl", "Candle")
            tp_dist= sig.get("tp_dist", 0)
            sl_dist= sig.get("sl_dist", 0)
            rr     = sig.get("rr_ratio", 0)
            print(f"  Stop Loss   : {RED}{sig['sl']:.5f}{RESET}  "
                  f"(dist: {sl_dist:.5f})  [{m_sl}]")
            print(f"  Take Profit : {GREEN}{sig['tp']:.5f}{RESET}  "
                  f"(dist: {tp_dist:.5f})  [{m_tp}]")
            rr_color = GREEN if rr >= 2 else YELLOW if rr >= 1 else RED
            print(f"  R:R Ratio   : {BOLD}{rr_color}1:{rr}{RESET}  "
                  f"(SL kecil, TP dari analisis)")

        print(f"\n  Reasons:")
        for score, reason in sig.get("reasons", []):
            r_color = GREEN if score > 0 else RED if score < 0 else YELLOW
            arrow = "^" if score > 0 else "v" if score < 0 else "-"
            is_news = "News Bias" in reason
            prefix  = "N" if is_news else " "
            print(f"    {r_color}{arrow}{prefix} {reason}{RESET}")

        nb = result.get("news_bias", {})
        if nb:
            b     = nb.get("bias", "NEUTRAL")
            bs    = nb.get("score", 0)
            bc    = nb.get("confidence", "LOW")
            bc_   = GREEN if b == "BULLISH" else RED if b == "BEARISH" else YELLOW
            print(f"\n  {BOLD}[BERITA → PREDIKSI ARAH]{RESET}")
            print(f"  Bias Berita : {BOLD}{bc_}{b}{RESET}  ({bs:+.1f})  Konfiden: {bc}")
            for r in nb.get("reasons", [])[:4]:
                rc = GREEN if "BULLISH" in r else RED if "BEARISH" in r else YELLOW
                print(f"    {rc}• {r[:90]}{RESET}")
        print(sep)

        # ── REALTIME DATA (Tick + OrderBook) ──────────────────────────
        rt = result.get("realtime_data", {})
        if rt:
            rt_bias_  = rt.get("realtime_bias", "NEUTRAL")
            rt_score_ = rt.get("realtime_score", 0)
            rt_c      = GREEN if rt_bias_ == "BUY" else RED if rt_bias_ == "SELL" else YELLOW
            tick_m    = rt.get("tick_momentum", {})
            ob        = rt.get("orderbook", {})
            spread_d  = rt.get("spread", {})
            ctick     = rt.get("current_tick", {})

            print(f"  {BOLD}--- REALTIME (Tick + OrderBook) -------------------------{RESET}")

            # Current price
            if ctick:
                print(f"  Bid/Ask     : {ctick.get('bid',0):.5f} / {ctick.get('ask',0):.5f}"
                      f"  Spread: {ctick.get('spread',0):.5f}"
                      + (f"  {YELLOW}[WIDE!]{RESET}" if spread_d.get("wide") else ""))

            # Tick momentum
            if tick_m.get("tick_count", 0) > 0:
                tm_dir = tick_m.get("direction", "NEUTRAL")
                tm_c   = GREEN if tm_dir == "BUY" else RED if tm_dir == "SELL" else YELLOW
                br     = tick_m.get("bull_ratio", 0.5)
                chg    = tick_m.get("price_change", 0)
                print(f"  Tick (60s)  : {BOLD}{tm_c}{tm_dir}{RESET}  "
                      f"bull:{br:.0%}  up:{tick_m.get('up_ticks',0)} "
                      f"dn:{tick_m.get('down_ticks',0)}  "
                      f"total:{tick_m.get('tick_count',0)}  "
                      f"chg:{chg:+.3f}")

            # Order book
            ob_bias_ = ob.get("bias", "NEUTRAL")
            if ob.get("bid_vol", 0) + ob.get("ask_vol", 0) > 0:
                ob_c  = GREEN if ob_bias_ == "BUY" else RED if ob_bias_ == "SELL" else YELLOW
                imbal = ob.get("imbalance", 0)
                print(f"  OrderBook   : {BOLD}{ob_c}{ob_bias_}{RESET}  "
                      f"imbalance:{imbal:+.2f}  "
                      f"bid:{ob.get('bid_vol',0):.1f}  "
                      f"ask:{ob.get('ask_vol',0):.1f}")
            else:
                print(f"  OrderBook   : {DIM}N/A (DOM tidak tersedia di broker ini){RESET}")

            print(f"  RT Score    : {BOLD}{rt_c}{rt_bias_}{RESET}  ({rt_score_:+.2f})")
            print(sep)

        if ml:
            d = ml["direction"]
            ml_color = GREEN if d == "BUY" else RED if d == "SELL" else YELLOW
            tr_sym = ml.get("trained_symbol", self.train_result.get("trained_symbol", self.symbol))
            tr_tf  = ml.get("trained_tf",     self.train_result.get("trained_timeframe", self.timeframe))
            tr_n   = ml.get("trained_candles", self.train_result.get("trained_n_candles", 0))
            sym_ok = ml.get("symbol_match", True)
            sym_c  = GREEN if sym_ok else RED
            print(f"  {BOLD}--- ML PREDICTION ----------------------------------------{RESET}")
            print(f"  Model       : {ML_MODEL_TYPE.upper()} (Ensemble+FeatureSelect+SMC)")
            print(f"  Trained on  : {sym_c}{BOLD}{tr_sym}{RESET} / {tr_tf}  ({tr_n:,} candles)"
                  + (f"  {RED}[!] SYMBOL MISMATCH{RESET}" if not sym_ok else ""))
            _cacc_val   = self.train_result.get('conf_accuracy', 0)
            _prec_val   = self.train_result.get('precision_buy', 0)
            _f1_val     = self.train_result.get('f1', 0)
            _ml_active  = _cacc_val >= ML_MIN_CONFIDENT_ACC
            _cacc_color = GREEN if _ml_active else (YELLOW if _cacc_val >= 70 else RED)
            _ml_status  = (f"{GREEN}✓ ML-Only AKTIF{RESET}" if _ml_active else
                           f"{YELLOW}~ ML-Only NONAKTIF (hanya confirm/veto){RESET}")
            print(f"  Overall Acc : {self.train_result.get('accuracy', 0)}%  |  "
                  f"Conf Acc: {BOLD}{_cacc_color}{_cacc_val}%{RESET}  "
                  f"[threshold: {ML_MIN_CONFIDENT_ACC}%]")
            print(f"  Precision   : {_prec_val}%  |  F1: {_f1_val}%  |  "
                  f"Vote threshold: {ML_VOTE_THRESHOLD}%")
            print(f"  ML Status   : {_ml_status}")
            print(f"  Prediction  : {BOLD}{ml_color}{d}{RESET}  (confidence: {ml['confidence']}%)")
            if ml.get("uncertain"):
                print(f"  {YELLOW}[!] Confidence di bawah threshold - sinyal lemah{RESET}")
            if ml.get("warning"):
                print(f"  {RED}[!] {ml['warning']}{RESET}")
            print(f"  Proba BUY   : {GREEN}{ml['proba_buy']}%{RESET}  |  Proba SELL: {RED}{ml['proba_sell']}%{RESET}")
            print(sep)

        lstm = result.get("lstm_pred", {})
        if lstm and lstm.get("direction") != "UNKNOWN":
            ld = lstm["direction"]
            lc = GREEN if ld == "BUY" else RED if ld == "SELL" else YELLOW
            print(f"  {BOLD}--- LSTM (TensorFlow/Keras) ------------------------------{RESET}")
            print(f"  Model       : Bidirectional LSTM + Conv1D")
            print(f"  Overall Acc : {self.lstm_result.get('accuracy', 0)}%  |  "
                  f"Confident Acc: {BOLD}{self.lstm_result.get('conf_accuracy', 0)}%{RESET}")
            print(f"  Prediction  : {BOLD}{lc}{ld}{RESET}  (confidence: {lstm['confidence']}%)")
            if lstm.get("uncertain"):
                print(f"  {YELLOW}[!] LSTM confidence rendah{RESET}")
            print(f"  Proba BUY   : {GREEN}{lstm['proba_buy']}%{RESET}  |  Proba SELL: {RED}{lstm['proba_sell']}%{RESET}")
            print(sep)

        fa         = result.get("final_advice", result["consensus"])
        exec_dir   = result.get("exec_direction", "WAIT")
        exec_src   = result.get("exec_source", "")
        ex_color   = GREEN if exec_dir == "BUY" else RED if exec_dir == "SELL" else YELLOW

        # News daily hint
        nb = result.get("news_bias", {})
        nb_bias  = nb.get("bias", "NEUTRAL")
        nb_score = nb.get("score", 0)
        nb_color = GREEN if nb_bias == "BULLISH" else RED if nb_bias == "BEARISH" else YELLOW

        print(f"  {BOLD}--- FINAL DECISION ---------------------------------------{RESET}")

        # Market State
        _mstate   = result["signal"].get("market_state", "?")
        _ms_color = GREEN if _mstate == "TREND" else (YELLOW if _mstate == "UNCLEAR" else RED)
        _conf_val = result["signal"].get("confidence", 0)
        _conf_max = 7  # 6 from signals + 1 ML
        _conf_c   = GREEN if _conf_val >= 5 else (YELLOW if _conf_val >= 4 else RED)
        print(f"  Market State: {BOLD}{_ms_color}{_mstate}{RESET}  |  "
              f"Confidence: {BOLD}{_conf_c}{_conf_val}/{_conf_max}{RESET}")

        # Session Bias (dari analisis jam tutup sesi)
        try:
            from data.session_bias import get_current_bias
            _sb = get_current_bias()
            if _sb and _sb.get("direction") in ("BUY", "SELL", "NEUTRAL"):
                _sb_dir = _sb["direction"]
                _sb_str = _sb.get("strength", "")
                _sb_sco = _sb.get("score", 0)
                _sb_ses = _sb.get("session", "")
                _sb_ts  = _sb.get("timestamp", "")[:16]
                _sb_c   = GREEN if _sb_dir == "BUY" else (RED if _sb_dir == "SELL" else YELLOW)
                print(f"  Session Bias: {BOLD}{_sb_c}{_sb_dir}{RESET}  "
                      f"(skor: {_sb_sco:+.1f}, {_sb_str})  "
                      f"{DIM}← {_sb_ses} close  {_sb_ts}{RESET}")
        except Exception:
            pass

        print(f"  News Hari ini: {BOLD}{nb_color}{nb_bias}{RESET}  "
              f"(skor: {nb_score:+.1f})  "
              f"{DIM}← hint harian saja{RESET}")
        print(f"  Consensus   : {result['consensus']}")

        pnum      = result.get("priority_num", 0)
        src_color = GREEN if "★★★" in exec_src else (
                    GREEN if "★★" in exec_src  else (
                    GREEN if "★"  in exec_src  else (
                    RED   if ("CANCEL" in exec_src or "VETO" in exec_src or "GUARD" in exec_src)
                    else CYAN)))
        print(f"  Priority    : {CYAN}#{pnum}{RESET}")
        print(f"  {BOLD}EKSEKUSI    : {ex_color}{exec_dir}{RESET}  "
              f"{src_color}({exec_src}){RESET}")

        # Risk note
        _risk = result.get("final_advice", "")
        if "⚠" in _risk or "News:" in _risk:
            print(f"  {YELLOW}Risk Note   : {_risk}{RESET}")

        if exec_dir == "WAIT":
            src = result.get("exec_source", "")
            if   "CANCEL"    in src:  wait_reason = "ML cancels rule signal"
            elif "GUARD"     in src:  wait_reason = "Max posisi aktif tercapai — tunggu posisi tutup"
            elif "NoSignal"  in src:  wait_reason = "Tidak ada signal cukup kuat"
            elif "SKIP"      in src:  wait_reason = "Filter block — lihat filter log"
            else:                     wait_reason = "Menunggu kondisi terpenuhi"
            print(f"  {YELLOW}[!] WAIT — {wait_reason}{RESET}")

        # Akurasi historis signal serupa dari candle log
        sa = result.get("sig_accuracy", {})
        if sa and sa.get("total", 0) >= 3:
            sa_acc  = sa["accuracy"]
            sa_tot  = sa["total"]
            sa_win  = sa["win"]
            sa_loss = sa["loss"]
            sa_pct  = sa.get("avg_pct", 0)
            sa_c    = GREEN if sa_acc >= 60 else YELLOW if sa_acc >= 50 else RED
            rel_tag = f"  {GREEN}[RELIABLE]{RESET}" if sa.get("reliable") else ""
            print(f"  Hist Akurasi: {BOLD}{sa_c}{sa_acc}%{RESET}  "
                  f"({sa_win}W/{sa_loss}L dari {sa_tot} signal serupa)  "
                  f"avg:{sa_pct:+.3f}%{rel_tag}")

        print(sep)

        print()
