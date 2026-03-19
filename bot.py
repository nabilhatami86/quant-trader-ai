"""
Trading Bot - Data Fetcher & Main Logic
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
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
    hour = datetime.utcnow().hour
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
        self.use_lstm      = use_lstm and TF_AVAILABLE
        self.use_news      = use_news
        self.mt5_conn      = mt5_connector
        self.use_tv        = use_tv
        self.tv_user       = tv_user
        self.tv_pass       = tv_pass
        self.predictor     = CandlePredictor(timeframe=timeframe)
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
            raw = self.mt5_conn.get_ohlcv(self.symbol, self.timeframe, count=1500)
            if raw.empty:
                print("[!] MT5 data kosong, fallback ke Yahoo Finance...")
                raw = fetch_data(self.symbol, self.timeframe)
            else:
                print(f"[MT5] {len(raw)} candles live dari broker  "
                      f"(close terakhir: {float(raw['Close'].iloc[-1]):.5f})")
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
        self.df_ind  = add_all_indicators(raw)
        self.df_ind_hist = add_all_indicators(merged) if len(merged) > len(raw) else self.df_ind

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
              f"dengan {len(train_data)} candles historis...")
        result = self.predictor.train(train_data)
        self.trained     = True
        self.train_result = result
        acc   = result['accuracy']
        cacc  = result['conf_accuracy']
        grade = "EXCELLENT" if cacc >= 80 else "GOOD" if cacc >= 65 else "FAIR" if cacc >= 55 else "LOW"
        print(f"[OK] Model trained!")
        print(f"     ----------------------------------------")
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

    def analyze(self) -> dict:
        if self.df_ind.empty:
            return {}

        # Ambil news bias jika tersedia
        news_bias = self.news_sentiment.get("direction_bias") if self.news_sentiment else None

        # Rule-based signal — sertakan news bias
        sig = generate_signal(self.df_ind, news_bias=news_bias)

        # ML prediction (Random Forest / Gradient Boosting)
        ml_pred = {}
        if ML_ENABLED and self.trained:
            ml_pred = self.predictor.predict(self.df_ind)

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

        # Prioritas: rule-based > ML-only (jika confidence tinggi) > WAIT
        ml_conf  = ml_pred.get("confidence", 0) if ml_pred else 0
        ml_dir   = ml_pred.get("direction", "WAIT") if ml_pred else "WAIT"

        if sig["direction"] in ("BUY", "SELL"):
            # Rule-based sudah ada sinyal — pakai ini (sudah ada TP/SL)
            exec_direction = sig["direction"]
            exec_source    = "Rule-Based"
        elif ml_dir in ("BUY", "SELL") and ml_conf >= 65 and not ml_pred.get("uncertain"):
            # Rule-based WAIT tapi ML confident ≥ 65% — ikut ML
            # Rebuild sinyal dengan direction dari ML, TP/SL dari rule-based engine
            exec_direction = ml_dir
            exec_source    = f"ML({ml_conf}%)"
            # Update sig direction & hitung ulang SL/TP
            from analysis.signals import calculate_smart_tp_sl
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
        else:
            exec_direction = "WAIT"
            exec_source    = "WAIT"

        news_risk  = self.news_sentiment.get("risk_level", "UNKNOWN")
        news_bias_ = (news_bias or {}).get("bias", "NEUTRAL")

        # Blok hanya jika news HIGH berlawanan arah signal
        news_conflicts = (
            (exec_direction == "BUY"  and news_bias_ == "BEARISH") or
            (exec_direction == "SELL" and news_bias_ == "BULLISH")
        )
        if news_risk == "HIGH" and news_conflicts:
            final_advice   = "AVOID - High Impact News berlawanan sinyal!"
            exec_direction = "WAIT"
        elif news_risk == "HIGH":
            final_advice = f"{consensus} [HIGH news - searah, lanjut dengan hati-hati]"
        elif news_risk == "MEDIUM":
            final_advice = f"{consensus} [with caution - MEDIUM news]"
        else:
            final_advice = consensus

        result = {
            "symbol":       self.symbol,
            "timeframe":    self.timeframe,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "final_advice":    final_advice,
            "news_risk":       news_risk,
            "news_bias":       news_bias or {},
            "news_sentiment": self.news_sentiment,
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
        tech_sc = sig.get("score_technical", sig["score"])
        news_sc = sig.get("score_news", 0)
        print(f"  Score       : {sig['score']:+.2f} / 10.0  "
              f"(Teknikal: {tech_sc:+.2f}  |  Berita: {news_sc:+.2f})")
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

        if ml:
            d = ml["direction"]
            ml_color = GREEN if d == "BUY" else RED if d == "SELL" else YELLOW
            print(f"  {BOLD}--- ML PREDICTION ----------------------------------------{RESET}")
            print(f"  Model       : {ML_MODEL_TYPE.upper()} (Ensemble+FeatureSelect)")
            print(f"  Overall Acc : {self.train_result.get('accuracy', 0)}%  |  "
                  f"Confident Acc: {BOLD}{self.train_result.get('conf_accuracy', 0)}%{RESET}")
            print(f"  Precision   : {self.train_result.get('precision_buy', 0)}%  |  "
                  f"F1: {self.train_result.get('f1', 0)}%")
            print(f"  Prediction  : {BOLD}{ml_color}{d}{RESET}  (confidence: {ml['confidence']}%)")
            if ml.get("uncertain"):
                print(f"  {YELLOW}[!] Confidence di bawah threshold - sinyal lemah{RESET}")
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

        news_risk  = result.get("news_risk", "UNKNOWN")
        nr_color   = RED if news_risk == "HIGH" else YELLOW if news_risk == "MEDIUM" else GREEN
        fa         = result.get("final_advice", result["consensus"])
        exec_dir   = result.get("exec_direction", "WAIT")
        exec_src   = result.get("exec_source", "")
        ex_color   = GREEN if exec_dir == "BUY" else RED if exec_dir == "SELL" else YELLOW

        print(f"  {BOLD}--- FINAL DECISION ---------------------------------------{RESET}")
        print(f"  News Risk   : {BOLD}{nr_color}{news_risk}{RESET}")
        print(f"  Consensus   : {result['consensus']}")
        print(f"  {BOLD}EKSEKUSI    : {ex_color}{exec_dir}{RESET}  "
              f"{CYAN}(dari: {exec_src}){RESET}")
        if exec_dir == "WAIT" and result["consensus"] not in ("NO SIGNAL", "SPLIT (tie)"):
            ml_c = result.get("ml_pred", {}).get("confidence", 0)
            print(f"  {YELLOW}[!] Sinyal ada tapi belum masuk — "
                  f"ML conf {ml_c}% (min 65% untuk auto-trade){RESET}")
        print(sep)

        print()
