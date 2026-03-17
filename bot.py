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
from core.indicators import add_all_indicators
from core.signals import generate_signal
from ml.model import CandlePredictor
from ml.deep_model import LSTMPredictor, TF_AVAILABLE
from data.news_filter import NewsFilter


def fetch_data(symbol_key: str = DEFAULT_SYMBOL,
               timeframe: str  = DEFAULT_TIMEFRAME,
               period: str     = None) -> pd.DataFrame:
    """Ambil data OHLCV dari Yahoo Finance"""
    ticker = SYMBOLS.get(symbol_key, symbol_key)
    period = period or DATA_PERIOD.get(timeframe, "60d")

    # Yahoo Finance interval mapping
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
    """Identifikasi sesi trading aktif"""
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
                 use_lstm: bool = False, use_news: bool = True):
        self.symbol        = symbol
        self.timeframe     = timeframe
        self.use_lstm      = use_lstm and TF_AVAILABLE
        self.use_news      = use_news
        self.predictor     = CandlePredictor(timeframe=timeframe)
        self.lstm          = LSTMPredictor(timeframe=timeframe) if self.use_lstm else None
        self.news_filter   = NewsFilter(symbol=symbol) if use_news else None
        self.df            = pd.DataFrame()
        self.df_ind        = pd.DataFrame()
        self.trained       = False
        self.lstm_trained  = False
        self.train_result  = {}
        self.lstm_result   = {}
        self.news_sentiment= {}

    def load_data(self) -> bool:
        """Load & proses data"""
        print(f"[~] Fetching {self.symbol} data ({self.timeframe})...")
        raw = fetch_data(self.symbol, self.timeframe)
        if raw.empty:
            return False
        self.df     = raw
        self.df_ind = add_all_indicators(raw)
        print(f"[OK] {len(self.df_ind)} candles loaded")
        return True

    def train_model(self):
        """Train ML model"""
        if self.df_ind.empty:
            print("[!] Data belum di-load")
            return
        print(f"[~] Training ML model ({ML_MODEL_TYPE.upper()})...")
        result = self.predictor.train(self.df_ind)
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
        """Ambil berita & economic calendar"""
        if self.use_news and self.news_filter:
            print(f"[~] Fetching news & economic calendar...")
            self.news_sentiment = self.news_filter.get_sentiment()
            risk = self.news_sentiment.get("risk_level", "UNKNOWN")
            print(f"[OK] News fetched - Risk: {risk} | {self.news_sentiment.get('total_news', 0)} articles")

    def analyze(self) -> dict:
        """Analisis lengkap: rule-based + ML + LSTM + News"""
        if self.df_ind.empty:
            return {}

        # Rule-based signal
        sig = generate_signal(self.df_ind)

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

        # Consensus: hitung dari semua model
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
            consensus = "NO SIGNAL"
        elif buy_votes > sell_votes:
            consensus = f"BUY  ({buy_votes}/{len(votes)} votes)"
        elif sell_votes > buy_votes:
            consensus = f"SELL ({sell_votes}/{len(votes)} votes)"
        else:
            consensus = "SPLIT (tie)"

        # News risk override
        news_risk = self.news_sentiment.get("risk_level", "UNKNOWN")
        if news_risk == "HIGH":
            final_advice = "AVOID - High Impact News Active!"
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
            "signal":       sig,
            "ml_pred":      ml_pred,
            "lstm_pred":    lstm_pred,
            "consensus":    consensus,
            "final_advice": final_advice,
            "news_risk":    news_risk,
            "news_sentiment": self.news_sentiment,
        }
        return result

    def print_analysis(self, result: dict) -> None:
        """Cetak hasil analisis ke terminal"""
        if not result:
            print("[!] Tidak ada hasil analisis")
            return

        sig    = result["signal"]
        ml     = result.get("ml_pred", {})
        direct = sig["direction"]

        # Warna terminal (ANSI)
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

        # Price
        print(f"  Close : {result['close']:.5f}  |  Open: {result['open']:.5f}")
        print(f"  High  : {result['high']:.5f}  |  Low : {result['low']:.5f}")
        print(sep)

        # Indicators
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
        print(f"  EMA {EMA_FAST}/{EMA_SLOW}/{EMA_TREND}  : {result[f'ema{EMA_FAST}']:.5f} / {result[f'ema{EMA_SLOW}']:.5f} / {result[f'ema{EMA_TREND}']:.5f}")
        print(f"  BB Upper/Low: {result['bb_upper']:.5f} / {result['bb_lower']:.5f}")
        print(sep)

        # Rule-based Signal
        print(f"  {BOLD}--- RULE-BASED SIGNAL -----------------------------------{RESET}")
        print(f"  Score       : {sig['score']:+.2f} / 10.0")
        print(f"  Direction   : {BOLD}{color}{direct}{RESET}")
        if sig["sl"]:
            print(f"  Stop Loss   : {sig['sl']:.5f}")
            print(f"  Take Profit : {sig['tp']:.5f}")
            print(f"  R:R Ratio   : 1:{sig['rr_ratio']}")

        # Reasons
        print(f"\n  Reasons:")
        for score, reason in sig.get("reasons", []):
            r_color = GREEN if score > 0 else RED if score < 0 else YELLOW
            arrow = "^" if score > 0 else "v" if score < 0 else "-"
            print(f"    {r_color}{arrow} {reason}{RESET}")
        print(sep)

        # ML Prediction
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

        # LSTM Prediction
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

        # Final Consensus & News
        news_risk = result.get("news_risk", "UNKNOWN")
        nr_color  = RED if news_risk == "HIGH" else YELLOW if news_risk == "MEDIUM" else GREEN
        fa        = result.get("final_advice", result["consensus"])
        fa_color  = RED if "AVOID" in fa else GREEN if "BUY" in fa else RED if "SELL" in fa else YELLOW

        print(f"  {BOLD}--- FINAL DECISION ---------------------------------------{RESET}")
        print(f"  News Risk   : {BOLD}{nr_color}{news_risk}{RESET}")
        print(f"  Consensus   : {result['consensus']}")
        print(f"  DECISION    : {BOLD}{fa_color}{fa}{RESET}")
        print(sep)

        print()
