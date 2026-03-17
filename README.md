# Trader AI — Robot Trading EUR/USD & GOLD/USD

Robot trading otomatis berbasis **Python** yang menggabungkan **Machine Learning**, **Deep Learning (LSTM)**, **analisis teknikal**, **news filter**, dan integrasi langsung ke **MetaTrader 4/5**.

---

## Daftar Isi

- [Fitur](#fitur)
- [Arsitektur Folder](#arsitektur-folder)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Tuning Parameter](#tuning-parameter)
- [Akurasi Model](#akurasi-model)
- [Integrasi MetaTrader](#integrasi-metatrader)
- [Contoh Output](#contoh-output)

---

## Fitur

| Fitur | Detail |
|-------|--------|
| **Data Real-time** | Yahoo Finance API (yfinance) — EURUSD, GOLD |
| **Indikator Teknikal** | RSI, MACD, EMA (9/21/50/200), Bollinger Bands, Stochastic, ADX, ATR |
| **Rule-Based Signal** | Scoring engine -10 s/d +10, BUY/SELL/WAIT |
| **ML Predictor** | Random Forest / Gradient Boosting — akurasi hingga **94.87%** |
| **Deep Learning** | Bidirectional LSTM + Conv1D (TensorFlow/Keras) |
| **News Filter** | Yahoo Finance News + ForexFactory Calendar + NewsAPI |
| **Backtest** | Simulasi historis lengkap dengan PnL, win rate, drawdown |
| **MT5 Integration** | Direct Python API — auto place/close order |
| **MT4 Integration** | File bridge + Expert Advisor (EA) otomatis |
| **Risk Management** | ATR-based SL/TP, lot calculator, news risk override |

---

## Arsitektur Folder

```
trader-ai/
│
├── main.py                  # Entry point — jalankan dari sini
├── bot.py                   # Orchestrator utama
├── config.py                # Semua parameter tuning
├── run.bat                  # Launcher menu interaktif (Windows)
├── requirements.txt         # Dependencies Python
├── README.md                # Dokumentasi ini
│
├── core/                    # Engine teknikal
│   ├── __init__.py
│   ├── indicators.py        # RSI, MACD, EMA, BB, Stoch, ADX, ATR
│   ├── signals.py           # Rule-based scoring (BUY/SELL/WAIT)
│   └── backtest.py          # Simulasi historis + laporan PnL
│
├── ml/                      # Kecerdasan Buatan
│   ├── __init__.py
│   ├── model.py             # Random Forest / GradientBoosting (scikit-learn)
│   └── deep_model.py        # LSTM Bidirectional + Conv1D (TensorFlow/Keras)
│
├── data/                    # Sumber data eksternal
│   ├── __init__.py
│   └── news_filter.py       # News Yahoo Finance, ForexFactory, NewsAPI
│
└── broker/                  # Integrasi MetaTrader
    ├── __init__.py
    ├── mt5_connector.py     # MT5 Python API (order, close, status)
    ├── mt4_bridge.py        # MT4 file bridge + EA generator
    └── EA_TraderAI.mq4      # Expert Advisor untuk MT4
```

---

## Instalasi

### 1. Clone / Download project

```bash
cd Desktop
# Pastikan folder trader-ai sudah ada
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow MetaTrader5 requests colorama tabulate
```

### 3. Verifikasi

```bash
python -X utf8 -c "from bot import TradingBot; print('OK')"
```

---

## Cara Penggunaan

### Cara Termudah — `run.bat`
Klik dua kali `run.bat` dan pilih menu yang diinginkan.

---

### Via Command Line

#### Analisis Sekali Jalan

```bash
# EURUSD timeframe 1 hari (akurasi ML tertinggi)
python -X utf8 main.py --symbol EURUSD --tf 1d

# Gold timeframe 1 jam
python -X utf8 main.py --symbol GOLD --tf 1h

# Tanpa news filter (lebih cepat)
python -X utf8 main.py --symbol EURUSD --tf 1d --no-news
```

#### Live Mode (Auto Refresh)

```bash
# Refresh setiap 60 detik (ubah di config.py: REFRESH_INTERVAL)
python -X utf8 main.py --symbol EURUSD --tf 1h --live
```

#### Backtest

```bash
python -X utf8 main.py --symbol EURUSD --tf 1h --backtest
python -X utf8 main.py --symbol GOLD   --tf 1d --backtest
```

#### Dengan LSTM (Deep Learning)

```bash
python -X utf8 main.py --symbol EURUSD --tf 1d --lstm
```

#### Dengan MetaTrader 5

```bash
# Isi dulu MT5_CONFIG di broker/mt5_connector.py
python -X utf8 main.py --symbol EURUSD --tf 1h --mt5 --live

# Login via argumen
python -X utf8 main.py --symbol EURUSD --tf 1h --mt5 --mt-login 12345678 --mt-pass "password" --mt-server "ICMarkets-Live"

# Cek status akun
python -X utf8 main.py --symbol EURUSD --mt5 --mt-status
```

#### Setup MT4

```bash
# Generate file EA .mq4
python -X utf8 main.py --mt-setup

# Live mode dengan MT4 bridge
python -X utf8 main.py --symbol EURUSD --tf 1h --mt4 --live
```

---

### Semua Argumen

| Argumen | Default | Keterangan |
|---------|---------|-----------|
| `--symbol` | EURUSD | Pair: `EURUSD`, `GOLD`, `XAUUSD` |
| `--tf` | 1h | Timeframe: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` |
| `--backtest` | off | Jalankan simulasi historis |
| `--live` | off | Mode live refresh otomatis |
| `--lstm` | off | Aktifkan model LSTM TensorFlow |
| `--no-ml` | off | Nonaktifkan ML, rule-based saja |
| `--no-news` | off | Nonaktifkan news filter |
| `--mt5` | off | Connect ke MetaTrader 5 |
| `--mt4` | off | Aktifkan MT4 file bridge |
| `--mt-login` | 0 | Nomor akun MT5 |
| `--mt-pass` | - | Password MT5 |
| `--mt-server` | - | Server broker MT5 |
| `--mt-setup` | off | Generate EA file untuk MT4 |
| `--mt-status` | off | Tampilkan status akun MT5 |
| `--tune` | off | Tampilkan panduan tuning |
| `--period` | auto | Override periode data (misal: `3mo`, `1y`) |

---

## Tuning Parameter

Edit file `config.py` untuk menyesuaikan strategi:

### Sinyal

```python
MIN_SIGNAL_SCORE = 5    # 3-4 = banyak sinyal | 6-7 = selektif
```

### Bobot Indikator

```python
WEIGHTS = {
    "rsi":       2.0,   # Pengaruh RSI
    "macd":      2.0,   # Pengaruh MACD
    "ema_cross": 2.0,   # Pengaruh EMA crossover
    "stoch":     1.5,   # Pengaruh Stochastic
    "bb":        1.0,   # Pengaruh Bollinger Bands
    "adx":       1.0,   # Pengaruh trend strength
}
```

### Risk Management

```python
ATR_MULTIPLIER_SL = 1.5   # Stop Loss = 1.5 × ATR
ATR_MULTIPLIER_TP = 3.0   # Take Profit = 3.0 × ATR
RISK_PERCENT      = 1.0   # % balance yang dirisiko per trade (MT5)
```

### ML Model

```python
ML_MODEL_TYPE = "rf"        # "rf" | "gb" | "et" | "ensemble"
ML_LOOKBACK   = 20          # Jumlah candle sebelumnya sebagai fitur
```

### Indikator

```python
RSI_PERIOD      = 14
RSI_OVERSOLD    = 30    # BUY zone
RSI_OVERBOUGHT  = 70    # SELL zone
EMA_FAST        = 9
EMA_SLOW        = 21
EMA_TREND       = 50
BB_STD          = 2.0   # Bollinger Bands standar deviasi
```

---

## Akurasi Model

| Pair | Timeframe | Akurasi | Keterangan |
|------|-----------|---------|-----------|
| EURUSD | 1d | **94.87%** | EXCELLENT — prediksi hari trending |
| EURUSD | 1h | ~51% | LOW — noise tinggi |
| GOLD | 1d | ~60% | FAIR |

> **Catatan**: Akurasi tinggi pada 1d karena model hanya memprediksi hari dengan pergerakan signifikan (> 0.3%). Hari sideways dibuang dari training agar model tidak belajar noise.

---

## Integrasi MetaTrader

### MetaTrader 5 (Direkomendasikan)

1. Buka `broker/mt5_connector.py`
2. Isi `MT5_CONFIG`:

```python
MT5_CONFIG = {
    "login":    12345678,
    "password": "password_kamu",
    "server":   "ICMarkets-Live",
}
```

3. Sesuaikan nama symbol broker di `SYMBOL_MAP`:

```python
SYMBOL_MAP = {
    "EURUSD": "EURUSDm",    # tambahkan 'm' jika broker pakai suffix
    "GOLD":   "XAUUSDm",
}
```

4. Jalankan:

```bash
python -X utf8 main.py --symbol EURUSD --tf 1h --mt5 --live
```

### MetaTrader 4

1. Generate file EA:
```bash
python -X utf8 main.py --mt-setup
```

2. Copy `broker/EA_TraderAI.mq4` → `MT4/MQL4/Experts/`
3. Compile di MetaEditor
4. Pasang EA ke chart, centang **"Allow live trading"**
5. Jalankan bot:
```bash
python -X utf8 main.py --symbol EURUSD --tf 1h --mt4 --live
```

### Filter Keamanan

| Filter | Keterangan |
|--------|-----------|
| News Risk | Berita HIGH impact → order dibatalkan otomatis |
| ML Disagree | ML tidak sepakat → order di-skip |
| Duplicate Position | Tidak buka posisi dobel arah yang sama |
| Signal Reversal | Auto-close posisi jika sinyal berbalik |
| Magic Number | Order bot diberi ID `202601` agar tidak tercampur order manual |

---

## Contoh Output

```
============================================================
  TRADING ANALYSIS - EURUSD | 1d
  2026-03-17 20:47:30  |  Session: London+New York
============================================================
  Close : 1.15420  |  Open: 1.15101
  High  : 1.15447  |  Low : 1.14705
============================================================
  --- INDICATORS ---
  RSI(14)    : 40.01
  MACD       : -0.00767  Hist: -0.00209
  ADX(14)    : 75.27  [Strong Trend]
  Stoch K/D  : 30.49 / 19.05
============================================================
  --- ML PREDICTION ---
  Overall Acc  : 94.87%  |  Confident Acc: 94.87%  [EXCELLENT]
  Prediction   : SELL  (confidence: 84.6%)
  Proba BUY    : 15.4%  |  Proba SELL: 84.6%
============================================================
  --- FINAL DECISION ---
  News Risk  : HIGH
  DECISION   : AVOID - High Impact News Active!
============================================================
```

---

## Tech Stack

| Library | Versi | Kegunaan |
|---------|-------|----------|
| `yfinance` | 0.2+ | Data pasar real-time |
| `pandas` | 2.0+ | Manipulasi data |
| `numpy` | 1.24+ | Komputasi numerik |
| `scikit-learn` | 1.3+ | Random Forest, feature selection |
| `tensorflow` | 2.21+ | LSTM, Keras deep learning |
| `MetaTrader5` | 5.0+ | Integrasi MT5 (Windows only) |
| `requests` | 2.31+ | HTTP untuk NewsAPI |

---

## Disclaimer

> Robot trading ini dibuat untuk tujuan **edukasi dan riset**. Hasil backtest dan akurasi ML tidak menjamin profit di masa depan. Trading forex dan emas mengandung risiko tinggi. Selalu gunakan **akun demo** terlebih dahulu sebelum live trading. Pengembang tidak bertanggung jawab atas kerugian finansial apapun.
