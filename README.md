# Trader AI — Robot Trading GOLD & Multi-Pair

Robot trading otomatis berbasis **Python** yang menggabungkan **Machine Learning ensemble (LightGBM + XGBoost + RF)**, **analisis teknikal lengkap (20+ indikator)**, **Smart Money Concepts**, **prediksi berita (News Lag Effect)**, dan eksekusi langsung ke **MetaTrader 5** — dilengkapi **REST API (FastAPI)** dan penyimpanan ke **PostgreSQL**.

---

## Daftar Isi

- [Stack Teknologi](#stack-teknologi)
- [Arsitektur Folder](#arsitektur-folder)
- [Alur Sistem](#alur-sistem)
- [Indikator & Sinyal](#indikator--sinyal)
- [Machine Learning](#machine-learning)
- [Fitur Lengkap](#fitur-lengkap)
- [Instalasi](#instalasi)
- [Konfigurasi `.env`](#konfigurasi-env)
- [Cara Menjalankan](#cara-menjalankan)
- [REST API](#rest-api)
- [Trading Mode](#trading-mode)
- [Dynamic Lot Sizing](#dynamic-lot-sizing)
- [Semua Argumen CLI](#semua-argumen-cli)
- [Tuning Parameter](#tuning-parameter)
- [Symbol yang Didukung](#symbol-yang-didukung)
- [Disclaimer](#disclaimer)

---

## Stack Teknologi

| Layer | Teknologi |
|-------|-----------|
| **REST API** | FastAPI + Uvicorn |
| **Database** | PostgreSQL + SQLAlchemy 2.0 async + asyncpg |
| **ML Ensemble** | LightGBM + XGBoost + Random Forest (VotingClassifier, dikalibrasi isotonic) |
| **ML Deep (opsional)** | Bidirectional LSTM + Conv1D (TensorFlow) |
| **Broker** | MetaTrader 5 API (data live + auto-order) |
| **Data fallback** | TradingView (tvdatafeed) → Yahoo Finance |
| **Config** | Pydantic BaseSettings + `.env` + `config.py` |
| **Logging** | Python logging → `logs/api.log` |

---

## Arsitektur Folder

```
trader-ai/
│
├── api_main.py          ← Entry point API  (uvicorn api_main:app)
├── main.py              ← Entry point CLI  (python main.py --live --mt5 --real)
├── bot.py               ← Orchestrator bot (load data, train, analyze)
├── config.py            ← Semua parameter teknikal & trading
│
├── api/                 ← HTTP Layer
│   ├── deps.py          ← Dependency injection (db session, bot instance)
│   └── routes/
│       ├── signal.py    ← GET /signal, GET /signal/full, POST /signal/analyze
│       ├── trade.py     ← POST /trade/force, GET /trade/positions
│       ├── journal.py   ← GET /journal, GET /journal/stats
│       ├── backtest.py  ← POST /backtest/run
│       ├── webhook.py   ← POST /webhook/tradingview, POST /admin/migrate
│       └── txlog.py     ← GET /txlog, GET /txlog/summary
│
├── core/                ← Infrastruktur FastAPI
│   ├── config.py        ← Settings dari .env (Pydantic BaseSettings)
│   ├── logging.py       ← Structured logging ke terminal + file
│   └── security.py      ← API key auth via X-API-Key header
│
├── db/                  ← Database Layer
│   ├── database.py      ← SQLAlchemy async engine + session factory
│   ├── models.py        ← 6 tabel ORM
│   └── crud/
│       ├── candles.py   ← bulk_upsert_candles()
│       ├── signals.py   ← insert_signal(), get_recent_signals()
│       ├── trades.py    ← insert_trade(), get_trade_stats()
│       ├── ml_results.py← insert_ml_result()
│       └── tx_log.py    ← log_event(), get_tx_logs()
│
├── schemas/             ← Pydantic request/response models
│   ├── signal.py        ← SignalOut, SignalHistoryOut
│   ├── trade.py         ← TradeOut, ForceTradeRequest, TradeStatsOut
│   ├── candle.py        ← CandleOut
│   └── common.py        ← SuccessResponse, ErrorResponse
│
├── services/            ← Business Logic
│   ├── bot_service.py   ← Bot background thread (singleton, auto-order MT5)
│   ├── db_logger.py     ← Sync wrapper save ke DB dari CLI
│   └── migrate_service.py ← Migrasi semua CSV → PostgreSQL
│
├── analysis/            ← Technical Analysis Engine
│   ├── indicators.py    ← 20+ indikator: RSI, MACD, EMA, BB, Stoch, ADX, ATR,
│   │                       SMA, Fibonacci, SMC (FVG/OB/BOS/Liquidity), RSI Div,
│   │                       Momentum Chain, OBV, VWAP, Williams %R, CCI
│   └── signals.py       ← Scoring engine, Market Regime, Smart TP/SL
│
├── broker/              ← Broker Integration
│   ├── mt5_connector.py ← MT5: connect, order, dynamic lot, trailing, multi-TP
│   ├── mt4_bridge.py    ← MT4 file bridge + generator EA
│   └── EA_TraderAI.mq4  ← Expert Advisor untuk MT4
│
├── ml/                  ← Machine Learning
│   ├── model.py         ← Ensemble: LightGBM + XGBoost + RF (VotingClassifier)
│   │                       TP/SL simulation labeling, 70 features, 91.2% akurasi
│   └── deep_model.py    ← Bidirectional LSTM + Conv1D (TensorFlow, opsional)
│
├── data/                ← Data Handling
│   ├── candle_db.py     ← Simpan & merge candle ke CSV lokal
│   ├── candle_log.py    ← Log pergerakan candle per cycle
│   ├── trade_journal.py ← Catat entry/exit trade, hitung win rate
│   ├── news_filter.py   ← Berita + News Lag Effect + decay scoring
│   ├── tv_feed.py       ← Fetch candle dari TradingView
│   ├── history/         ← Database candle CSV (XAUUSD_1h, XAUUSD_5m, ...)
│   └── news_cache/      ← Cache berita harian per tanggal (JSON)
│
├── backtest/
│   └── engine.py        ← Simulasi historis: PnL, win rate, max drawdown
│
├── scripts/
│   └── backfill.py      ← Backfill data historis ke PostgreSQL
│
├── tests/
│   └── test_order.py    ← Test koneksi & order ke MT5
│
├── utils/
│   ├── datetime_utils.py
│   └── response.py
│
├── logs/                ← Log otomatis (auto-generated)
├── models/              ← Model ML tersimpan (auto-generated)
│
├── .env                 ← Kredensial & konfigurasi (tidak di-commit)
├── .env.example         ← Template .env
├── requirements.txt
└── run.bat              ← Menu launcher (Windows)
```

---

## Alur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                      STARTUP (1x per run)                            │
│  create DB tables → connect MT5 → init bot → fetch news → start     │
└────────────────────────────┬────────────────────────────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │  API Mode (api_main.py)                     │  CLI Mode (main.py)
        │  Background thread setiap 60 detik          │  Loop setiap 60 detik
        └─────────────────────┬──────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                     TIAP CYCLE                                        │
│                                                                       │
│  1. FETCH DATA                                                        │
│     MT5 live → TradingView → Yahoo Finance (fallback)                │
│     Simpan candle baru ke CSV lokal + PostgreSQL                     │
│                                                                       │
│  2. HITUNG INDIKATOR (20+ indikator)                                 │
│     RSI, MACD, EMA, BB, Stochastic, ADX, ATR                        │
│     SMA(10/20/50/200), Fibonacci Retracement (5 level)               │
│     OBV, VWAP, Williams %R, CCI, Volume Divergence                  │
│     Smart Money: FVG, Order Block, BOS/ChoCH, Liquidity Sweep        │
│     RSI Divergence (regular + hidden), Momentum Chain (HH/HL/LL/LH) │
│     Market Regime Detection: TREND / RANGE / VOLATILE                │
│                                                                       │
│  3. TRAIN ML (jika belum tertraining)                                │
│     LightGBM + XGBoost + Random Forest → VotingClassifier           │
│     TP/SL simulation labeling, 70 features, 13.000+ candle          │
│     Akurasi: 91.2% (threshold 70%) / 93.7% (threshold 85%)          │
│                                                                       │
│  4. SCORING & ANALISIS                                                │
│     Traditional: RSI, MACD, EMA cross, BB, Stochastic, ADX          │
│     Volume: OBV, VWAP, Williams %R, CCI, Volume spike               │
│     Smart Money: FVG + Order Block + BOS/ChoCH + Liquidity Sweep    │
│     Structure: RSI Divergence (bobot 4.0), Momentum Chain            │
│     Trend/Level: SMA cross, Fibonacci retracement                    │
│     Market Regime multiplier: TREND×1.3 / VOLATILE×1.2 / RANGE×0.8 │
│     Signal Quality Gate: score < 3.0 → WAIT                         │
│     RANGE tanpa SMC/Divergence → WAIT                                │
│     News Lag Effect → bias bullish/bearish dari berita               │
│     ML prediction → konfirmasi (70% confidence threshold)            │
│                                                                       │
│  5. SMART TP/SL                                                       │
│     SL = 1x ATR  |  TP = 10x ATR  (RR 1:10, real mode)             │
│                                                                       │
│  6. EKSEKUSI KE MT5 (jika sinyal = BUY/SELL)                        │
│     Dynamic lot: base × 1.0–2.0x (sesuai skor) + 0.2–0.4x (ML)     │
│     Breakeven otomatis saat profit = jarak SL                        │
│     Multi-TP: partial close 50% di TP1 (jika aktif)                 │
│     Trailing stop (jika aktif)                                        │
│     DCA: tambah order tiap X menit (jika aktif)                      │
│                                                                       │
│  7. SIMPAN KE POSTGRESQL                                              │
│     signals → candle_logs → tx_log → trades (jika order masuk)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Database (PostgreSQL)

| Tabel | Isi |
|-------|-----|
| `candles` | OHLCV per symbol + timeframe |
| `candle_logs` | Candle + semua indikator per cycle |
| `signals` | Sinyal analisis: direction, score, SL, TP, ML pred, news risk |
| `trades` | Order yang masuk: ticket, entry, SL, TP, lot, result, PnL |
| `ml_results` | Akurasi ML per training cycle |
| `tx_log` | Event log append-only: BOT_START, SIGNAL_*, ORDER_OPEN, BOT_ERROR |

---

## Indikator & Sinyal

### Indikator Teknikal (20+)

| Kategori | Indikator | Keterangan |
|----------|-----------|------------|
| **Trend** | EMA 9/20/50/200 | Exponential Moving Average cross + slope |
| **Trend** | SMA 10/20/50/200 | Simple MA — Golden Cross / Death Cross |
| **Momentum** | RSI(14) | Overbought/oversold + RSI Divergence (regular + hidden) |
| **Momentum** | MACD(12/26/9) | Crossover + histogram slope |
| **Momentum** | Stochastic(14/3/3) | %K/%D cross + extreme zones |
| **Level** | Bollinger Bands(20,2) | Bandwidth, %B, squeeze detection |
| **Level** | Fibonacci Retracement | 5 level dinamis (23.6%–78.6%) dari swing N candle |
| **Volatility** | ATR(14) | Average True Range — dasar SL/TP |
| **Trend strength** | ADX(14) | Kekuatan trend (>25 = trending) |
| **Volume** | OBV | On Balance Volume + EMA(20) |
| **Volume** | VWAP | Volume Weighted Average Price |
| **Volume** | Williams %R | Momentum -100..0 |
| **Volume** | CCI | Commodity Channel Index |
| **Volume** | Volume Divergence | Arah harga vs volume |
| **SMC** | Fair Value Gap | Imbalance bullish/bearish |
| **SMC** | Order Block | Area akumulasi/distribusi institusi |
| **SMC** | BOS/ChoCH | Break of Structure / Change of Character |
| **SMC** | Liquidity Sweep | Pembersihan stop loss di high/low |
| **Structure** | RSI Divergence | Regular & hidden divergence (bobot tertinggi: 4.0) |
| **Structure** | Momentum Chain | HH+HL (bullish) / LL+LH (bearish) — market structure |
| **Pattern** | Candle Pattern | Hammer, Doji, Engulfing, Shooting Star, dll |
| **Pattern** | Extended Pattern | Three Soldiers/Crows, Morning/Evening Star, Harami |

### Bobot Sinyal (`config.py → WEIGHTS`)

| Indikator | Bobot | Keterangan |
|-----------|-------|------------|
| `rsi_div` | **4.0** | RSI Divergence — reversal terkuat |
| `smc` | **3.0** | Smart Money Concepts |
| `rsi` | 2.0 | RSI classic |
| `macd` | 2.0 | MACD cross |
| `ema_cross` | 2.0 | EMA fast/slow cross |
| `fibonacci` | 2.0 | Proximity ke level Fibonacci |
| `momentum_chain` | 2.0 | Market structure chain |
| `obv` | 1.5 | OBV vs EMA |
| `vwap` | 1.5 | Price vs VWAP |
| `stoch` | 1.5 | Stochastic |
| `pattern_ex` | 1.5 | Extended candle patterns |
| `sma` | 1.5 | SMA Golden/Death Cross |
| `bb` | 1.0 | Bollinger Bands |
| `adx` | 1.0 | ADX strength |
| `williams_r` | 1.0 | Williams %R |
| `cci` | 1.0 | CCI |
| `volume` | 1.0 | Volume spike + divergence |
| `candle` | 0.5 | Basic candle pattern |

### Market Regime Detection

Bot mendeteksi kondisi pasar sebelum scoring:

| Regime | Kondisi | Multiplier |
|--------|---------|------------|
| **TREND** | ADX > 25 | ×1.3 — semua sinyal diperkuat |
| **VOLATILE** | ATR spike, BB melebar | ×1.2 — sinyal masih valid |
| **RANGE** | ADX < 20, BB sempit | ×0.8 — sinyal dilemahkan |

> **Signal Quality Gate**: Skor akhir < 3.0 → WAIT. RANGE tanpa SMC atau RSI Divergence → WAIT otomatis.

---

## Machine Learning

### Model

```
VotingClassifier (soft voting)
├── LightGBM (n=200, leaf-wise, calibrated isotonic cv=2)
├── XGBoost  (n=200, regularized boosting, calibrated isotonic cv=2)
└── Random Forest (n=150, balanced class weight)

Feature Selection: SelectKBest (f_classif, k=70)
Scaler: StandardScaler
```

### Labeling — TP/SL Simulation

Label bukan dari forward return biasa, melainkan dari **simulasi trade nyata**:

```
Untuk setiap candle i:
  TP = close[i] + 3 × ATR[i]
  SL = close[i] - 1 × ATR[i]

Cek 30 candle ke depan:
  High[i+k] >= TP lebih dulu → label = 1 (BUY menang)
  Low[i+k]  <= SL lebih dulu → label = 0 (SELL menang)
  Selisih hit <= 2 candle   → NaN (buang, ambiguous)
  Keduanya tidak kena        → NaN (sideways, buang)
```

### Hasil Benchmark (XAUUSD 5m, 13.487 candle)

| Confidence Threshold | Akurasi | Coverage |
|---------------------|---------|----------|
| 50% | 83.1% | 100% |
| 60% | 88.0% | 78.3% |
| 65% | 90.1% | 66.3% |
| **70%** | **91.2%** | **55.6%** ← setting aktif |
| 75% | 91.7% | 44.0% |
| 80% | 93.3% | 29.8% |
| 85% | 93.7% | 15.6% |

> **Setting aktif**: `CONFIDENCE_THRESHOLD = 70%` → akurasi 91.2%, bot hanya eksekusi sinyal yang model sangat yakin.

### 70 Features ML

| Grup | Fitur |
|------|-------|
| **Core indicators** | RSI, MACD, ADX, Stochastic, ATR, BB |
| **EMA** | 4 level (9/20/50/200) + alignment + slope |
| **SMA** | 4 level + Golden/Death cross |
| **Fibonacci** | Jarak ke 3 level utama + posisi retracement |
| **SMC** | 10 sinyal (FVG, OB, BOS, ChoCH, Liquidity) + net composite |
| **RSI Divergence** | 4 tipe (regular + hidden, bull + bear) + net |
| **Momentum Chain** | HH, HL, LL, LH, bull_chain, bear_chain, net |
| **Volume** | OBV, VWAP, Williams %R, CCI, vol_ratio |
| **Candle** | body ratio, shadow, quality score, direction |
| **Multi-window** | RSI mean (3/7/14/21), ROC (3/7/14/21) |
| **Channel** | Posisi di range 10/20/50 bar |
| **Lag features** | 8 lag × 10 key kolom (80 lag features) |
| **Regime** | Encoding TREND/RANGE/VOLATILE |

---

## Fitur Lengkap

| Fitur | Detail |
|-------|--------|
| **Data Live MT5** | Candle real-time langsung dari broker |
| **Fallback otomatis** | MT5 → TradingView → Yahoo Finance |
| **20+ Indikator** | RSI, MACD, EMA, SMA, BB, Stoch, ADX, ATR, Fibonacci, OBV, VWAP, CCI, Williams %R, SMC |
| **Smart Money Concepts** | FVG, Order Block, BOS/ChoCH, Liquidity Sweep |
| **RSI Divergence** | Regular + Hidden divergence (bobot 4.0, sinyal terkuat) |
| **Momentum Chain** | Deteksi HH+HL / LL+LH market structure |
| **Market Regime** | TREND/RANGE/VOLATILE detection + multiplier scoring |
| **Signal Quality Gate** | Score < 3.0 atau RANGE tanpa konfirmasi → WAIT |
| **ML Ensemble** | LightGBM + XGBoost + RF, akurasi 91.2% (threshold 70%) |
| **TP/SL Simulation Label** | Label dari hasil simulasi trade nyata, bukan forward return |
| **LSTM (opsional)** | Bidirectional LSTM + Conv1D (butuh TensorFlow) |
| **News Lag Effect** | Berita berlaku beberapa hari (decay otomatis per kategori) |
| **Candle Memory** | Cari pola serupa dari 13.000+ candle historis |
| **Dynamic Lot Sizing** | Lot otomatis × 1.0–2.0x berdasarkan skor sinyal + ML confidence |
| **Smart TP/SL** | SL/TP dari ATR (RR 1:10 di real mode) |
| **Breakeven Otomatis** | SL geser ke entry saat profit = jarak SL |
| **Multi-TP** | Partial close 50% di TP1 → profit terkunci, 50% bebas risiko |
| **Trailing Stop** | SL ikuti harga otomatis |
| **Bulk Order** | N order sekaligus saat sinyal masuk |
| **DCA Otomatis** | Tambah order tiap X menit selama sinyal searah |
| **Risk-Based SL** | SL dihitung dari maks kerugian USD per order |
| **SL Risk Guard** | Warning 30%, auto-close 10% sebelum SL |
| **REST API** | FastAPI + Swagger UI — akses sinyal, trade, journal via HTTP |
| **PostgreSQL** | Semua data tersimpan permanen |
| **Transaction Log** | Setiap event tercatat ke `tx_log` |
| **Backtest Engine** | Simulasi historis: PnL, win rate, max drawdown |
| **MT4 Bridge** | File bridge + EA untuk MetaTrader 4 |
| **Multi Symbol** | XAUUSD, EURUSD, GBPUSD, XAUEUR, BTCUSD, DXY |

---

## Instalasi

```bash
# 1. Clone project
git clone <repo-url> trader-ai
cd trader-ai

# 2. Install dependensi Python
pip install -r requirements.txt

# 3. Install ML libraries (wajib untuk akurasi penuh)
pip install lightgbm xgboost

# 4. Install MetaTrader5 (Windows, untuk auto-trade)
pip install MetaTrader5

# 5. Install TradingView feed (opsional)
pip install tvdatafeed

# 6. Install TensorFlow (opsional, untuk LSTM)
pip install tensorflow

# 7. Setup PostgreSQL
# Buat database: CREATE DATABASE forex_trading;
# User default: postgres / password: sesuaikan di .env

# 8. Buat file .env
copy .env.example .env
# Edit .env — isi kredensial MT5 dan PostgreSQL
```

---

## Konfigurasi `.env`

```env
# ── MetaTrader 5 ─────────────────────────────────────────────────
MT5_LOGIN=12345678
MT5_PASSWORD=password_kamu
MT5_SERVER=Exness-MT5Trial14
MT5_PATH=                     # kosongkan = auto-detect

# ── PostgreSQL ───────────────────────────────────────────────────
DB_HOST=localhost
DB_PORT=5432
DB_NAME=forex_trading
DB_USER=postgres
DB_PASSWORD=password_db

# ── FastAPI ──────────────────────────────────────────────────────
APP_NAME=Trader AI API
APP_VERSION=1.0.0
DEBUG=false
API_KEY=                      # kosong = tanpa auth (dev mode)

# ── Bot ──────────────────────────────────────────────────────────
BOT_SYMBOL=XAUUSD
BOT_TIMEFRAME=5m
BOT_USE_NEWS=true

# ── MT5 Auto-order ───────────────────────────────────────────────
BOT_USE_MT5=true              # true = API server auto-execute order

# ── Trading Mode ─────────────────────────────────────────────────
BOT_REAL_MODE=true            # true = setting akun real (~$60 USD)
BOT_MICRO_MODE=false          # true = filter ketat, 1 order, lot min

# ── Trading Params (diabaikan jika BOT_REAL_MODE=true) ───────────
BOT_LOT=0.01
BOT_ORDERS=1
BOT_TRAIL_PIPS=0.0
BOT_DCA_MINUTES=0.0
BOT_RISK_USD=0.0
```

### Sesuaikan nama symbol broker

Edit `broker/mt5_connector.py` → `SYMBOL_MAP`:

```python
SYMBOL_MAP = {
    "XAUUSD": "XAUUSDm",   # sesuaikan nama di broker kamu
    "EURUSD": "EURUSDm",
}
```

### Aktifkan Auto Trading di MT5

Klik tombol **Algo Trading** di toolbar MT5 hingga **hijau (ON)**.

---

## Cara Menjalankan

### Opsi 1 — Menu Launcher (Windows, rekomendasi)

```bat
run.bat
```

```
 ─── API SERVER (rekomendasi) ─────────────────────────────
 1. Jalankan API + Bot XAUUSD  [REAL MODE, MT5 auto-order]
 2. Jalankan API + Bot XAUUSD  [DEBUG mode, reload otomatis]

 ─── CLI LIVE MODE ────────────────────────────────────────
 3. XAUUSD 1h  --live --mt5 --real
 4. XAUUSD 5m  --live --mt5 --real
 5. XAUUSD 1h  --live --mt5 --micro
 6. EURUSD 1h  --live --mt5 --real

 ─── ANALISIS SEKALI ──────────────────────────────────────
 7. XAUUSD 1h  (analisis + MT5 status)
 8. XAUUSD 5m
 9. EURUSD 1h
```

### Opsi 2 — API Server (production)

```bash
# Start server — bot otomatis jalan di background thread
uvicorn api_main:app --host 0.0.0.0 --port 8000

# Atau langsung
python api_main.py
```

API server akan:
1. Buat tabel PostgreSQL otomatis
2. Connect ke MT5 (jika `BOT_USE_MT5=true`)
3. Terapkan real mode (jika `BOT_REAL_MODE=true`)
4. Jalankan bot analisis di background setiap 60 detik
5. Auto-execute order saat sinyal BUY/SELL

Akses dokumentasi API: **http://localhost:8000/docs**

### Opsi 3 — CLI Live

```bash
# Real mode — akun ~$60 USD
python main.py --symbol XAUUSD --tf 5m --live --mt5 --real

# Micro mode — akun <1 juta IDR
python main.py --symbol XAUUSD --tf 1h --live --mt5 --micro

# Analisis sekali (tanpa live loop)
python main.py --symbol XAUUSD --tf 1h --mt5

# Backtest
python main.py --symbol XAUUSD --tf 1h --backtest
```

### Backfill data ke PostgreSQL (jalankan sekali)

```bash
python scripts/backfill.py --symbol XAUUSD
python scripts/backfill.py --symbol XAUUSD --tf 1h
```

---

## REST API

Setelah server jalan, buka **http://localhost:8000/docs** untuk Swagger UI.

### Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | Info API & version |
| `GET` | `/health` | Status bot + cycle terakhir |
| `GET` | `/signal` | Sinyal terakhir dari bot |
| `GET` | `/signal/full` | Analisis lengkap + ML + news |
| `POST` | `/signal/analyze` | Trigger analisis manual (satu cycle) |
| `GET` | `/signal/history` | Riwayat sinyal dari DB |
| `POST` | `/trade/force` | Paksa order BUY/SELL ke MT5 |
| `GET` | `/trade/positions` | Posisi aktif MT5 |
| `GET` | `/journal` | Trade journal dari DB |
| `GET` | `/journal/stats` | Statistik win rate & PnL |
| `POST` | `/backtest/run` | Jalankan backtest |
| `POST` | `/webhook/tradingview` | Terima sinyal dari TradingView alert |
| `POST` | `/admin/migrate` | Migrasi semua CSV ke PostgreSQL |
| `GET` | `/txlog` | Transaction log (semua event) |
| `GET` | `/txlog/summary` | Ringkasan event per tipe |

### Contoh response `GET /signal`

```json
{
  "symbol": "XAUUSD",
  "timeframe": "5m",
  "direction": "BUY",
  "score": 6.75,
  "sl": 4843.50,
  "tp": 4925.00,
  "exec_direction": "BUY",
  "news_risk": "MEDIUM",
  "ml_prediction": "BUY",
  "ml_confidence": 78.3,
  "regime": "TREND",
  "timestamp": "2026-03-23T10:15:00"
}
```

### Contoh response `GET /health`

```json
{
  "api": "ok",
  "bot": {
    "running": true,
    "cycle": 42,
    "last_update": "2026-03-23T10:15:00",
    "last_error": "",
    "symbol": "XAUUSD",
    "timeframe": "5m",
    "mt5_active": true
  }
}
```

---

## Trading Mode

### Mode Real (`--real` / `BOT_REAL_MODE=true`)

Untuk akun ~$60 USD. Semua parameter diset otomatis:

| Pengaturan | Nilai |
|---|---|
| Lot per order | Auto-lot (balance/10000) — min 0.01, max 0.10 |
| Jumlah order | 1 (tidak tumpuk) |
| Max tumpuk | 3 (sinyal sangat kuat boleh tambah) |
| DCA | OFF |
| Multi-TP | ON — tutup 50% di TP1, sisa jalan ke TP2 |
| Trailing stop | 15 pips |
| SL | 1× ATR |
| TP | 10× ATR (RR 1:10) |
| ML min confidence | **70%** (akurasi 91.2%) |
| ADX minimum | 28 |
| Risk per trade | 10% balance |

```bash
python main.py --symbol XAUUSD --tf 5m --live --mt5 --real
```

### Mode Mikro (`--micro` / `BOT_MICRO_MODE=true`)

Untuk akun <1 juta IDR. Filter ekstra ketat:

| Pengaturan | Normal | Micro |
|---|---|---|
| Jumlah order | N | **1** |
| Lot | custom | **0.01 (min)** |
| Min skor sinyal | 5/10 | **7/10** |
| ADX minimum | 25 | **30** |
| ML confidence | 70% | **70%** |
| DCA | boleh | **OFF** |
| Risk per trade | - | **0.5% balance** |

```bash
python main.py --symbol XAUUSD --tf 1h --live --mt5 --micro
```

### Lot Aman per Saldo (XAUUSD, leverage 1:200)

| Saldo | Auto-lot | Risk per trade (~2%) |
|-------|----------|----------------------|
| $60   | **0.01** | ~$5 |
| $200  | **0.02** | ~$5 |
| $500  | **0.05** | ~$10 |
| $1,000 | **0.10** | ~$20 |
| $2,500 | **0.10 (capped)** | ~$50 |

---

## Dynamic Lot Sizing

Bot secara otomatis menyesuaikan lot berdasarkan kekuatan sinyal:

```
Base lot = auto-lot dari saldo  (contoh: $100 → 0.01)

Multiplier dari skor sinyal:
  score < 5   → ×1.0  (sinyal lemah — lot minimum)
  score 5–8   → ×1.3  (sinyal sedang)
  score 8–12  → ×1.7  (sinyal kuat)
  score >= 12 → ×2.0  (sinyal sangat kuat)

Bonus ML confidence:
  confidence 70–84% → +0.2×
  confidence >= 85%  → +0.4×

Safety: lot tidak boleh > 40% free margin yang tersedia
```

Contoh: score = 10, confidence = 88%, balance = $200
- Base lot = 0.02
- Score multiplier = 1.7×
- ML bonus = +0.4×
- Final lot = 0.02 × 2.1 = **0.042** (dibulatkan ke 0.04)

---

## Logika Keputusan

```
BUY  : EMA20 > EMA50  AND  RSI < 70  AND  ADX >= 25
       AND  score total >= MIN_SIGNAL_SCORE
       AND  score setelah quality gate >= 3.0
       AND  ML confidence >= 70%
       AND  News tidak HIGH berlawanan sinyal

SELL : EMA20 < EMA50  AND  RSI > 30  AND  ADX >= 25
       AND  score total >= MIN_SIGNAL_SCORE
       AND  score setelah quality gate >= 3.0
       AND  ML confidence >= 70%
       AND  News tidak HIGH berlawanan sinyal

WAIT : kondisi tidak terpenuhi, atau:
       - Regime RANGE tanpa SMC signal dan tanpa RSI Divergence
       - Score < 3.0 (terlalu lemah)
       - ML confidence < 70%
```

### Breakeven Otomatis

```
Entry BUY @ 2500  SL @ 2490  (jarak = 10 poin)
Harga naik ke 2510 (profit = jarak SL)
→ SL otomatis digeser ke 2500.x (entry + buffer kecil)
→ Worst case: impas, bukan rugi
```

### Multi-TP — Profit Dikunci Bertahap

```
Entry BUY @ 2500.00   SL @ 2495.00   TP @ 2550.00
         TP1 @ 2525.00  (50% jarak ke TP penuh)

Harga menyentuh TP1:
→ Tutup 50% posisi → profit terkunci
→ SL geser ke breakeven (2500.xx)
→ Sisa 50% BEBAS RISIKO menuju TP penuh
```

---

## Semua Argumen CLI

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--symbol` | XAUUSD | Pair trading |
| `--tf` | 5m | Timeframe: `1m` `5m` `15m` `1h` `4h` `1d` |
| `--live` | off | Refresh otomatis tiap 60 detik |
| `--mt5` | off | Connect MT5 (data live + auto-order) |
| `--real` | off | Mode akun real ~$60: auto-lot, multi-TP, trail, ML 70% |
| `--micro` | off | Mode mikro <1jt IDR: 1 order, lot min, filter ketat |
| `--lot` | 0.01 | Lot size per order (override auto-lot) |
| `--orders` | 15 | Jumlah order sekaligus |
| `--dca` | 0 | Jeda DCA dalam menit (0 = off) |
| `--trail` | 0 | Trailing stop dalam pips (0 = off) |
| `--multi-tp` | off | Partial close 50% di TP1 |
| `--risk` | 0 | Maks kerugian per order dalam USD (0 = ATR-based) |
| `--force-trade` | - | Paksa order: `BUY` atau `SELL` |
| `--backtest` | off | Jalankan simulasi historis |
| `--lstm` | off | Aktifkan LSTM (butuh TensorFlow) |
| `--no-ml` | off | Nonaktifkan ML |
| `--no-news` | off | Nonaktifkan news filter |
| `--tv` | off | Ambil data dari TradingView |
| `--mt4` | off | Aktifkan MT4 file bridge |
| `--mt-setup` | off | Generate EA file untuk MT4 |
| `--mt-login` | .env | Override nomor akun MT5 |
| `--mt-pass` | .env | Override password MT5 |
| `--mt-server` | .env | Override server broker MT5 |
| `--mt-status` | off | Tampilkan status akun MT5 |
| `--period` | auto | Override periode data: `3mo`, `1y`, dll |
| `--tune` | off | Tampilkan panduan tuning parameter |

---

## Tuning Parameter

Edit `config.py`:

```python
# ─── SINYAL ───────────────────────────────────────────────────────
MIN_SIGNAL_SCORE  = 5      # Threshold sinyal 0–10 (5=normal, 7=ketat)
ADX_TREND_MIN     = 25     # ADX minimal untuk entry (25=trending)
RSI_OVERBOUGHT    = 70     # RSI maks untuk BUY
RSI_OVERSOLD      = 30     # RSI min untuk SELL

# ─── SL / TP (mode normal) ────────────────────────────────────────
ATR_MULTIPLIER_SL = 1.0    # SL = 1x ATR
ATR_MULTIPLIER_TP = 10.0   # TP = 10x ATR (RR 1:10)
MIN_RR_RATIO      = 5.0    # Tolak sinyal jika RR < 5

# ─── REAL MODE ─────────────────────────────────────────────────────
REAL_ATR_SL       = 1.0    # SL = 1x ATR
REAL_ATR_TP       = 10.0   # TP = 10x ATR (RR 1:10)
REAL_TRAIL_PIPS   = 15.0   # Trailing stop 15 pips
REAL_AUTO_LOT     = True   # Hitung lot otomatis dari saldo
REAL_AUTO_LOT_MAX = 0.10   # Batas lot maksimal
REAL_ML_CONF      = 70     # ML confidence minimum 70%

# ─── BREAKEVEN ─────────────────────────────────────────────────────
BREAKEVEN_ENABLED = True
BREAKEVEN_TRIGGER = 1.0    # Geser SL saat profit = 1x jarak SL
BREAKEVEN_BUFFER  = 0.1    # Buffer di atas entry (pips)

# ─── MULTI-TP ──────────────────────────────────────────────────────
MULTI_TP_ENABLED  = False  # Aktifkan via --multi-tp atau --real
TP1_RATIO         = 0.5    # TP1 di 50% jarak ke TP penuh
TP1_CLOSE_PCT     = 50.0   # Tutup 50% volume di TP1

# ─── ML ─────────────────────────────────────────────────────────────
# CONFIDENCE_THRESHOLD diset di ml/model.py = 70.0
# Untuk >90% akurasi: 70% (coverage 55.6%) atau 85% (coverage 15.6%)

# ─── REFRESH ───────────────────────────────────────────────────────
REFRESH_INTERVAL  = 60     # Detik antar cycle
```

---

## Symbol yang Didukung

| Symbol | Deskripsi | Broker MT5 (contoh) |
|--------|-----------|---------------------|
| `XAUUSD` | Gold / US Dollar | XAUUSDm |
| `EURUSD` | Euro / US Dollar | EURUSDm |
| `GBPUSD` | British Pound / Dollar | GBPUSDm |
| `XAUEUR` | Gold / Euro | XAUEURm |
| `USDJPY` | US Dollar / Yen | USDJPYm |
| `BTCUSD` | Bitcoin / Dollar | BTCUSDm |
| `DXY` | US Dollar Index | USDXm |

> Nama symbol tiap broker berbeda. Sesuaikan `SYMBOL_MAP` di `broker/mt5_connector.py`.

---

## Disclaimer

> Bot ini dibuat untuk tujuan **edukasi dan riset**. Hasil backtest dan akurasi ML **tidak menjamin profit** di masa depan. Trading forex dan emas mengandung risiko tinggi kehilangan modal. Selalu gunakan **akun demo** sebelum live trading. Pengembang tidak bertanggung jawab atas kerugian finansial apapun.
