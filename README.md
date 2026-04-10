# Trader AI — Robot Trading GOLD (XAUUSD)

Robot trading otomatis berbasis **Python** yang menggabungkan **Machine Learning ensemble (LightGBM + XGBoost)**, **analisis teknikal 20+ indikator**, **Smart Money Concepts**, **analisis candle real-time**, dan eksekusi langsung ke **MetaTrader 5** — dilengkapi **REST API (FastAPI)**, **WebSocket live stream**, dan penyimpanan ke **PostgreSQL**.

---

## Daftar Isi

- [Stack Teknologi](#stack-teknologi)
- [Arsitektur Folder](#arsitektur-folder)
- [Alur Sistem](#alur-sistem)
- [API Endpoints](#api-endpoints)
- [Machine Learning](#machine-learning)
- [Analisis Candle Real-time](#analisis-candle-real-time)
- [Proteksi & Risk Management](#proteksi--risk-management)
- [Instalasi](#instalasi)
- [Konfigurasi `.env`](#konfigurasi-env)
- [Cara Menjalankan](#cara-menjalankan)
- [Semua Argumen CLI](#semua-argumen-cli)
- [Symbol yang Didukung](#symbol-yang-didukung)
- [Disclaimer](#disclaimer)

---

## Stack Teknologi

| Layer | Teknologi |
|-------|-----------|
| **REST API** | FastAPI + Uvicorn |
| **WebSocket** | FastAPI WebSocket — live stream ke frontend |
| **Database** | PostgreSQL + SQLAlchemy 2.0 async + asyncpg |
| **ML Ensemble** | LightGBM + XGBoost (VotingClassifier + Optuna tuning) |
| **Broker** | MetaTrader 5 API (data live + auto-order) |
| **Data fallback** | TradingView → Yahoo Finance |
| **Config** | Pydantic BaseSettings + `.env` + `config.py` |
| **Adaptive Learning** | AdaptiveLearner — belajar dari setiap trade |
| **Signal Logger** | JSONL log sinyal + analisis + outcome per trade |

---

## Arsitektur Folder

```
trader-ai/
│
├── app/                              ← SOURCE CODE UTAMA
│   ├── api/
│   │   ├── routes/                   ← Semua endpoint API
│   │   │   ├── signal.py             ← GET /signal
│   │   │   ├── trade.py              ← POST /trade/force
│   │   │   ├── positions.py          ← GET/DELETE /positions
│   │   │   ├── adaptive.py           ← GET /adaptive
│   │   │   ├── analysis.py           ← GET /analysis
│   │   │   ├── stats.py              ← GET /stats/daily, /stats/history
│   │   │   ├── bot_control.py        ← GET/POST /bot/status, pause, resume
│   │   │   ├── settings.py           ← GET/PUT /settings (TP/SL, lot, target)
│   │   │   ├── journal.py            ← GET /journal
│   │   │   ├── backtest.py           ← POST /backtest
│   │   │   ├── ws.py                 ← WebSocket /ws/live
│   │   │   └── ...
│   │   ├── schemas/                  ← Pydantic request/response models
│   │   └── deps.py                   ← Dependency injection
│   │
│   ├── core/
│   │   ├── config.py                 ← Settings dari .env (Pydantic)
│   │   ├── logging.py                ← Setup logger
│   │   ├── security.py               ← API key auth
│   │   └── paths.py                  ← Path constants (ROOT, DATA_DIR, dll)
│   │
│   ├── database/
│   │   ├── session.py                ← AsyncSessionLocal, create_tables
│   │   ├── models.py                 ← Tabel SQLAlchemy
│   │   └── crud/                     ← signals, trades, candle_logs, tx_log
│   │
│   ├── engine/
│   │   ├── bot.py                    ← TradingBot — loop analisis utama
│   │   ├── broker/
│   │   │   ├── mt5_connector.py      ← MT5 connect, place_order, SignalExecutor
│   │   │   └── mt4_bridge.py         ← File bridge untuk MT4
│   │   ├── signals/
│   │   │   ├── indicators.py         ← 20+ indikator teknikal
│   │   │   └── signals.py            ← Scoring & signal generation
│   │   └── backtest/
│   │       └── engine.py             ← Backtest simulator
│   │
│   ├── services/
│   │   ├── bot_service.py            ← Background thread + semua getter FE
│   │   ├── ai/
│   │   │   ├── ml/
│   │   │   │   ├── predictor.py      ← ScalpingPredictor (inference + filters)
│   │   │   │   ├── trainer.py        ← train_from_history()
│   │   │   │   └── features.py       ← Feature engineering
│   │   │   ├── models/               ← .joblib model files
│   │   │   ├── adaptive.py           ← AdaptiveLearner
│   │   │   └── signal_logger.py      ← Log sinyal + outcome ke JSONL
│   │   ├── news/
│   │   │   ├── news_filter.py        ← NewsFilter + lag effect
│   │   │   └── news_model.py         ← News sentiment
│   │   ├── tradingview/
│   │   │   └── tv_feed.py            ← TradingView data feed
│   │   ├── candle_db.py              ← Simpan candle ke CSV lokal
│   │   ├── candle_log.py             ← Log candle runtime
│   │   ├── journal.py                ← trade_journal.csv
│   │   └── session_bias.py           ← Session bias (Tokyo/London/NY)
│   │
│   └── utils/
│       ├── response.py               ← ok() / err() response helper
│       ├── formatters.py
│       └── helpers.py
│
├── data/                             ← FILE DATA (bukan kode)
│   ├── history/                      ← CSV candle historis (XAUUSD_5m.csv, dll)
│   ├── cache/                        ← Cache berita harian JSON
│   ├── journal/                      ← trade_journal.csv
│   └── session/                      ← session_bias_state.json
│
├── logs/                             ← Runtime logs
│   ├── trade_journal.csv             ← Semua trade (entry, exit, PnL)
│   ├── signal_analysis.jsonl         ← Log sinyal + analisis candle + outcome
│   └── api.log
│
├── ai/                               ← Stub → app/services/ai/
├── backend/                          ← Stub → app/engine/
├── api/, db/, core/, services/       ← Stub → app/... (backward compat)
│
├── config.py                         ← Parameter trading (TP/SL, lot, filter, dll)
├── main.py                           ← Entry point CLI / live bot
├── app/api/main.py                   ← Entry point FastAPI server
├── runtime_settings.json             ← Override config via API (auto-generated)
├── .env                              ← Kredensial MT5, DB, API key
├── requirements.txt
└── run.bat
```

---

## Alur Sistem

```
STARTUP
  create DB tables → connect MT5 → load ScalpML model → init bot → fetch news
                         ↓
        ┌────────────────┴───────────────┐
        │  API Mode (app.api.main)       │  CLI Mode (main.py)
        │  Background thread 60 detik    │  Loop 60 detik
        └────────────────┬───────────────┘
                         ↓
TIAP CYCLE:
  1. Fetch candle M5 dari MT5 (atau TradingView / Yahoo fallback)
  2. Hitung 20+ indikator → scoring rule-based
  3. ScalpingPredictor (ML):
       - Hard Filter: Trend EMA50, Circuit Breaker, Consecutive Loss Block
       - ML inference: LightGBM + XGBoost ensemble → prob BUY / SELL
       - Analisis candle: momentum 8c, body shape, posisi range, ATR rank
       - Swing-based SL/TP: cari swing low/high terdekat
       - TP reachability score (0-4)
  4. Pre-trade check (tepat sebelum order):
       - EMA trend, momentum 5c, posisi range, rejection wick, candle 3c
       - Verdict: GO / CAUTION / SKIP
  5. Eksekusi ke MT5 (jika sinyal = BUY/SELL dan semua guard pass)
  6. Log sinyal ke signal_analysis.jsonl (disimpan + di-update saat tutup)
  7. Sync closed positions → update journal + adaptive learner
  8. Adaptive: record outcome, update indicator hit rate, cek retrain trigger
```

---

## API Endpoints

### REST

| Method | Endpoint | Keterangan |
|--------|----------|------------|
| GET | `/signal` | Sinyal rule-based + ScalpML terbaru |
| GET | `/bot/status` | Status bot + akun MT5 + circuit breaker + P&L |
| GET | `/bot/scalp` | ScalpML signal + analisis candle |
| POST | `/bot/pause` | Pause bot (tidak ada trade baru) |
| POST | `/bot/resume` | Resume bot |
| POST | `/bot/analyze` | Trigger analisis manual sekarang |
| GET | `/positions` | Posisi terbuka + balance |
| DELETE | `/positions/{ticket}` | Tutup posisi by ticket |
| DELETE | `/positions` | Tutup semua posisi |
| GET | `/adaptive` | Adaptive learning stats (WR, mode, sumber) |
| GET | `/adaptive/circuit` | Status circuit breaker & consecutive loss |
| GET | `/analysis` | Log sinyal + analisis candle + outcome |
| GET | `/analysis/summary` | Akurasi per tp_score & alignment |
| GET | `/stats/daily` | Statistik trading hari ini |
| GET | `/stats/history?days=7` | Statistik per hari |
| GET | `/journal` | Trade journal |
| POST | `/trade/force` | Force buka order manual |
| GET | `/settings` | Baca semua setting aktif |
| PUT | `/settings/tp-sl` | Update SL/TP live |
| PUT | `/settings/daily-target` | Update target & limit harian |
| PUT | `/settings/lot` | Update lot settings |
| PUT | `/settings/filters` | Update filter sinyal |
| DELETE | `/settings/reset` | Reset ke default config.py |

### WebSocket

```
ws://localhost:8000/ws/live
```

Update setiap 5 detik:

```json
{
  "type": "update",
  "price": 3285.5,
  "signal": { "direction": "BUY", "score": 6.5, "sl": 3279.5, "tp": 3297.5 },
  "scalp": {
    "direction": "BUY",
    "prob_buy": 0.742,
    "confidence": "HIGH",
    "notes": ["Momentum 8c: 6↑/2↓ (NAIK)", "Alignment: ML BUY ✓ sejalan candle NAIK"],
    "warnings": []
  },
  "positions": [...],
  "account": { "balance": 120.5, "equity": 132.3, "profit": 11.8 },
  "daily": { "win": 4, "loss": 2, "net_pnl": 18.5 },
  "circuit": { "circuit_breaker_active": false, "consec_loss_buy": 1 }
}
```

---

## Machine Learning

### Model (ScalpingPredictor)

```
VotingClassifier (soft voting)
├── LightGBM  (tuned via Optuna 20 trials)
└── XGBoost   (tuned via Optuna 20 trials)

Feature selection : SelectKBest (mutual_info_classif, k=30–95)
Scaler           : StandardScaler
Validation       : Walk-forward (TimeSeriesSplit 5-fold)
Training data    : 72.000+ candle M5 XAUUSD (Apr 2025 – Apr 2026)
```

### Hasil Training Terakhir

| Metrik | Nilai |
|--------|-------|
| AUC Test | 0.723 |
| WF-AUC (walk-forward) | 0.715 |
| Accuracy | 63.8% |
| Prob threshold | 0.62 |
| Training candles | 10.178 valid rows |

### Labeling

```
TP = 12pt, SL = 6pt, lookahead = 12 candle
- High melewati TP lebih dulu → label BUY (1)
- Low melewati SL lebih dulu  → label SELL (0)
- Keduanya / tidak ada        → buang (ambiguous)
Crash candles (ATR > 2× median) → sample_weight = 3.0
```

---

## Analisis Candle Real-time

Setiap sinyal dilengkapi analisis candle yang ditampilkan di output dan disimpan ke log:

| Analisis | Keterangan |
|----------|------------|
| **Momentum 8c** | Berapa candle naik/turun dari 8 candle terakhir |
| **Momentum 3c** | Arah 3 candle terbaru |
| **Candle terakhir** | Warna, kekuatan body, shape (Engulf/Doji/Normal) |
| **Posisi range** | Harga di atas/tengah/bawah range 20 candle |
| **ATR rank** | Volatilitas relatif (persentil ATR 50 candle) |
| **Alignment** | Apakah ML searah dengan candle momentum |
| **TP reachability** | Score 0-4: apakah TP mungkin tercapai |
| **SL/TP swing** | Level SL di bawah swing low, TP di swing high terdekat |
| **Consecutive loss** | Warning jika 2-5x loss berturut-turut arah sama |

Semua data ini disimpan ke `logs/signal_analysis.jsonl` dan di-update dengan outcome (WIN/LOSS) saat trade tutup — bisa diolah untuk evaluasi dan training ulang.

---

## Proteksi & Risk Management

| Proteksi | Keterangan |
|----------|------------|
| **Trend Filter (EMA50)** | Blok BUY jika harga >0.15% di bawah EMA50, blok SELL jika di atas |
| **Consecutive Loss Block** | Blok arah setelah 5x loss berturut-turut |
| **Circuit Breaker** | Stop trading 2 jam setelah 5 loss dalam satu hari |
| **ATR Spike Filter** | Skip trading saat ATR > 2.5× median (news/crash) |
| **Pre-trade Check** | Cek EMA trend, momentum, wick rejection, posisi range sebelum order |
| **RR Guard** | Tolak sinyal jika RR < 1.2 |
| **Spread Filter** | Blok order jika spread > 1.5 pip |
| **Max Trades/Hour** | Maksimal 2 trade per jam |
| **Daily Profit Target** | Bot berhenti saat target % tercapai |
| **Daily Loss Limit** | Bot berhenti saat limit rugi % tercapai |
| **Lot Cap** | `MAX_LOT_SAFE` saat normal, `MAX_LOT_LOSING` saat WR < 35% |
| **Adaptive Mode** | CONSERVATIVE/NORMAL/AGGRESSIVE berdasarkan WR 10 trade terakhir |

---

## Instalasi

```bash
# 1. Clone
git clone <repo-url> trader-ai
cd trader-ai

# 2. Buat virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. Install dependensi
pip install -r requirements.txt

# 4. Install MetaTrader5 (Windows only)
pip install MetaTrader5

# 5. Setup PostgreSQL (opsional — untuk API mode)
# CREATE DATABASE forex_trading;

# 6. Buat .env
copy .env.example .env
# Edit .env — isi kredensial MT5 dan PostgreSQL
```

---

## Konfigurasi `.env`

```env
# MetaTrader 5
MT5_LOGIN=12345678
MT5_PASSWORD=password_kamu
MT5_SERVER=Exness-MT5Trial14

# PostgreSQL (untuk API mode)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=forex_trading
DB_USER=postgres
DB_PASSWORD=password_db

# FastAPI
APP_NAME=Trader AI API
APP_VERSION=1.0.0
DEBUG=false
API_KEY=              # kosong = tanpa auth

# Bot
BOT_SYMBOL=XAUUSD
BOT_TIMEFRAME=5m
BOT_USE_MT5=true
BOT_REAL_MODE=true    # true = akun real, false = demo
```

---

## Cara Menjalankan

### CLI — Bot Live Trading

```bash
# Demo mode (paper trading)
python main.py --live --mt5

# Real mode (akun real MT5)
python main.py --live --mt5 --real

# Dengan data TradingView
python main.py --live --mt5 --tv

# Tanpa ML (hanya rule-based)
python main.py --live --mt5 --no-ml
```

### API Server (FastAPI)

```bash
# Jalankan API + bot di background
python -m app.api.main

# Production
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Docs tersedia di:
# http://localhost:8000/docs
# http://localhost:8000/redoc
```

### Backtest

```bash
python main.py --backtest --symbol XAUUSD --timeframe 5m
```

### Retrain ML Model

```bash
python -m app.services.ai.ml.predictor --retrain
```

---

## Semua Argumen CLI

| Argumen | Keterangan |
|---------|------------|
| `--live` | Mode live (loop terus) |
| `--mt5` | Hubungkan ke MetaTrader 5 |
| `--real` | Aktifkan real mode (lot, filter, target akun real) |
| `--micro` | Micro mode — filter ketat, 1 order, lot min |
| `--backtest` | Mode backtest historis |
| `--symbol` | Symbol (default: XAUUSD) |
| `--timeframe` | Timeframe (default: 5m) |
| `--tv` | Gunakan TradingView sebagai sumber data |
| `--no-ml` | Nonaktifkan ML prediction |
| `--no-news` | Nonaktifkan news filter |
| `--lstm` | Aktifkan LSTM (butuh TensorFlow) |

---

## Update Settings via API (Tanpa Restart)

```bash
# Ubah TP/SL
curl -X PUT http://localhost:8000/settings/tp-sl \
  -H "Content-Type: application/json" \
  -d '{"fixed_sl_pips": 5.0, "fixed_tp_pips": 8.0}'

# Set target harian $10, stop rugi $5
curl -X PUT http://localhost:8000/settings/daily-target \
  -d '{"daily_profit_target_usd": 10.0, "daily_loss_limit_usd": 5.0}'

# Pause bot
curl -X POST http://localhost:8000/bot/pause

# Resume bot
curl -X POST http://localhost:8000/bot/resume
```

Semua perubahan disimpan ke `runtime_settings.json` dan aktif kembali setelah restart.

---

## Symbol yang Didukung

| Symbol | Keterangan |
|--------|------------|
| `XAUUSD` | Gold / USD ← **utama** |
| `EURUSD` | Euro / USD |
| `GBPUSD` | GBP / USD |
| `USDJPY` | USD / JPY |
| `XAUEUR` | Gold / EUR |
| `BTCUSD` | Bitcoin / USD |
| `DXY` | US Dollar Index |

---

## Disclaimer

> Robot trading ini dibuat untuk tujuan edukasi dan pengembangan. Trading forex dan emas mengandung risiko tinggi termasuk kehilangan modal. Hasil backtest tidak menjamin performa live. Gunakan dengan bijak dan selalu gunakan risk management yang ketat.
