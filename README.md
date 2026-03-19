# Trader AI — Robot Trading GOLD & EUR/USD

Robot trading otomatis berbasis **Python** yang menggabungkan **Machine Learning**, **analisis teknikal multi-indikator**, **prediksi arah dari berita (News Lag Effect)**, dan integrasi langsung ke **MetaTrader 5**.

---

## Daftar Isi

- [Fitur](#fitur)
- [Arsitektur Folder](#arsitektur-folder)
- [Cara Kerja](#cara-kerja)
- [Instalasi](#instalasi)
- [Konfigurasi MT5](#konfigurasi-mt5)
- [Cara Penggunaan](#cara-penggunaan)
- [Semua Argumen CLI](#semua-argumen-cli)
- [Tuning Parameter](#tuning-parameter)
- [Disclaimer](#disclaimer)

---

## Fitur

| Fitur | Detail |
|-------|--------|
| **Data Live MT5** | Candle real-time langsung dari broker (via MT5 API) |
| **Fallback Yahoo Finance** | Auto-fallback jika MT5 tidak connect |
| **Database Candle** | Semua candle disimpan lokal — makin lama makin akurat |
| **8 Indikator Teknikal** | RSI, MACD, EMA (9/21/50/200), Bollinger Bands, Stochastic, ADX, ATR |
| **Pola Candle** | 10+ pola: Hammer, Engulfing, Doji, Shooting Star, dll |
| **Rule-Based Signal** | Scoring engine −10 s/d +10, arah BUY/SELL/WAIT |
| **ML Predictor** | Ensemble (RF + GB + ET) dengan feature selection otomatis |
| **LSTM Deep Learning** | Bidirectional LSTM + Conv1D (opsional, butuh TensorFlow) |
| **News Lag Effect** | Berita hari-hari lalu masih pengaruhi sinyal (decay otomatis) |
| **Prediksi Arah dari Berita** | Keyword scoring BULLISH/BEARISH per kategori berita |
| **Smart TP/SL** | TP dari Swing High/Low terdekat atau BB; SL sekecil mungkin (RR optimal) |
| **Auto Execute MT5** | Buka/tutup order otomatis dengan lot, SL, TP |
| **DCA Otomatis** | Pasang order tambahan tiap X menit selama sinyal masih searah |
| **Sync Posisi Manual** | Pasang SL/TP ke posisi manual yang belum punya |
| **Trailing Stop** | Otomatis geser SL mengikuti harga |
| **Backtest Engine** | Simulasi historis lengkap: PnL, win rate, max drawdown |
| **MT4 Bridge** | File bridge + Expert Advisor untuk MT4 |

---

## Arsitektur Folder

```
trader-ai/
│
├── main.py              # Entry point — jalankan dari sini
├── bot.py               # Orchestrator utama (load data, train, analyze)
├── config.py            # Semua parameter tuning
├── test_order.py        # Utility test order ke MT5
├── requirements.txt     # Dependensi Python
├── .env                 # Kredensial MT5 (tidak di-commit)
├── .env.example         # Template .env
│
├── analysis/            # Engine analisis teknikal
│   ├── indicators.py    # Hitung semua indikator + pola candle
│   └── signals.py       # Scoring engine, Smart TP/SL, generate_signal()
│
├── ml/                  # Machine Learning
│   ├── model.py         # Ensemble ML (Random Forest + GB + Extra Trees)
│   └── deep_model.py    # LSTM Bidirectional + Conv1D (TensorFlow)
│
├── data/                # Data & berita
│   ├── candle_db.py     # Simpan & gabung candle historis ke CSV
│   ├── news_filter.py   # Berita + kalender ekonomi + News Lag Effect
│   ├── history/         # Database candle CSV per symbol/timeframe
│   └── news_cache/      # Cache berita harian per tanggal (JSON)
│
├── broker/              # Integrasi MetaTrader
│   ├── mt5_connector.py # MT5 API: connect, order, close, DCA, trailing, status
│   ├── mt4_bridge.py    # MT4 file bridge + generator EA
│   └── EA_TraderAI.mq4  # Expert Advisor untuk MT4
│
├── backtest/            # Backtesting
│   └── engine.py        # Simulasi historis, laporan PnL & drawdown
│
├── models/              # Model ML tersimpan (auto-generated)
└── logs/                # Log eksekusi (auto-generated)
```

---

## Cara Kerja

```
┌─────────────────────────────────────────────────────┐
│                   SETIAP CYCLE                      │
│                                                     │
│  1. FETCH DATA                                      │
│     MT5 live (1500 candle) → fallback Yahoo Finance │
│     Simpan candle baru ke database lokal            │
│                                                     │
│  2. TRAIN ML                                        │
│     Pakai semua candle historis dari DB             │
│     Ensemble: Random Forest + GB + Extra Trees      │
│                                                     │
│  3. FETCH NEWS (1x per hari, cached)                │
│     Yahoo Finance + ForexFactory Calendar           │
│     Hitung News Lag Effect (berita 5 hari lalu)     │
│                                                     │
│  4. ANALISIS                                        │
│     Rule-based score (indikator teknikal)           │
│     + News Bias score (dari berita + kalender)      │
│     + ML prediction (confidence ≥ 65%)             │
│                                                     │
│  5. SMART TP/SL                                     │
│     TP = Swing High/Low terdekat atau BB            │
│     SL = bawah/atas candle terakhir (sekecil mungkin│
│                                                     │
│  6. EKSEKUSI ke MT5                                 │
│     Cek filter → place_order() → manage_positions() │
└─────────────────────────────────────────────────────┘
```

### Logika Eksekusi Order

```
Rule-based score ≥ threshold  →  BUY/SELL langsung
           ↓ tidak
ML confidence ≥ 65%           →  ikut arah ML
           ↓ tidak
WAIT — tidak masuk trade

Filter sebelum eksekusi:
  × News HIGH + berlawanan sinyal  →  dibatalkan
  × ML berlawanan (confident)      →  di-skip
  × DCA: belum cukup waktu jeda    →  tunggu dulu
```

### News Lag Effect

Berita tidak langsung hilang dampaknya. Sistem menyimpan berita tiap hari dan menghitung efek berdasarkan usia:

| Kategori Berita | Durasi Efek | Contoh |
|----------------|-------------|--------|
| Perang / invasi / nuklir | 5 hari | Iran-Israel conflict |
| Krisis / sanksi / geopolitik | 4 hari | Banking crisis |
| Rate cut/hike / hawkish | 2 hari | Fed rate decision |
| NFP / CPI / data ekonomi | 1–2 hari | Non-Farm Payrolls |
| Dollar naik/turun harian | 1 hari | DXY drops |

Semakin lama beritanya, semakin kecil bobotnya (linear decay 1.0 → 0.1).

Berita hari ini yang HIGH impact **tidak memblok order** selama arah bias berita **searah** dengan sinyal teknikal.

### DCA (Dollar Cost Averaging)

Dengan `--dca X`, bot akan terus menambah order tiap X menit selama sinyal masih searah:

```
Menit 0   → sinyal BUY → order #1 masuk
Menit 15  → sinyal BUY → order #2 masuk
Menit 30  → sinyal BUY → order #3 masuk
Menit 45  → sinyal WAIT → tidak masuk
Menit 60  → sinyal SELL → posisi BUY ditutup semua
```

---

## Instalasi

### 1. Clone / download project

```bash
git clone <repo-url> trader-ai
cd trader-ai
```

### 2. Install dependensi wajib

```bash
pip install -r requirements.txt
```

### 3. Install MetaTrader5 (Windows, untuk auto-trade)

```bash
pip install MetaTrader5
```

### 4. Install TensorFlow (opsional, untuk LSTM)

```bash
pip install tensorflow
```

### 5. Buat file `.env`

```bash
cp .env.example .env
# Edit .env, isi kredensial MT5
```

---

## Konfigurasi MT5

### Lewat file `.env` (direkomendasikan)

```env
MT5_LOGIN=12345678
MT5_PASSWORD=password_kamu
MT5_SERVER=Exness-MT5Trial14
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

### Sesuaikan nama symbol broker

Buka `broker/mt5_connector.py`, edit `SYMBOL_MAP`:

```python
SYMBOL_MAP = {
    "GOLD":   "XAUUSDm",   # sesuaikan dengan nama symbol di broker
    "EURUSD": "EURUSDm",
}
```

### Aktifkan Auto Trading di MT5

Wajib: klik tombol **Algo Trading** di toolbar MT5 hingga **hijau (ON)**.

---

## Cara Penggunaan

### Analisis sekali jalan

```bash
python main.py --symbol GOLD --tf 5m
python main.py --symbol EURUSD --tf 1h
```

### Live mode (auto refresh tiap 60 detik)

```bash
python main.py --symbol GOLD --tf 5m --live
```

### Live + data real-time dari MT5

```bash
python main.py --symbol GOLD --tf 5m --mt5 --live
```

### Live + auto execute order ke MT5

```bash
python main.py --symbol GOLD --tf 5m --mt5 --live
```

> MT5 harus terbuka, Algo Trading harus ON (hijau).

### DCA — tambah order tiap X menit selama sinyal sama

```bash
# Tiap 15 menit pasang order baru jika sinyal masih BUY/SELL
python main.py --symbol GOLD --tf 5m --mt5 --live --dca 15

# DCA tiap 30 menit
python main.py --symbol GOLD --tf 5m --mt5 --live --dca 30
```

### Trailing stop

```bash
# SL otomatis geser mengikuti harga, jarak 20 pips
python main.py --symbol GOLD --tf 5m --mt5 --live --trail 20
```

### Test koneksi order ke MT5

```bash
python main.py --symbol GOLD --mt5 --force-trade BUY
python main.py --symbol GOLD --mt5 --force-trade SELL
```

### Backtest

```bash
python main.py --symbol GOLD --tf 1h --backtest
python main.py --symbol EURUSD --tf 1d --backtest
```

### Dengan LSTM (butuh TensorFlow)

```bash
python main.py --symbol GOLD --tf 1h --mt5 --live --lstm
```

### Cek status akun MT5

```bash
python main.py --mt5 --mt-status
```

---

## Semua Argumen CLI

| Argumen | Default | Keterangan |
|---------|---------|-----------|
| `--symbol` | EURUSD | Pair: `EURUSD`, `GOLD` |
| `--tf` | 1h | Timeframe: `1m` `5m` `15m` `1h` `4h` `1d` |
| `--live` | off | Refresh otomatis tiap 60 detik |
| `--mt5` | off | Connect ke MT5 (data live + auto order) |
| `--mt4` | off | Aktifkan MT4 file bridge |
| `--lstm` | off | Aktifkan model LSTM (butuh TensorFlow) |
| `--no-ml` | off | Nonaktifkan ML, pakai rule-based saja |
| `--no-news` | off | Nonaktifkan news filter |
| `--backtest` | off | Jalankan simulasi historis |
| `--force-trade` | - | Paksa order tanpa filter: `BUY` atau `SELL` |
| `--dca` | 0 | Jeda antar order DCA dalam menit (0 = off) |
| `--trail` | 0 | Trailing stop dalam pips (0 = off) |
| `--mt-login` | .env | Override nomor akun MT5 |
| `--mt-pass` | .env | Override password MT5 |
| `--mt-server` | .env | Override server broker MT5 |
| `--mt-status` | off | Tampilkan status akun MT5 saja |
| `--mt-setup` | off | Generate EA file untuk MT4 |
| `--period` | auto | Override periode data: `3mo`, `1y`, dll |
| `--tune` | off | Tampilkan panduan tuning parameter |

---

## Tuning Parameter

Edit `config.py`:

```python
# Threshold sinyal (0-10) — turunkan = lebih banyak sinyal
MIN_SIGNAL_SCORE = 5

# Bobot indikator
WEIGHTS = {
    "rsi":       2.0,
    "macd":      2.0,
    "ema_cross": 2.0,
    "stoch":     1.5,
    "bb":        1.0,
    "adx":       1.0,
    "candle":    0.5,
}

# Smart TP/SL
SL_BUFFER_MULT  = 0.3   # Buffer SL di luar candle (× ATR)
TP_MIN_SL_RATIO = 1.5   # TP minimal 1.5× jarak SL

# Refresh interval live mode (detik)
REFRESH_INTERVAL = 60
```

Lot size dan risk diatur di `broker/mt5_connector.py`:

```python
DEFAULT_LOT  = 0.01   # Ukuran lot per order
MAX_LOT      = 1.0    # Batas atas lot dari kalkulasi risiko
RISK_PERCENT = 1.0    # % balance yang dirisiko per trade
```

---

## Disclaimer

> Bot ini dibuat untuk tujuan **edukasi dan riset**. Hasil backtest dan akurasi ML tidak menjamin profit di masa depan. Trading forex dan emas mengandung risiko tinggi kehilangan modal. Selalu gunakan **akun demo** sebelum live trading. Pengembang tidak bertanggung jawab atas kerugian finansial apapun.
