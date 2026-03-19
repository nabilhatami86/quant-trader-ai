# Trader AI — Robot Trading GOLD & Multi-Pair

Robot trading otomatis berbasis **Python** yang menggabungkan **Machine Learning**, **analisis teknikal**, **prediksi arah dari berita (News Lag Effect)**, dan integrasi langsung ke **MetaTrader 5** serta **TradingView**.

---

## Daftar Isi

- [Fitur](#fitur)
- [Arsitektur Folder](#arsitektur-folder)
- [Cara Kerja](#cara-kerja)
- [Instalasi](#instalasi)
- [Konfigurasi MT5](#konfigurasi-mt5)
- [Cara Penggunaan](#cara-penggunaan)
- [Semua Argumen CLI](#semua-argumen-cli)
- [Symbol yang Didukung](#symbol-yang-didukung)
- [Tuning Parameter](#tuning-parameter)
- [Disclaimer](#disclaimer)

---

## Fitur

| Fitur | Detail |
|-------|--------|
| **Data Live MT5** | Candle real-time langsung dari broker via MetaTrader 5 API |
| **Data TradingView** | Fetch candle via `tvdatafeed` (opsional, tanpa akun premium) |
| **Fallback Yahoo Finance** | Auto-fallback jika MT5/TV tidak connect |
| **Database Candle** | Semua candle disimpan lokal — makin lama makin akurat |
| **Candle Log** | Log pergerakan candle per cycle ke CSV (OHLC, arah, pola, RSI, EMA) |
| **8 Indikator Teknikal** | RSI(14), MACD, EMA(9/20/50/200), Bollinger Bands, Stochastic, ADX, ATR |
| **Pola Candle** | 10+ pola: Hammer, Engulfing, Doji, Shooting Star, dll |
| **Decision Logic** | EMA20 > EMA50 + RSI < 70 + ADX trending + candle trend = BUY |
| **Candle Trend Filter** | Cek 5 candle terakhir — hindari entry saat candle berlawanan arah |
| **ML Predictor** | Ensemble (RF + GB + ET) dengan feature selection otomatis |
| **LSTM Deep Learning** | Bidirectional LSTM + Conv1D (opsional, butuh TensorFlow) |
| **News Lag Effect** | Berita hari-hari lalu masih pengaruhi sinyal (decay otomatis) |
| **Prediksi Arah Berita** | Keyword scoring BULLISH/BEARISH — berita = bias harian tetap |
| **Smart TP/SL** | SL = 1.5x ATR, TP = 2x ATR atau SwingHigh/Low jika lebih jauh |
| **Breakeven Otomatis** | SL digeser ke entry saat profit = jarak SL (worst case impas) |
| **SL Risk Guard** | Warning saat harga 30% dari SL, auto-close saat 10% dari SL |
| **Bulk Order** | Pasang N order sekaligus saat sinyal masuk (default: 15 × 0.01 lot) |
| **DCA Otomatis** | Tambah order tiap X menit selama sinyal masih searah |
| **Lot Fleksibel** | Set lot per order via CLI (`--lot 0.5`) atau auto dari risk % |
| **Trailing Stop** | SL otomatis geser mengikuti harga |
| **Multi-TP (Partial Close)** | Tutup 50% posisi di TP1 → profit terkunci, sisa 50% jalan ke TP2 bebas risiko |
| **Mode Mikro** | Khusus akun <1 juta IDR: 1 order, lot minimum, filter ekstra ketat (`--micro`) |
| **Trade Journal** | Catat semua entry/exit, hitung win rate & total P&L |
| **Multi Symbol** | XAUUSD, XAUEUR, EURUSD, GBPUSD, USDJPY, DXY, BTCUSD |
| **Backtest Engine** | Simulasi historis: PnL, win rate, max drawdown |
| **MT4 Bridge** | File bridge + Expert Advisor untuk MetaTrader 4 |

---

## Arsitektur Folder

```
trader-ai/
│
├── main.py              # Entry point — jalankan dari sini
├── bot.py               # Orchestrator (load data, train, analyze)
├── config.py            # Semua parameter tuning
├── test_order.py        # Utility test order ke MT5
├── requirements.txt     # Dependensi Python
├── .env                 # Kredensial MT5 (tidak di-commit)
├── .env.example         # Template .env
│
├── analysis/            # Engine analisis teknikal
│   ├── indicators.py    # Hitung semua indikator + pola candle
│   └── signals.py       # Decision logic, Smart TP/SL, generate_signal()
│
├── ml/                  # Machine Learning
│   ├── model.py         # Ensemble ML (Random Forest + GB + Extra Trees)
│   └── deep_model.py    # LSTM Bidirectional + Conv1D (TensorFlow)
│
├── data/                # Data & berita
│   ├── candle_db.py     # Simpan & gabung candle historis ke CSV
│   ├── candle_log.py    # Log pergerakan candle tiap cycle
│   ├── trade_journal.py # Catat entry/exit trade, hitung win rate
│   ├── news_filter.py   # Berita + kalender ekonomi + News Lag Effect
│   ├── tv_feed.py       # Fetch data dari TradingView (tvdatafeed)
│   ├── history/         # Database candle CSV per symbol/timeframe
│   └── news_cache/      # Cache berita harian per tanggal (JSON)
│
├── broker/              # Integrasi MetaTrader
│   ├── mt5_connector.py # MT5: connect, order, bulk, DCA, partial close, breakeven, SL guard
│   ├── mt4_bridge.py    # MT4 file bridge + generator EA
│   └── EA_TraderAI.mq4  # Expert Advisor untuk MT4
│
├── backtest/            # Backtesting
│   └── engine.py        # Simulasi historis, laporan PnL & drawdown
│
├── logs/                # Log otomatis (auto-generated)
│   ├── candles_XAUUSD_5m.csv   # Log pergerakan candle
│   └── journal_XAUUSD_5m.csv   # Trade journal (entry/exit/hasil)
│
└── models/              # Model ML tersimpan (auto-generated)
```

---

## Cara Kerja

```
┌─────────────────────────────────────────────────────────┐
│                    STARTUP (1x)                         │
│  Fetch news → simpan bias berita harian                 │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  TIAP CYCLE (60 detik)                  │
│                                                         │
│  1. FETCH DATA                                          │
│     MT5 live → TradingView → Yahoo Finance (fallback)   │
│     Simpan candle baru ke database lokal                │
│                                                         │
│  2. TRAIN ML                                            │
│     Ensemble: Random Forest + GB + Extra Trees          │
│     Data: semua candle historis dari DB                 │
│                                                         │
│  3. ANALISIS                                            │
│     EMA20 vs EMA50 → arah utama                        │
│     RSI(14) → filter overbought/oversold                │
│     ADX(14) → filter sideways (hanya masuk saat trend)  │
│     Candle trend → 5 candle terakhir harus searah       │
│     News bias → konteks bullish/bearish harian          │
│     ML prediction → konfirmasi (confidence ≥ 65%)      │
│                                                         │
│  4. SMART TP/SL                                         │
│     SL = 1.5x ATR dari entry                           │
│     TP = 2x ATR atau SwingHigh/Low (mana lebih jauh)    │
│                                                         │
│  5. EKSEKUSI ke MT5                                     │
│     Bulk order (default 15 × 0.01 lot)                 │
│     Multi-TP: partial close 50% di TP1 (jika aktif)    │
│     Breakeven otomatis saat profit = jarak SL           │
│     SL Guard: warning 30%, auto-close 10% sebelum SL   │
│     DCA: tambah order tiap X menit (jika aktif)         │
│                                                         │
│  6. LOG                                                 │
│     Candle log → logs/candles_SYMBOL_TF.csv             │
│     Trade journal → logs/journal_SYMBOL_TF.csv          │
└─────────────────────────────────────────────────────────┘
```

### Logika Keputusan

```
BUY  : EMA20 > EMA50  AND  RSI < 70  AND  ADX >= 25
       AND  candle terakhir tidak mayoritas BEARISH

SELL : EMA20 < EMA50  AND  RSI > 30  AND  ADX >= 25
       AND  candle terakhir tidak mayoritas BULLISH

WAIT : kondisi tidak terpenuhi (pasar sideways atau konflik sinyal)

Override news:
  × News HIGH + berlawanan sinyal → WAIT
  ✓ News HIGH + searah sinyal     → lanjut (dengan catatan)
```

### SL Risk Guard

```
Entry BUY @ 4850  SL @ 4835  (jarak = 15 poin)

Harga 4839.5  → sisa 4.5 poin (30%) → [SL-BAHAYA] ⚠ WARNING
Harga 4836.5  → sisa 1.5 poin (10%) → [SL-KRITIS] → AUTO CLOSE
```

### Breakeven Otomatis

```
Entry BUY @ 4850  SL @ 4835  (jarak = 15 poin)

Harga naik ke 4865 (profit 15 poin = jarak SL)
→ SL otomatis digeser ke 4850.x (entry + buffer kecil)
→ Worst case: impas, bukan rugi
```

### Multi-TP — Profit Dikunci Bertahap

Aktifkan dengan `--multi-tp`. Mencegah profit habis saat harga berbalik sebelum TP penuh.

```
Entry BUY @ 2500.00   SL @ 2495.00   TP @ 2510.00
                               │
           TP1 @ 2505.00 ──────┤  (50% jarak ke TP)
                               │
    Harga menyentuh TP1:
    → Tutup 50% posisi → profit $X terkunci
    → SL digeser ke 2500.xx (breakeven)
    → Sisa 50% posisi BEBAS RISIKO
                               │
           TP2 @ 2510.00 ──────┘  (TP penuh seperti biasa)
    → Tutup sisa 50% posisi
```

**Output saat TP1 kena:**
```
[TP1] #12345 BUY — harga 2505.00 mencapai TP1 2505.00
→ Partial close 50% posisi (P&L sementara: +$25.00)
✓ SL digeser ke breakeven 2500.003 — sisa 0.005 lot bebas risiko
```

### Mode Mikro — Akun < 1 Juta IDR

Aktifkan dengan `--micro`. Semua parameter diperketat otomatis untuk melindungi modal kecil.

| Pengaturan | Normal | `--micro` |
|---|---|---|
| Jumlah order | 15 sekaligus | **1 saja** |
| Lot per order | 0.01 | **0.01 (minimum)** |
| DCA | boleh aktif | **OFF (paksa)** |
| Min skor sinyal | 5/10 | **7/10** |
| ADX minimum | 25 | **30** |
| ML harus setuju | tidak wajib | **wajib** |
| ML confidence minimum | 65% | **75%** |
| Boleh tumpuk posisi | ya | **tidak — tunggu tutup dulu** |

```bash
# Micro mode saja
python main.py --symbol XAUUSD --tf 1h --mt5 --live --micro

# Micro + kunci profit bertahap (kombinasi terbaik untuk modal kecil)
python main.py --symbol XAUUSD --tf 1h --mt5 --live --micro --multi-tp
```

### News Lag Effect

Berita tidak langsung hilang. Sistem simpan per hari dan hitung efek berdasarkan usia:

| Kategori | Durasi Efek | Contoh |
|----------|-------------|--------|
| Perang / invasi | 5 hari | Iran-Israel conflict |
| Krisis geopolitik | 4 hari | Banking crisis |
| Rate cut/hike | 2 hari | Fed rate decision |
| Data ekonomi (NFP/CPI) | 1–2 hari | Non-Farm Payrolls |
| Dollar harian | 1 hari | DXY drops |

News diambil **sekali per hari** (cache harian), bukan tiap cycle. Refresh otomatis saat ganti hari.

---

## Instalasi

```bash
# 1. Clone project
git clone <repo-url> trader-ai
cd trader-ai

# 2. Install dependensi wajib
pip install -r requirements.txt

# 3. Install MetaTrader5 (Windows, untuk auto-trade)
pip install MetaTrader5

# 4. Install TradingView feed (opsional)
pip install tvdatafeed

# 5. Install TensorFlow (opsional, untuk LSTM)
pip install tensorflow

# 6. Buat file .env
cp .env.example .env
# Edit .env, isi kredensial MT5
```

---

## Konfigurasi MT5

### File `.env`

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
    "XAUUSD": "XAUUSDm",   # sesuaikan nama di broker kamu
    "EURUSD": "EURUSDm",
    "XAUEUR": "XAUEURm",
}
```

### Aktifkan Auto Trading di MT5

Klik tombol **Algo Trading** di toolbar MT5 hingga **hijau (ON)**.

---

## Cara Penggunaan

### Default (XAUUSD, 5m, MT5, live)

```bash
python main.py --symbol XAUUSD --tf 5m --mt5 --live
```

Ini sudah otomatis: **15 order × 0.01 lot** saat sinyal masuk.

### Pilih data source

```bash
# Data dari MT5 (paling akurat, harga broker)
python main.py --symbol XAUUSD --tf 5m --mt5 --live

# Data dari TradingView
python main.py --symbol XAUUSD --tf 5m --tv --live

# Data dari TradingView + execute ke MT5
python main.py --symbol XAUUSD --tf 5m --tv --mt5 --live
```

### Set lot & jumlah order

```bash
# Default: 15 order × 0.01 lot = 0.15 lot total
python main.py --symbol XAUUSD --mt5 --live

# Custom: 10 order × 0.05 lot = 0.5 lot total
python main.py --symbol XAUUSD --mt5 --live --orders 10 --lot 0.05

# 1 order besar
python main.py --symbol XAUUSD --mt5 --live --orders 1 --lot 1.0
```

### DCA (tambah order tiap X menit)

```bash
# Tiap 15 menit pasang order baru selama sinyal sama
python main.py --symbol XAUUSD --mt5 --live --dca 15
```

### Multi-TP — Kunci profit sebagian

```bash
# Tutup 50% posisi di TP1, sisa jalan ke TP2
python main.py --symbol XAUUSD --tf 1h --mt5 --live --multi-tp
```

### Mode Akun Mikro (<1 juta IDR)

```bash
# Filter ketat, 1 order, lot minimum
python main.py --symbol XAUUSD --tf 1h --mt5 --live --micro

# Kombinasi terbaik untuk modal kecil
python main.py --symbol XAUUSD --tf 1h --mt5 --live --micro --multi-tp
```

### Trailing stop

```bash
python main.py --symbol XAUUSD --mt5 --live --trail 20
```

### Test koneksi order

```bash
python main.py --symbol XAUUSD --mt5 --force-trade BUY
python main.py --symbol XAUUSD --mt5 --force-trade SELL
```

### Backtest

```bash
python main.py --symbol XAUUSD --tf 1h --backtest
python main.py --symbol EURUSD --tf 1d --backtest
```

### Analisis multi symbol (jalankan terpisah)

```bash
# Terminal 1
python main.py --symbol XAUUSD --tf 5m --mt5 --live

# Terminal 2
python main.py --symbol EURUSD --tf 15m --mt5 --live

# Terminal 3
python main.py --symbol XAUEUR --tf 1h --mt5 --live
```

---

## Semua Argumen CLI

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--symbol` | XAUUSD | Pair trading |
| `--tf` | 1h | Timeframe: `1m` `5m` `15m` `1h` `4h` `1d` |
| `--live` | off | Refresh otomatis tiap 60 detik |
| `--mt5` | off | Connect ke MetaTrader 5 (data + auto order) |
| `--tv` | off | Ambil data dari TradingView |
| `--tv-user` | - | Username TradingView (opsional) |
| `--tv-pass` | - | Password TradingView (opsional) |
| `--lot` | 0.01 | Lot size per order |
| `--orders` | 15 | Jumlah order sekaligus saat sinyal masuk |
| `--dca` | 0 | Jeda DCA dalam menit (0 = off) |
| `--trail` | 0 | Trailing stop dalam pips (0 = off) |
| `--multi-tp` | off | Partial close: kunci 50% profit di TP1, sisa jalan ke TP2 |
| `--micro` | off | Mode akun mikro <1 juta IDR: 1 order, lot min, filter ketat |
| `--force-trade` | - | Paksa order: `BUY` atau `SELL` |
| `--lstm` | off | Aktifkan LSTM (butuh TensorFlow) |
| `--no-ml` | off | Nonaktifkan ML |
| `--no-news` | off | Nonaktifkan news filter |
| `--backtest` | off | Jalankan simulasi historis |
| `--mt-login` | .env | Override nomor akun MT5 |
| `--mt-pass` | .env | Override password MT5 |
| `--mt-server` | .env | Override server broker MT5 |
| `--mt-status` | off | Tampilkan status akun MT5 |
| `--mt4` | off | Aktifkan MT4 file bridge |
| `--mt-setup` | off | Generate EA file untuk MT4 |
| `--period` | auto | Override periode data: `3mo`, `1y`, dll |
| `--tune` | off | Tampilkan panduan tuning |

---

## Symbol yang Didukung

| Symbol | Deskripsi | MT5 Broker | TradingView |
|--------|-----------|-----------|-------------|
| `XAUUSD` | Gold / US Dollar | XAUUSDm | XAUUSD:OANDA |
| `GOLD` | Gold Futures (CME) | XAUUSDm | XAUUSD:OANDA |
| `XAUEUR` | Gold / Euro | XAUEURm | XAUEUR:OANDA |
| `EURUSD` | Euro / US Dollar | EURUSDm | EURUSD:OANDA |
| `GBPUSD` | British Pound / Dollar | GBPUSDm | GBPUSD:OANDA |
| `USDJPY` | US Dollar / Yen | USDJPYm | USDJPY:OANDA |
| `DXY` | US Dollar Index | USDXm | DXY:TVC |
| `BTCUSD` | Bitcoin / Dollar | BTCUSDm | BTCUSD:COINBASE |

> Nama symbol di broker berbeda-beda. Sesuaikan `SYMBOL_MAP` di `broker/mt5_connector.py`.

---

## Tuning Parameter

Edit `config.py`:

```python
# ─── DECISION LOGIC ───────────────────────────────────────
ADX_TREND_MIN    = 25     # ADX minimal untuk masuk trade (25 = trending)
RSI_OVERBOUGHT   = 70     # RSI maks untuk BUY
RSI_OVERSOLD     = 30     # RSI min untuk SELL

# ─── SL / TP ──────────────────────────────────────────────
ATR_MULTIPLIER_SL = 1.5   # SL = 1.5x ATR (naikkan = SL lebih longgar)
ATR_MULTIPLIER_TP = 2.0   # TP = 2x ATR   (naikkan = target lebih jauh)

# ─── BREAKEVEN ────────────────────────────────────────────
BREAKEVEN_ENABLED = True
BREAKEVEN_TRIGGER = 1.0   # Geser SL ke entry saat profit = 1x jarak SL
BREAKEVEN_BUFFER  = 0.1   # Buffer kecil di atas entry price

# ─── SL GUARD ─────────────────────────────────────────────
SL_DANGER_PCT     = 0.30  # Warning saat 30% sisa jarak ke SL
SL_CRITICAL_PCT   = 0.10  # Auto-close saat 10% sisa jarak ke SL

# ─── MULTI-TP (Partial Close) ──────────────────────────────
MULTI_TP_ENABLED  = False  # Aktifkan via --multi-tp
TP1_RATIO         = 0.5   # TP1 di 50% jarak ke TP penuh
TP1_CLOSE_PCT     = 50.0  # Tutup 50% volume di TP1

# ─── MICRO ACCOUNT ────────────────────────────────────────
MICRO_LOT         = 0.01  # Lot per trade (minimum)
MICRO_MAX_ORDERS  = 1     # Hanya 1 order aktif
MICRO_ML_CONF     = 75    # ML harus confidence >= 75%
MICRO_ADX_MIN     = 30    # ADX lebih ketat dari normal
MICRO_MIN_SCORE   = 7     # Skor sinyal minimum (normal: 5)

# ─── REFRESH ──────────────────────────────────────────────
REFRESH_INTERVAL  = 60    # Detik antar cycle
```

---

## Output Tiap Cycle

```
------------------------------------------------------------
  Cycle #5  |  09:15:00
------------------------------------------------------------
[MT5] 1500 candles live (close: 4855.750)
[DB] Database: 1540 candles (no update)
[OK] 1481 candles aktif | 1516 candles training ML
[OK] Model trained! Conf Acc: 71.05% [GOOD]
[OK] News [CACHE] - Risk: HIGH | Bias: BULLISH (+6.0)

  TRADING ANALYSIS - XAUUSD | 5m
  ══════════════════════════════════════════════════════════
  EMA20 > EMA50  ↑ | RSI: 61.5 | ADX: 27.3 (trending)
  Candle Trend: 4/5 candle BULLISH
  Direction: BUY  Score: +4.80
  SL: 4843.50  TP: 4877.00  R:R 1:2.0
  ══════════════════════════════════════════════════════════

  BULK ORDER — BUY × 15
  ══════════════════════════════════════════════════════════
  Lot    : 0.01 lot/order  (total: 0.15 lot)
  SL : 4843.50   TP : 4877.00
  Tickets : #123456 — #123470
  ✓ 15/15 order masuk
  ══════════════════════════════════════════════════════════

  [SL-BAHAYA] #123456 BUY — 28% sisa jarak ke SL  ⚠

  CANDLE LOG — XAUUSD 5m (last 8)
  Time               C   Close    Body  WickU  WickD   RSI  EMA20>50
  2026-03-19 09:05  B  4851.20   3.50   1.20   0.80  59.2        ↑
  2026-03-19 09:10  B  4853.80   2.60   0.50   1.10  61.5        ↑  BUY

  TRADE JOURNAL — XAUUSD 5m
  Total: 12  Win: 7  Loss: 5  Win Rate: 58.3%  P&L: +$142.50
```

---

## Disclaimer

> Bot ini dibuat untuk tujuan **edukasi dan riset**. Hasil backtest dan akurasi ML tidak menjamin profit di masa depan. Trading forex dan emas mengandung risiko tinggi kehilangan modal. Selalu gunakan **akun demo** sebelum live trading. Pengembang tidak bertanggung jawab atas kerugian finansial apapun.
