#======================================================
#  UMUM — berlaku untuk semua mode kecuali di-override
# ==============================================================

SYMBOLS = {
    "GOLD":     "GC=F",
    "XAUUSD":   "GC=F",
    "XAUEUR":   "GLD",
    "EURUSD":   "EURUSD=X",
    "GBPUSD":   "GBPUSD=X",
    "USDJPY":   "JPY=X",
    "DXY":      "DX-Y.NYB",
    "BTCUSD":   "BTC-USD",
}
DEFAULT_SYMBOL    = "XAUUSD"
DEFAULT_TIMEFRAME = "5m"

SYMBOL_DESC = {
    "GOLD":   "Gold Futures (CME)",
    "XAUUSD": "Gold / US Dollar",
    "XAUEUR": "Gold / Euro",
    "EURUSD": "Euro / US Dollar",
    "GBPUSD": "British Pound / US Dollar",
    "USDJPY": "US Dollar / Japanese Yen",
    "DXY":    "US Dollar Index",
    "BTCUSD": "Bitcoin / US Dollar",
}

TIMEFRAMES  = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
DATA_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d",
    "1h": "730d", "4h": "730d", "1d": "5y",
}

# Indikator
RSI_PERIOD       = 14
RSI_OVERBOUGHT   = 70
RSI_OVERSOLD     = 30

MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

EMA_FAST         = 9
EMA_SLOW         = 20
EMA_TREND        = 50
EMA_LONG         = 200

BB_PERIOD        = 20
BB_STD           = 2.0

STOCH_K          = 14
STOCH_D          = 3
STOCH_SMOOTH     = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD   = 20

ATR_PERIOD       = 14
ADX_PERIOD       = 14
ADX_TREND_MIN    = 25

# Sinyal
MIN_SIGNAL_SCORE = 5
WEIGHTS = {
    # Traditional indicators
    # RSI & MACD diturunkan — terlalu reaktif di M5, sering SELL saat koreksi uptrend
    "rsi":            1.0,   # 2.0→1.0: oscillator M5 terlalu noise
    "macd":           1.0,   # 2.0→1.0: lagging, false SELL saat pullback uptrend
    "ema_cross":      2.5,   # 2.0→2.5: trend alignment lebih penting
    "bb":             1.0,
    "stoch":          1.0,   # 1.5→1.0: terlalu reaktif di M5
    "adx":            1.5,   # 1.0→1.5: trend strength sangat penting
    "candle":         0.5,
    # Volume indicators
    "obv":            1.5,   # On Balance Volume
    "vwap":           2.0,   # 1.5→2.0: price vs VWAP lebih reliable
    "williams_r":     1.0,   # Williams %R
    "cci":            1.0,   # Commodity Channel Index
    "volume":         1.0,   # Volume spike + divergence
    # Smart Money Concepts
    "smc":            3.0,   # FVG + Order Block + BOS/ChoCH + Liquidity Sweep
    "pattern_ex":     1.5,   # Three Soldiers/Crows, Morning/Evening Star, Harami
    # Structure & Divergence (bobot tinggi — sinyal konfirmasi kuat)
    "rsi_div":        4.0,   # RSI Divergence — reversal terkuat
    "momentum_chain": 2.5,   # 2.0→2.5: HH+HL / LL+LH structure
    # Trend & Level indicators
    "sma":            1.5,   # SMA Golden/Death Cross
    "fibonacci":      2.0,   # Fibonacci Retracement levels
    # Trend-following indicators — dinaikkan karena lebih akurat di M5 trending
    "supertrend":     3.5,   # 2.5→3.5: ATR-based, leading indicator terbaik
    "mfi":            1.5,   # Money Flow Index (volume-weighted RSI)
    "psar":           2.0,   # 1.5→2.0: real-time trend flip
    "ichimoku":       3.5,   # 2.5→3.5: best multi-timeframe trend filter
}

# ML
ML_ENABLED            = False  # DINONAKTIFKAN — pakai rule-based saja
ML_LOOKBACK           = 20
ML_TRAIN_SPLIT        = 0.8
ML_MODEL_TYPE         = "ensemble"
ML_MIN_CONFIDENT_ACC  = 75.0   # ML-Only aktif jika conf_accuracy >= nilai ini
                               # Kalibrasi: model saat ini conf_acc ~79% → threshold 75%
ML_VOTE_THRESHOLD     = 75     # ML prob% minimum untuk join voting di #1 Rule+ML
                               # Kalibrasi: model precision ~68% → threshold 75%
                               # (65% terlalu rendah: beberapa sinyal 65-75% masih loss)

# Decision Engine — hard filters
NEWS_HIGH_BLOCK       = False  # scalping: news HIGH = penalty skor, bukan hard block
NO_TRADE_ZONE_PCT     = 0.5    # ATR multiplier: jarak "terlalu dekat" support/resistance
MIN_SIGNAL_SCORE      = 3.0    # recalibrated: RSI+MACD diturunkan → max score lebih kecil
MIN_QUALITY_SCORE     = 4      # min quality points (max 6): trend+candle+structure+momentum
MAX_OPEN_POSITIONS    = 1      # 1 posisi saja — tunggu SL/TP kena baru buka lagi

# Anti-Overtrading
TRADE_COOLDOWN_MIN    = 3      # 3 menit cooldown setelah entry
SL_COOLDOWN_MIN       = 0      # tidak ada cooldown setelah SL
MAX_TRADES_PER_HOUR   = 0      # tidak ada batas per jam
MAX_DAILY_TRADES      = 0      # tidak ada batas harian

# Arah berlawanan
ALLOW_OPPOSITE        = False  # False = skip SELL jika ada BUY, skip BUY jika ada SELL

# Lot safety cap
MAX_LOT_SAFE          = 0.03
MAX_LOT_LOSING        = 0.01
MIN_SL_PIPS           = 5.0    # SL minimum 5 pips (noise filter)

# SL / TP — fixed pips (lebih predictable dari ATR)
# Set FIXED_SL_PIPS > 0 untuk pakai fixed, 0 untuk pakai ATR multiplier
FIXED_SL_PIPS     = 4.0    # SL = 6 pip dari entry   ← ubah di sini
FIXED_TP_PIPS     = 6.0    # TP = 8 pip dari entry   ← ubah di sini
AUTO_TP_SL        = False
ATR_MULTIPLIER_SL = 1.0    # dipakai hanya jika FIXED_SL_PIPS = 0
ATR_MULTIPLIER_TP = 1.33   # dipakai hanya jika FIXED_TP_PIPS = 0
MIN_RR_RATIO      = 1.0    # tolak sinyal jika RR < nilai ini

# Spread filter
MAX_SPREAD_PIPS       = 1.5    # blok order jika spread > 1.5 pip

# ── Upgrade 2: Entry Sniper ────────────────────────────────────────────────
ENTRY_ZONE_PCT        = 3.0    # ATR multiplier: BUY hanya jika support dalam 3x ATR di bawah
                               # SELL hanya jika resistance dalam 3x ATR di atas
                               # Set 0 untuk disable

# ── Upgrade 4: Volatility Filter ──────────────────────────────────────────
MIN_ATR               = 2.0    # XAUUSD 5m: skip jika ATR < 2 (market terlalu sepi)

# ── Upgrade 6: Session Filter Pro ─────────────────────────────────────────
SESSION_FILTER        = True   # hanya trade saat sesi aktif (UTC)
ASIAN_OPEN_UTC        = 22     # 22:00 UTC = 05:00 WIB (Tokyo/Sydney open)
ASIAN_CLOSE_UTC       = 7      # 07:00 UTC = 14:00 WIB
LONDON_OPEN_UTC       = 7      # 07:00 UTC = 14:00 WIB
LONDON_CLOSE_UTC      = 16     # 16:00 UTC = 23:00 WIB
NY_OPEN_UTC           = 12     # 12:00 UTC = 19:00 WIB
NY_CLOSE_UTC          = 21     # 21:00 UTC = 04:00 WIB (+1)

# ── Upgrade 3: Confirmation Delay ─────────────────────────────────────────
CONFIRM_DELAY_ENABLED = False  # DINONAKTIFKAN — entry langsung tanpa delay

# ── Upgrade 8: Loss Control System ────────────────────────────────────────
LOSS_STREAK_PAUSE     = 0      # DINONAKTIFKAN
LOSS_STREAK_MIN       = 0      # DINONAKTIFKAN

# ── Upgrade 7: Trade Management Intelligence ──────────────────────────────
PROFIT_LOCK_PCT       = 0.5    # lock profit (SL ke entry) saat floating >= 50% jarak TP
REVERSAL_CLOSE        = True   # tutup posisi profitable saat signal berbalik

# Breakeven
BREAKEVEN_ENABLED = True
BREAKEVEN_TRIGGER = 1.0
BREAKEVEN_BUFFER  = 0.1

# SL Guard
SL_DANGER_PCT   = 0.30
SL_CRITICAL_PCT = 0.10

# Multi-TP / Partial Close
MULTI_TP_ENABLED = False
TP1_RATIO        = 0.5
TP1_CLOSE_PCT    = 50.0

# Lain-lain
BACKTEST_PERIOD  = "1y"
INITIAL_CAPITAL  = 10000
REFRESH_INTERVAL = 60
MAX_STACK        = 5      # maks posisi tumpuk searah (general mode)
SHOW_CHART       = True


# ==============================================================
#  MODE MIKRO  —  aktif dengan flag --micro
#  Untuk akun demo / latihan / modal < Rp 1 juta
# ==============================================================

MICRO_LOT        = 0.01
MICRO_MAX_ORDERS = 1
MICRO_MIN_SCORE  = 7
MICRO_ML_CONF    = 70
MICRO_ADX_MIN    = 30
MICRO_RISK_PCT   = 0.5


# ==============================================================
#  MODE REAL  —  aktif dengan flag --real
#  Untuk akun real ~$60 USD, target profit $15-20 per trade
# ==============================================================

REAL_LOT        = 0.01   # lot fixed 0.01
REAL_MAX_ORDERS = 1      # jumlah order sekaligus per sinyal
REAL_MAX_STACK  = 3      # maks posisi tumpuk searah (sinyal bagus → pasang lagi)
REAL_ML_CONF    = 70     # ML wajib setuju minimal 70% → akurasi 91.2%, coverage 55.6%
REAL_ADX_MIN    = 20     # scalping: tidak perlu trend kuat, cukup ada pergerakan
REAL_TRAIL_PIPS = 15.0   # trailing stop 15 pips
REAL_RISK_PCT   = 10.0   # risk 10% balance per trade → lot proporsional ke SL

# Auto lot — bot baca saldo dan hitung lot otomatis
# Formula: lot = balance / 10000  (setiap $100 = 0.01 lot)
# Contoh: $59 → 0.01 | $200 → 0.02 | $500 → 0.05 | $1000 → 0.10 | $5000 → 0.50
REAL_AUTO_LOT     = False  # False = pakai REAL_LOT fixed 0.01
REAL_AUTO_LOT_MAX = 0.10   # batas maksimal lot
REAL_AUTO_LOT_MIN = 0.01   # batas minimal lot

# RR 1:3 — SL kecil, TP 3x SL (lebih realistis untuk M5)
# 1 trade menang = tutup 3 trade rugi → win rate 35% sudah cukup
REAL_ATR_SL     = 0.5    # SL = 0.5x ATR
REAL_ATR_TP     = 0.5    # TP = 0.5x ATR  →  RR 1:1

REAL_MAX_FLOATING_USD = 0.0   # DINONAKTIFKAN — tidak stop saat floating loss
REAL_SL_COOLDOWN      = 0     # DINONAKTIFKAN — tidak ada cooldown setelah SL

# ── Daily Profit/Loss Limit (dinamis, % dari balance awal hari) ───────────────
# Contoh: balance $500, REAL_DAILY_LIMIT_PCT=0.04 → limit = $20/hari
# Reset setiap hari baru, nominal dihitung ulang dari balance saat itu
REAL_DAILY_LIMIT_PCT  = 0.20  # 20% dari balance harian → profit & loss limit
REAL_MAX_DAILY_PROFIT = 0.0   # (legacy, tidak dipakai — digantikan REAL_DAILY_LIMIT_PCT)
REAL_MAX_DAILY_LOSS   = 0.0   # (legacy, tidak dipakai — digantikan REAL_DAILY_LIMIT_PCT)

# ── Risk Control Final (semua mode, bukan hanya REAL) ─────────────────────────
# Batas berdasarkan % balance — lebih aman dari nominal karena skala dengan akun
DAILY_LOSS_PCT      = 0.0   # DINONAKTIFKAN — tidak stop saat rugi % balance
WEEKLY_LOSS_PCT     = 0.0   # DINONAKTIFKAN — tidak pause minggu ini saat rugi
DAILY_LOSS_HARD_USD = 0.0   # DINONAKTIFKAN — tidak ada hard stop nominal
FORCE_TRADE_MAX_LOT = 0.03  # lot maksimum yang diizinkan meski via force-trade
