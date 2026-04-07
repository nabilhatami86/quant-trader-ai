"""
data/session_bias.py — Analisis saat sesi tutup, tentukan bias sesi berikutnya.

Jam tutup (WIB / UTC+7):
  Tokyo close   : ~16:00 WIB (09:00 UTC)  — bias sebelum London buka
  London close  : ~00:00 WIB (17:00 UTC)  — bias sebelum NY buka
  Daily close   : ~05:00 WIB (22:00 UTC)  — bias untuk hari berikutnya (paling penting)
  Weekly close  : Sabtu 05:00 WIB         — bias weekly

Fungsi utama:
  run_close_analysis(symbol, mt5_conn, session_label, next_session) -> dict
      Jalankan analisis HTF saat session close, simpan bias, tampilkan laporan.

  get_current_bias() -> dict
      Ambil bias aktif (kadaluarsa setelah 8 jam). Return {} jika tidak ada.

  is_near_session_close(window_min=15) -> (bool, label, next_label)
      Cek apakah sekarang dalam window 15 menit sebelum session close.

  is_market_closed(mt5_conn) -> bool
      Deteksi weekend/holiday via weekday check + MT5 tick staleness (>30 menit).

  run_market_close_deep_analysis(symbol, bot, mt5_conn) -> dict
      Analisis mendalam saat market tutup (4 langkah):
        1. HTF bias analysis (H4 + Daily)
        2. Weekly key levels (resistance, support, Fibonacci 38.2%/61.8%)
        3. Adaptive learning report (dari AdaptiveLearner)
        4. ML retrain dengan semua data
      Hasil disimpan ke data/session_plan.json untuk dipakai Senin open.

  get_session_plan() -> dict
      Baca pre-session plan yang dibuat saat market tutup.

HTF Bias Score Components:
  +2.0 H4 EMA20 > EMA50 + price atas EMA20 (uptrend)
  +1.0 RSI H4 > 55 (momentum bullish)
  +1.5 H4 struktur HH/HL (min 6 HH, 5 HL)
  +1.0 Daily: 2+ dari 3 candle terakhir bullish
  +0.5 Daily: price atas EMA50
  +/-0.5 Proximity ke swing high/low
  * 0.6 jika ADX H4 < 25 (trend lemah)
  Direction: score >= +2.5 = BUY, <= -2.5 = SELL, else NEUTRAL

State disimpan di: data/session_bias_state.json
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

STATE_PATH = os.path.join(os.path.dirname(__file__), "session_bias_state.json")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

# ── Session windows (UTC hour, name, next_session) ────────────────────────────
SESSION_CLOSES = [
    # (utc_hour_close, window_minutes, session_label,  next_session_label)
    (9,  15, "Tokyo",   "London"),          # 09:00 UTC = 16:00 WIB
    (17, 15, "London",  "New York"),         # 17:00 UTC = 00:00 WIB
    (22, 15, "Daily",   "Asian/Tokyo"),      # 22:00 UTC = 05:00 WIB ★
    (0,  15, "Weekly",  "Asian/Tokyo"),      # 00:00 UTC Senin = 07:00 WIB
]


# ═══════════════════════════════════════════════════════════════════════════════
def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def get_current_bias() -> dict:
    """Ambil bias sesi aktif. Return dict atau {} jika tidak ada."""
    state = _load_state()
    bias  = state.get("current_bias", {})
    if not bias:
        return {}
    # Bias expired setelah 8 jam
    try:
        ts = datetime.fromisoformat(bias.get("timestamp", ""))
        if (datetime.now() - ts).total_seconds() > 8 * 3600:
            return {}
    except Exception:
        pass
    return bias


def is_near_session_close(window_min: int = 15) -> tuple[bool, str, str]:
    """
    Cek apakah sekarang mendekati jam tutup sesi.
    Return: (is_close, session_label, next_session_label)
    """
    now_utc = _now_utc()
    hour    = now_utc.hour
    minute  = now_utc.minute

    for (close_hour, win, label, next_label) in SESSION_CLOSES:
        # Hitung menit sebelum close
        mins_to_close = (close_hour - hour) * 60 - minute
        if close_hour <= hour:
            mins_to_close += 24 * 60

        if 0 <= mins_to_close <= window_min:
            return True, label, next_label

    return False, "", ""


def was_analyzed_recently(session_label: str, hours: int = 6) -> bool:
    """Hindari analisis ganda dalam window yang sama."""
    state = _load_state()
    history = state.get("history", [])
    cutoff  = datetime.now() - timedelta(hours=hours)
    for h in history[-10:]:
        if h.get("session") == session_label:
            try:
                ts = datetime.fromisoformat(h["timestamp"])
                if ts > cutoff:
                    return True
            except Exception:
                pass
    return False


# ═══════════════════════════════════════════════════════════════════════════════
def run_close_analysis(symbol: str, mt5_connector=None,
                       session_label: str = "Daily",
                       next_session: str = "Asian/Tokyo") -> dict:
    """
    Jalankan analisis close session menggunakan data H4 + Daily.
    Return bias dict: {"direction": "BUY"/"SELL"/"NEUTRAL", "score": float, ...}
    """
    print(f"\n  {'='*60}")
    print(f"  {BOLD}{CYAN}[SESSION CLOSE ANALYSIS]{RESET}"
          f"  {session_label} Close -> Bias {next_session}")
    print(f"  {'='*60}")

    bias = _analyze_htf(symbol, mt5_connector, session_label)

    # Simpan ke state
    state = _load_state()
    bias["timestamp"]    = datetime.now().isoformat()
    bias["session"]      = session_label
    bias["next_session"] = next_session

    state["current_bias"] = bias
    if "history" not in state:
        state["history"] = []
    state["history"].append({
        "timestamp":    bias["timestamp"],
        "session":      session_label,
        "next_session": next_session,
        "direction":    bias.get("direction", "NEUTRAL"),
        "score":        bias.get("score", 0),
    })
    # Jaga max 20 history
    state["history"] = state["history"][-20:]
    _save_state(state)

    _print_bias(bias)
    return bias


def _analyze_htf(symbol: str, mt5_conn=None,
                 session_label: str = "Daily") -> dict:
    """
    Analisis Higher Timeframe (H4 + Daily).
    Tentukan bias berdasarkan: trend, momentum, key level, candle pattern.
    """
    reasons  = []
    score    = 0.0
    details  = {}

    # ── Ambil data H4 ────────────────────────────────────────────────────────
    df_h4    = _fetch_htf(symbol, "4h", 100, mt5_conn)
    df_daily = _fetch_htf(symbol, "1d", 30,  mt5_conn)

    if df_h4 is None or df_h4.empty:
        return {"direction": "NEUTRAL", "score": 0, "reasons": ["Data H4 tidak tersedia"],
                "details": {}}

    # ── 1. Trend H4 (EMA 20/50) ──────────────────────────────────────────────
    close_h4 = df_h4["Close"]
    ema20_h4 = close_h4.ewm(span=20, adjust=False).mean()
    ema50_h4 = close_h4.ewm(span=50, adjust=False).mean()

    last_close = float(close_h4.iloc[-1])
    last_e20   = float(ema20_h4.iloc[-1])
    last_e50   = float(ema50_h4.iloc[-1])

    details["h4_close"]  = round(last_close, 5)
    details["h4_ema20"]  = round(last_e20, 5)
    details["h4_ema50"]  = round(last_e50, 5)

    if last_e20 > last_e50 and last_close > last_e20:
        score += 2.0
        reasons.append("H4 EMA20>50 + price atas EMA20 (uptrend)")
        details["h4_trend"] = "BULLISH"
    elif last_e20 < last_e50 and last_close < last_e20:
        score -= 2.0
        reasons.append("H4 EMA20<50 + price bawah EMA20 (downtrend)")
        details["h4_trend"] = "BEARISH"
    else:
        reasons.append("H4 trend tidak jelas (EMA mixed)")
        details["h4_trend"] = "MIXED"

    # ── 2. RSI H4 ─────────────────────────────────────────────────────────────
    delta    = close_h4.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0, np.nan)
    rsi_h4   = 100 - (100 / (1 + rs))
    last_rsi = float(rsi_h4.iloc[-1]) if not rsi_h4.empty else 50.0

    details["h4_rsi"] = round(last_rsi, 1)

    if last_rsi > 55:
        score += 1.0
        reasons.append(f"RSI H4 {last_rsi:.1f} — momentum bullish")
    elif last_rsi < 45:
        score -= 1.0
        reasons.append(f"RSI H4 {last_rsi:.1f} — momentum bearish")
    else:
        reasons.append(f"RSI H4 {last_rsi:.1f} — netral")

    # ── 3. ADX H4 (trend kuat?) ───────────────────────────────────────────────
    high_h4 = df_h4["High"]
    low_h4  = df_h4["Low"]
    tr      = pd.concat([
        high_h4 - low_h4,
        (high_h4 - close_h4.shift()).abs(),
        (low_h4  - close_h4.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_h4  = tr.rolling(14).mean()
    dm_pos  = (high_h4.diff().clip(lower=0))
    dm_neg  = (-low_h4.diff().clip(upper=0))
    dip     = (dm_pos.rolling(14).mean() / atr_h4.replace(0, np.nan)) * 100
    din     = (dm_neg.rolling(14).mean() / atr_h4.replace(0, np.nan)) * 100
    dx      = ((dip - din).abs() / (dip + din).replace(0, np.nan)) * 100
    adx_h4  = dx.rolling(14).mean()
    last_adx = float(adx_h4.iloc[-1]) if not adx_h4.empty else 0.0

    details["h4_adx"] = round(last_adx, 1)

    if last_adx >= 25:
        reasons.append(f"ADX H4 {last_adx:.1f} — trend kuat (valid)")
    else:
        score *= 0.6   # Lemahkan score kalau trend lemah
        reasons.append(f"ADX H4 {last_adx:.1f} — trend lemah, score dikurangi")

    # ── 4. Structure (Higher Highs / Lower Lows di H4) ───────────────────────
    highs = df_h4["High"].tail(10).values
    lows  = df_h4["Low"].tail(10).values

    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    hl = sum(1 for i in range(1, len(lows))  if lows[i]  > lows[i-1])
    lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
    ll = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i-1])

    details["h4_structure"] = {"HH": hh, "HL": hl, "LH": lh, "LL": ll}

    if hh >= 6 and hl >= 5:
        score += 1.5
        reasons.append(f"H4 struktur HH/HL ({hh} HH, {hl} HL) — uptrend")
    elif ll >= 6 and lh >= 5:
        score -= 1.5
        reasons.append(f"H4 struktur LL/LH ({ll} LL, {lh} LH) — downtrend")

    # ── 5. Daily candle arah ─────────────────────────────────────────────────
    if df_daily is not None and len(df_daily) >= 3:
        d_close = df_daily["Close"]
        d_open  = df_daily["Open"]
        bull_count = sum(1 for i in range(-3, 0) if d_close.iloc[i] > d_open.iloc[i])
        bear_count = 3 - bull_count

        details["daily_bull_3"] = bull_count
        details["daily_bear_3"] = bear_count

        if bull_count >= 2:
            score += 1.0
            reasons.append(f"Daily: {bull_count}/3 candle terakhir bullish")
        elif bear_count >= 2:
            score -= 1.0
            reasons.append(f"Daily: {bear_count}/3 candle terakhir bearish")

        # Daily EMA50 alignment
        d_ema50 = d_close.ewm(span=50, adjust=False).mean()
        d_last  = float(d_close.iloc[-1])
        d_e50   = float(d_ema50.iloc[-1])
        details["daily_ema50"] = round(d_e50, 5)

        if d_last > d_e50:
            score += 0.5
            reasons.append(f"Daily: price {d_last:.2f} atas EMA50 ({d_e50:.2f})")
        else:
            score -= 0.5
            reasons.append(f"Daily: price {d_last:.2f} bawah EMA50 ({d_e50:.2f})")

    # ── 6. Key level proximity (support/resistance dari H4) ─────────────────
    swing_high = float(df_h4["High"].tail(50).max())
    swing_low  = float(df_h4["Low"].tail(50).min())
    mid_level  = (swing_high + swing_low) / 2
    atr_val    = float(atr_h4.iloc[-1]) if not atr_h4.empty else 10.0

    details["swing_high"] = round(swing_high, 5)
    details["swing_low"]  = round(swing_low, 5)
    details["mid_level"]  = round(mid_level, 5)

    near_high = abs(last_close - swing_high) < atr_val * 1.5
    near_low  = abs(last_close - swing_low)  < atr_val * 1.5
    near_mid  = abs(last_close - mid_level)  < atr_val * 1.0

    if near_high:
        score -= 0.5
        reasons.append(f"Harga dekat swing high {swing_high:.2f} — potensi reversal SELL")
    elif near_low:
        score += 0.5
        reasons.append(f"Harga dekat swing low {swing_low:.2f} — potensi reversal BUY")
    elif near_mid:
        reasons.append(f"Harga di area mid-range {mid_level:.2f} — konsolidasi")

    # ── Final direction ───────────────────────────────────────────────────────
    details["final_score"] = round(score, 2)

    if score >= 2.5:
        direction = "BUY"
        strength  = "KUAT" if score >= 4 else "SEDANG"
    elif score <= -2.5:
        direction = "SELL"
        strength  = "KUAT" if score <= -4 else "SEDANG"
    else:
        direction = "NEUTRAL"
        strength  = "LEMAH"

    return {
        "direction": direction,
        "strength":  strength,
        "score":     round(score, 2),
        "reasons":   reasons,
        "details":   details,
        "h4_price":  last_close,
    }


def _fetch_htf(symbol: str, timeframe: str, count: int,
               mt5_conn=None) -> pd.DataFrame | None:
    """Ambil data higher timeframe dari MT5."""
    if mt5_conn is None:
        return None
    try:
        import MetaTrader5 as mt5
        from backend.broker.mt5_connector import SYMBOL_MAP

        TF_MAP = {
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        mt5_tf  = TF_MAP.get(timeframe)
        if mt5_tf is None:
            return None
        sym_mt5 = SYMBOL_MAP.get(symbol.upper(), symbol + "m")
        rates   = mt5.copy_rates_from_pos(sym_mt5, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low",   "close": "Close",
            "tick_volume": "Volume",
        }, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        return None


def _print_bias(bias: dict):
    direction = bias.get("direction", "NEUTRAL")
    score     = bias.get("score", 0)
    strength  = bias.get("strength", "")
    reasons   = bias.get("reasons", [])
    details   = bias.get("details", {})
    session   = bias.get("session", "")
    next_sess = bias.get("next_session", "")

    dc = GREEN if direction == "BUY" else (RED if direction == "SELL" else YELLOW)

    print(f"\n  {BOLD}Hasil Analisis Close — {session}{RESET}")
    print(f"  Bias {next_sess}  : {dc}{BOLD}{direction}{RESET}  "
          f"(score: {score:+.1f}, {strength})")

    if details.get("h4_trend"):
        tc = GREEN if details["h4_trend"] == "BULLISH" else \
             (RED if details["h4_trend"] == "BEARISH" else YELLOW)
        print(f"  H4 Trend    : {tc}{details['h4_trend']}{RESET}  "
              f"EMA20={details.get('h4_ema20',0):.2f}  EMA50={details.get('h4_ema50',0):.2f}")
    if details.get("h4_rsi"):
        print(f"  H4 RSI      : {details['h4_rsi']}  "
              f"ADX={details.get('h4_adx', 0)}")
    if details.get("swing_high"):
        print(f"  Key Levels  : High={details['swing_high']:.2f}  "
              f"Low={details['swing_low']:.2f}  Mid={details['mid_level']:.2f}")

    print(f"\n  Alasan:")
    for r in reasons:
        icon = "^" if "bullish" in r.lower() or "uptrend" in r.lower() \
               else ("v" if "bearish" in r.lower() or "downtrend" in r.lower() else "-")
        c = GREEN if icon == "^" else (RED if icon == "v" else DIM)
        print(f"    {c}{icon} {r}{RESET}")

    print(f"  {'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET CLOSED / HOLIDAY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_market_closed(mt5_conn=None) -> bool:
    """
    Cek apakah market sedang tutup (weekend / holiday).
    MT5: coba ambil tick terakhir — kalau gagal atau stale >30 menit = CLOSED
    """
    now_utc = _now_utc()

    # Weekend: Sabtu 22:00 UTC s/d Minggu 22:00 UTC (Senin 05:00 WIB)
    weekday = now_utc.weekday()   # 5=Sabtu, 6=Minggu
    if weekday == 5 and now_utc.hour >= 22:
        return True
    if weekday == 6:
        return True
    if weekday == 0 and now_utc.hour < 22 and now_utc.hour >= 0:
        # Senin dini hari sebelum 05:00 WIB (22:00 UTC Minggu)
        # Sebenarnya Forex buka Senin 00:00 UTC — cukup cek Minggu saja
        pass

    # MT5 live check: tick stale?
    if mt5_conn is not None:
        try:
            import MetaTrader5 as mt5
            from backend.broker.mt5_connector import SYMBOL_MAP
            sym = SYMBOL_MAP.get("XAUUSD", "XAUUSDm")
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                return True
            import time
            age_sec = time.time() - tick.time
            if age_sec > 1800:   # > 30 menit tidak ada tick → tutup
                return True
        except Exception:
            pass

    return False


def run_market_close_deep_analysis(symbol: str, bot=None,
                                   mt5_conn=None) -> dict:
    """
    Analisis mendalam saat market tutup (weekend / holiday).
    Jalankan:
      1. HTF bias analysis (H1 + H4 + Daily)
      2. Weekly structure review
      3. ML retrain dengan semua data
      4. Simpan rencana sesi untuk next open
    """
    print(f"\n  {'='*60}")
    print(f"  {BOLD}{CYAN}[MARKET CLOSED — DEEP ANALYSIS]{RESET}")
    print(f"  {'='*60}")
    print(f"  {DIM}Market sedang tutup. Bot belajar dari semua data tersedia...{RESET}\n")

    results = {}

    # ── 1. HTF Bias (H4 + Daily + Weekly) ────────────────────────────────────
    print(f"  {BOLD}[1/4] Analisis Higher Timeframe (H4 + Daily)...{RESET}")
    bias = _analyze_htf(symbol, mt5_conn, "Weekly")
    bias["timestamp"]    = datetime.now().isoformat()
    bias["session"]      = "Weekly"
    bias["next_session"] = "Senin Open"

    # Simpan sebagai current bias (berlaku hingga 48 jam untuk weekend)
    state = _load_state()
    state["current_bias"] = bias
    state.setdefault("history", []).append({
        "timestamp":  bias["timestamp"],
        "session":    "Weekly",
        "direction":  bias.get("direction", "NEUTRAL"),
        "score":      bias.get("score", 0),
    })
    state["history"] = state["history"][-20:]
    _save_state(state)
    results["htf_bias"] = bias
    _print_bias(bias)

    # ── 2. Weekly structure: key levels untuk next week ───────────────────────
    print(f"  {BOLD}[2/4] Identifikasi Key Levels Minggu Depan...{RESET}")
    details  = bias.get("details", {})
    sh       = details.get("swing_high", 0)
    sl_      = details.get("swing_low", 0)
    mid      = details.get("mid_level", 0)
    if sh and sl_:
        fib382 = round(sh - (sh - sl_) * 0.382, 5)
        fib618 = round(sh - (sh - sl_) * 0.618, 5)
        results["key_levels"] = {
            "resistance": sh, "support": sl_,
            "mid": mid, "fib382": fib382, "fib618": fib618,
        }
        print(f"    Resistance : {sh:.2f}")
        print(f"    Fib 38.2%  : {fib382:.2f}")
        print(f"    Mid Range  : {mid:.2f}")
        print(f"    Fib 61.8%  : {fib618:.2f}")
        print(f"    Support    : {sl_:.2f}")

    # ── 3. Adaptive learning report ───────────────────────────────────────────
    print(f"\n  {BOLD}[3/4] Adaptive Learning Report...{RESET}")
    try:
        from ai.adaptive import get_learner
        learner = get_learner()
        learner.print_report()
        results["adaptive"] = {
            "min_score": learner.min_score,
            "total_trades": learner.state["total_trades"],
        }
    except Exception as e:
        print(f"    [!] Adaptive unavailable: {e}")

    # ── 4. ML Retrain dengan semua data ───────────────────────────────────────
    print(f"\n  {BOLD}[4/4] ML Retrain (full dataset)...{RESET}")
    if bot is not None:
        try:
            if bot.load_data():
                from joblib import parallel_backend
                with parallel_backend("threading", n_jobs=1):
                    bot.train_model()
                results["ml_retrained"] = True
                print(f"    {GREEN}[OK] ML retrained dengan data terbaru{RESET}")

                # Reset retrain counter
                from ai.adaptive import get_learner
                get_learner().mark_retrained()
        except Exception as e:
            print(f"    [!] ML retrain gagal: {e}")
            results["ml_retrained"] = False
    else:
        print(f"    {YELLOW}[!] Bot tidak tersedia — skip ML retrain{RESET}")
        results["ml_retrained"] = False

    # ── Simpan pre-session plan ───────────────────────────────────────────────
    plan_path = os.path.join(os.path.dirname(__file__), "session_plan.json")
    try:
        plan = {
            "generated_at":  datetime.now().isoformat(),
            "next_open":     "Senin",
            "htf_bias":      bias.get("direction", "NEUTRAL"),
            "htf_score":     bias.get("score", 0),
            "key_levels":    results.get("key_levels", {}),
            "ml_retrained":  results.get("ml_retrained", False),
            "min_score":     results.get("adaptive", {}).get("min_score", 5.0),
        }
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"\n  {GREEN}[OK] Pre-session plan disimpan → {plan_path}{RESET}")
    except Exception:
        pass

    print(f"\n  {'='*60}")
    print(f"  {BOLD}Deep analysis selesai.{RESET}")
    direction = bias.get("direction", "NEUTRAL")
    dc = GREEN if direction == "BUY" else (RED if direction == "SELL" else YELLOW)
    print(f"  Bias Senin depan : {dc}{BOLD}{direction}{RESET}  "
          f"(score: {bias.get('score', 0):+.1f})")
    print(f"  {'='*60}\n")

    return results


def get_session_plan() -> dict:
    """Ambil pre-session plan yang dibuat saat market tutup."""
    plan_path = os.path.join(os.path.dirname(__file__), "session_plan.json")
    if not os.path.exists(plan_path):
        return {}
    try:
        with open(plan_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Print current bias (dipanggil dari bot) ───────────────────────────────────
def print_current_bias():
    bias = get_current_bias()
    if not bias:
        return
    direction = bias.get("direction", "NEUTRAL")
    score     = bias.get("score", 0)
    session   = bias.get("session", "")
    next_s    = bias.get("next_session", "")
    ts        = bias.get("timestamp", "")[:16]
    dc = GREEN if direction == "BUY" else (RED if direction == "SELL" else YELLOW)
    print(f"  [Session Bias] {ts}  {session} Close: "
          f"{dc}{BOLD}{direction}{RESET} (score {score:+.1f}) "
          f"— berlaku untuk {next_s}")
