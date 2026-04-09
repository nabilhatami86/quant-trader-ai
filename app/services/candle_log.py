"""
data/candle_log.py — Logging candle + indikator + sinyal, memory pola historis.

Fungsi utama:
  log_candle(symbol, tf, df_ind, signal)
      Simpan candle terbaru + indikator + sinyal ke CSV:
      logs/candles_{symbol}_{tf}.csv

  find_similar_candles(symbol, tf, current_row, top_k=5)
      Cari N candle historis yang paling mirip secara teknikal
      menggunakan cosine similarity atas 12 fitur utama:
      body, wick_up, wick_down, rsi, macd, adx, atr, ema_ratio,
      bb_pct, volume_ratio, supertrend, stoch_k

  update_signal_outcomes(symbol, tf, df_ind, lookahead=3)
      Perbarui kolom outcome di CSV setelah 3 candle berlalu:
      "correct" jika harga bergerak searah sinyal, "wrong" jika tidak

  get_signal_accuracy(symbol, tf, current_row, direction, top_k=20)
      Hitung akurasi historis sinyal serupa:
      dari 20 candle serupa, berapa % yang outcome-nya correct?

  print_similar_report(result)
      Tampilkan laporan pola serupa di terminal (warna-warni)

  print_recent(symbol, tf, n)
      Tampilkan N candle terakhir dari CSV log

Storage:
  logs/candles_{symbol}_{tf}.csv — 38 kolom:
  time, OHLC, candle_dir, body, wick_up/down, pattern,
  rsi, ema20/50, macd, histogram, adx, atr, signal, score,
  ml_direction, ml_prob, outcome, outcome_time, ...
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# app/services/ -> app/ -> root -> logs/
LOG_DIR = str(Path(__file__).parent.parent.parent / 'logs')


def _log_path(symbol: str, timeframe: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"candles_{symbol}_{timeframe}.csv")


def log_candle(symbol: str, timeframe: str, df_ind: "pd.DataFrame",
               signal: dict = None, realtime_data: dict = None) -> None:
    if df_ind is None or df_ind.empty:
        return

    path  = _log_path(symbol, timeframe)
    row   = df_ind.iloc[-1]
    open_ = float(row.get("Open",  0))
    high  = float(row.get("High",  0))
    low   = float(row.get("Low",   0))
    close = float(row.get("Close", 0))

    candle_dir = "BULLISH" if close > open_ else "BEARISH" if close < open_ else "DOJI"
    body_size  = round(abs(close - open_), 5)
    wick_up    = round(high - max(open_, close), 5)
    wick_down  = round(min(open_, close) - low, 5)

    from config import EMA_SLOW, EMA_TREND
    record = {
        "time":      str(df_ind.index[-1])[:19],
        "open":      round(open_, 5),
        "high":      round(high,  5),
        "low":       round(low,   5),
        "close":     round(close, 5),
        "candle":    candle_dir,
        "body":      body_size,
        "wick_up":   wick_up,
        "wick_down": wick_down,
        "pattern":   str(row.get("candle_name", "")),
        "rsi":       round(float(row.get("rsi",  50)), 2),
        "ema20":     round(float(row.get(f"ema_{EMA_SLOW}",  close)), 5),
        "ema50":     round(float(row.get(f"ema_{EMA_TREND}", close)), 5),
        "macd":      round(float(row.get("macd", 0)), 5),
        "histogram": round(float(row.get("histogram", 0)), 5),
        "adx":       round(float(row.get("adx",  0)), 2),
        "atr":       round(float(row.get("atr",  0)), 5),
        "signal":    signal.get("direction", "") if signal else "",
        "score":     signal.get("score", "") if signal else "",
        "sl":        signal.get("sl", "") if signal else "",
        "tp":        signal.get("tp", "") if signal else "",
        # Realtime data dari MT5 (tick + orderbook)
        "tick_dir":   (realtime_data or {}).get("tick_momentum", {}).get("direction", ""),
        "tick_bull%": round((realtime_data or {}).get("tick_momentum", {}).get("bull_ratio", 0) * 100, 1),
        "ob_bias":    (realtime_data or {}).get("orderbook", {}).get("bias", ""),
        "ob_imbal":   (realtime_data or {}).get("orderbook", {}).get("imbalance", ""),
        "rt_bias":    (realtime_data or {}).get("realtime_bias", ""),
        "rt_score":   (realtime_data or {}).get("realtime_score", ""),
        "spread":     (realtime_data or {}).get("current_tick", {}).get("spread", ""),
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df_row = pd.DataFrame([record])
    current_cols = set(record.keys())

    if os.path.exists(path):
        try:
            existing = pd.read_csv(path, on_bad_lines="skip")
            # Schema changed (e.g. realtime columns added) → restart file
            if set(existing.columns) != current_cols:
                df_row.to_csv(path, index=False)
                return
            if len(existing) and existing["time"].iloc[-1] == record["time"]:
                existing = existing.iloc[:-1]
                existing = pd.concat([existing, df_row], ignore_index=True)
                existing.to_csv(path, index=False)
            else:
                df_row.to_csv(path, mode="a", header=False, index=False)
        except Exception:
            # Corrupted CSV → overwrite
            df_row.to_csv(path, index=False)
    else:
        df_row.to_csv(path, index=False)


def _v(row, key, default=0):
    """Safe float — returns default when key is missing or None/NaN."""
    val = row.get(key)
    if val is None:
        return float(default)
    try:
        v = float(val)
        return default if (v != v) else v   # NaN check
    except (TypeError, ValueError):
        return float(default)


def _candle_features(row: pd.Series) -> np.ndarray:
    body  = _v(row, "body")
    wu    = _v(row, "wick_up")
    wd    = _v(row, "wick_down")
    rng   = (body + wu + wd) or 1.0
    rsi   = _v(row, "rsi", 50)
    adx   = _v(row, "adx")
    ema20 = _v(row, "ema20", 1)
    ema50 = _v(row, "ema50", 1)
    hist  = _v(row, "histogram")
    bull  = 1.0 if str(row.get("candle", "")) == "BULLISH" else -1.0

    return np.array([
        body / rng,               # body ratio
        wu / rng,                 # upper wick ratio
        wd / rng,                 # lower wick ratio
        (rsi - 50) / 50,          # RSI normalised -1..1
        min(adx, 60) / 60,        # ADX normalised
        1.0 if ema20 > ema50 else -1.0,   # EMA cross direction
        np.sign(hist),            # MACD histogram sign
        bull,                     # candle direction
    ], dtype=float)


# Feature weights — RSI & EMA cross lebih penting dari ukuran wick
_WEIGHTS = np.array([1.5, 0.8, 0.8, 2.0, 1.0, 2.5, 1.5, 1.5])


def _load_df(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load candle log — DB sebagai sumber utama (lebih banyak data + SMC),
    fallback ke CSV kalau DB tidak tersedia.
    """
    try:
        from app.services.db_logger import load_candle_logs_df
        df = load_candle_logs_df(symbol, timeframe, limit=10000)
        if not df.empty:
            return df
    except Exception:
        pass
    # Fallback ke CSV
    path = _log_path(symbol, timeframe)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def find_similar_candles(symbol: str, timeframe: str,
                          current_row: pd.Series,
                          top_k: int = 15,
                          lookahead: int = 3) -> dict:
    """
    Cari candle historis yang paling mirip ke candle sekarang.
    Sumber data: DB (utama, lebih banyak) → CSV (fallback).

    Returns dict:
      bias       : "BUY" | "SELL" | "NEUTRAL"
      win_rate   : float (0-100)
      buy_count  : int
      sell_count : int
      total      : int
      avg_dist   : float
      matches    : list of dicts
      source     : "DB" | "CSV"
    """
    df = _load_df(symbol, timeframe)
    source = "DB" if df is not None and not df.empty else "CSV"

    if df is None or df.empty:
        return {"bias": "NEUTRAL", "win_rate": 0, "buy_count": 0,
                "sell_count": 0, "total": 0, "avg_dist": 99, "matches": [],
                "source": "none"}

    # Butuh minimal top_k + lookahead baris
    if len(df) < top_k + lookahead + 5:
        return {"bias": "NEUTRAL", "win_rate": 0, "buy_count": 0,
                "sell_count": 0, "total": 0, "avg_dist": 99, "matches": [],
                "source": source}

    cur_feat = _candle_features(current_row)

    # Hitung jarak ke setiap candle (kecuali lookahead terakhir — belum ada outcome)
    history = df.iloc[:-(lookahead)].copy().reset_index(drop=True)

    dists = []
    for i, row in history.iterrows():
        feat = _candle_features(row)
        diff = (cur_feat - feat) * _WEIGHTS
        dist = float(np.sqrt((diff ** 2).sum()))
        dists.append((i, dist))

    dists.sort(key=lambda x: x[1])
    top_matches = dists[:top_k]

    buy_count  = 0
    sell_count = 0
    matches    = []

    for idx, dist in top_matches:
        # Outcome: lihat close candle ke-lookahead vs close candle saat ini
        future_idx = idx + lookahead
        if future_idx >= len(df):
            continue

        close_now    = float(df.iloc[idx]["close"])
        close_future = float(df.iloc[future_idx]["close"])

        if close_now == 0:
            continue

        outcome = "BUY" if close_future > close_now else "SELL"
        pct_chg = round((close_future - close_now) / close_now * 100, 3)

        if outcome == "BUY":
            buy_count += 1
        else:
            sell_count += 1

        matches.append({
            "time":    str(df.iloc[idx]["time"])[:16],
            "candle":  str(df.iloc[idx]["candle"]),
            "rsi":     float(df.iloc[idx]["rsi"]),
            "outcome": outcome,
            "pct":     pct_chg,
            "dist":    round(dist, 3),
        })

    total = buy_count + sell_count
    if total == 0:
        return {"bias": "NEUTRAL", "win_rate": 0, "buy_count": 0,
                "sell_count": 0, "total": 0, "avg_dist": 99, "matches": []}

    buy_rate  = buy_count / total * 100
    sell_rate = sell_count / total * 100
    avg_dist  = round(sum(m["dist"] for m in matches) / len(matches), 3)

    if buy_rate >= 65:
        bias = "BUY"
    elif sell_rate >= 65:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    win_rate = buy_rate if bias == "BUY" else sell_rate if bias == "SELL" else max(buy_rate, sell_rate)

    return {
        "bias":       bias,
        "win_rate":   round(win_rate, 1),
        "buy_count":  buy_count,
        "sell_count": sell_count,
        "total":      total,
        "avg_dist":   avg_dist,
        "matches":    matches[:5],
        "source":     source,
    }


def backfill_candle_log(symbol: str, timeframe: str,
                         df_ind: pd.DataFrame) -> int:
    """
    Tulis semua candle dari df_ind ke log CSV sekaligus.
    Candle yang sudah ada (berdasarkan time) tidak di-duplikat.
    Returns jumlah candle baru yang ditulis.
    """
    if df_ind is None or df_ind.empty:
        return 0

    path = _log_path(symbol, timeframe)

    from config import EMA_SLOW, EMA_TREND

    records = []
    for ts, row in df_ind.iterrows():
        open_ = float(row.get("Open",  0))
        high  = float(row.get("High",  0))
        low   = float(row.get("Low",   0))
        close = float(row.get("Close", 0))

        candle_dir = "BULLISH" if close > open_ else "BEARISH" if close < open_ else "DOJI"
        full_range = high - low if high != low else 0.0001
        body_size  = round(abs(close - open_), 5)
        wick_up    = round(high - max(open_, close), 5)
        wick_down  = round(min(open_, close) - low, 5)

        records.append({
            "time":      str(ts)[:19],
            "open":      round(open_, 5),
            "high":      round(high,  5),
            "low":       round(low,   5),
            "close":     round(close, 5),
            "candle":    candle_dir,
            "body":      body_size,
            "wick_up":   wick_up,
            "wick_down": wick_down,
            "pattern":   str(row.get("candle_name", "")),
            "rsi":       round(float(row.get("rsi",  50)), 2),
            "ema20":     round(float(row.get(f"ema_{EMA_SLOW}",  close)), 5),
            "ema50":     round(float(row.get(f"ema_{EMA_TREND}", close)), 5),
            "macd":      round(float(row.get("macd", 0)), 5),
            "histogram": round(float(row.get("histogram", 0)), 5),
            "adx":       round(float(row.get("adx",  0)), 2),
            "atr":       round(float(row.get("atr",  0)), 5),
            "signal":    "",
            "score":     "",
            "sl":        "",
            "tp":        "",
            "logged_at": str(ts)[:19],
        })

    df_new = pd.DataFrame(records)

    if os.path.exists(path):
        existing  = pd.read_csv(path)
        known     = set(existing["time"].astype(str))
        df_new    = df_new[~df_new["time"].isin(known)]
        if df_new.empty:
            return 0
        df_out    = pd.concat([existing, df_new], ignore_index=True)
        df_out.sort_values("time", inplace=True)
        df_out.to_csv(path, index=False)
    else:
        df_new.sort_values("time", inplace=True)
        df_new.to_csv(path, index=False)

    return len(df_new)


def update_signal_outcomes(symbol: str, timeframe: str,
                           df_ind: "pd.DataFrame",
                           lookahead: int = 3) -> int:
    """
    Perbarui kolom outcome/outcome_pct untuk baris yang sudah ada signal-nya
    tapi belum ada outcome (karena saat itu belum ada cukup candle ke depan).

    outcome     : WIN | LOSS | FLAT
    outcome_pct : % perubahan harga setelah lookahead candle

    Dipanggil setiap siklus — otomatis update baris lama yang sudah bisa dievaluasi.
    Returns: jumlah baris yang diupdate.
    """
    # Load dari DB (utama) atau CSV (fallback)
    df = _load_df(symbol, timeframe)
    use_db = False
    try:
        from app.services.db_logger import load_candle_logs_df as _ldf
        _test = _ldf(symbol, timeframe, limit=1)
        use_db = not _test.empty
    except Exception:
        pass

    path = _log_path(symbol, timeframe)
    if df is None or df.empty:
        if not os.path.exists(path):
            return 0
        df = pd.read_csv(path)
        use_db = False

    if len(df) < lookahead + 2:
        return 0

    if "outcome" not in df.columns:
        df["outcome"]     = ""
        df["outcome_pct"] = ""

    if df_ind is not None and not df_ind.empty:
        price_map = {str(ts)[:19]: float(r["Close"])
                     for ts, r in df_ind.iterrows()}
    else:
        price_map = {}

    updated_csv = 0
    db_updates  = []
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():
        sig = str(row.get("signal", "")).strip()
        if sig not in ("BUY", "SELL"):
            continue
        if str(row.get("outcome", "")).strip() not in ("", "nan"):
            continue
        if i + lookahead >= len(df):
            continue

        close_now = float(row.get("close", 0) or 0)
        future_time  = str(df.iloc[i + lookahead]["time"])[:19]
        close_future = price_map.get(future_time,
                       float(df.iloc[i + lookahead]["close"] or 0))

        if close_now == 0:
            continue

        pct = round((close_future - close_now) / close_now * 100, 4)
        outcome = ("WIN"  if (sig == "BUY"  and pct >  0.005) else
                   "LOSS" if (sig == "BUY"  and pct < -0.005) else
                   "WIN"  if (sig == "SELL" and pct < -0.005) else
                   "LOSS" if (sig == "SELL" and pct >  0.005) else "FLAT")

        df.at[i, "outcome"]     = outcome
        df.at[i, "outcome_pct"] = pct
        updated_csv += 1

        # Kumpulkan untuk update DB
        try:
            import pandas as _pd
            ts = _pd.Timestamp(str(row.get("time", ""))[:19]).tz_localize("UTC")
            db_updates.append({"timestamp": ts, "outcome": outcome, "outcome_pct": pct})
        except Exception:
            pass

    # Update CSV
    if updated_csv > 0 and os.path.exists(path):
        df.to_csv(path, index=False)

    # Update DB
    if db_updates and use_db:
        try:
            from app.services.db_logger import update_outcomes_in_db
            update_outcomes_in_db(symbol, timeframe, db_updates)
        except Exception:
            pass

    return updated_csv


def get_signal_accuracy(symbol: str, timeframe: str,
                        current_row: "pd.Series",
                        signal_dir: str,
                        top_k: int = 20) -> dict:
    """
    Cari candle historis yang paling mirip DAN punya signal yang sama,
    hitung akurasi (WIN rate) dari signal tersebut.

    Returns dict:
      accuracy   : float (0-100) — % WIN dari signal serupa
      total      : int — jumlah sample
      win        : int
      loss       : int
      avg_pct    : float — rata-rata % keuntungan/kerugian
      reliable   : bool — True kalau sample >= 5 dan accuracy >= 55
    """
    df = _load_df(symbol, timeframe)
    if df is None or df.empty:
        return {"accuracy": 0, "total": 0, "reliable": False}

    # Filter: hanya baris yg punya signal sama DAN sudah ada outcome
    df_sig = df[
        (df["signal"] == signal_dir) &
        (df["outcome"].isin(["WIN", "LOSS", "FLAT"]))
    ].copy()

    if len(df_sig) < 3:
        return {"accuracy": 0, "total": len(df_sig), "reliable": False}

    cur_feat = _candle_features(current_row)

    dists = []
    for i, row in df_sig.iterrows():
        feat = _candle_features(row)
        diff = (cur_feat - feat) * _WEIGHTS
        dist = float(np.sqrt((diff ** 2).sum()))
        dists.append((i, dist))

    dists.sort(key=lambda x: x[1])
    top_idx = [i for i, _ in dists[:top_k]]
    subset  = df_sig.loc[df_sig.index.isin(top_idx)]

    win  = int((subset["outcome"] == "WIN").sum())
    loss = int((subset["outcome"] == "LOSS").sum())
    flat = int((subset["outcome"] == "FLAT").sum())
    total = win + loss + flat

    if total == 0:
        return {"accuracy": 0, "total": 0, "reliable": False}

    accuracy = round(win / total * 100, 1)

    pcts = []
    for v in subset["outcome_pct"]:
        try:
            pcts.append(float(v))
        except (ValueError, TypeError):
            pass
    avg_pct = round(sum(pcts) / len(pcts), 4) if pcts else 0

    return {
        "accuracy":  accuracy,
        "total":     total,
        "win":       win,
        "loss":      loss,
        "flat":      flat,
        "avg_pct":   avg_pct,
        "reliable":  total >= 5 and accuracy >= 55,
    }


def print_similar_report(result: dict) -> None:
    if not result or result["total"] == 0:
        return

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    bias = result["bias"]
    bc   = GREEN if bias == "BUY" else RED if bias == "SELL" else YELLOW
    wr   = result["win_rate"]

    src_tag = f"  {DIM}[{result.get('source','?').upper()}]{RESET}"
    print(f"\n  {BOLD}[CANDLE MEMORY]{RESET}{src_tag}  "
          f"mirip ditemukan: {result['total']}  |  "
          f"BUY {result['buy_count']}  SELL {result['sell_count']}  |  "
          f"bias: {bc}{BOLD}{bias}{RESET}  ({wr:.0f}%)")

    for m in result["matches"]:
        oc = GREEN if m["outcome"] == "BUY" else RED
        print(f"    {DIM}{m['time']}  rsi:{m['rsi']:>5.1f}  "
              f"{m['candle'][:4]}  dist:{m['dist']:.2f}{RESET}  "
              f"→ {oc}{m['outcome']}{RESET} {m['pct']:+.2f}%")


def print_recent(symbol: str, timeframe: str, n: int = 0) -> None:
    path = _log_path(symbol, timeframe)
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    if n and n > 0:
        df = df.tail(n)
    if df.empty:
        return

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    label = f"last {n}" if n and n > 0 else "all"
    print(f"\n  {BOLD}--- CANDLE LOG — {symbol} {timeframe} ({label}) ---{RESET}")
    print(f"  {'Time':<18} {'C':>7} {'Body':>7} {'WickU':>6} {'WickD':>6} "
          f"{'RSI':>5} {'EMA20>50':>8} {'Signal':>6}")
    print(f"  {'-'*72}")

    for _, r in df.iterrows():
        dc        = GREEN if r["candle"] == "BULLISH" else RED if r["candle"] == "BEARISH" else YELLOW
        sc        = GREEN if r["signal"] == "BUY" else RED if r["signal"] == "SELL" else RESET
        ema_cross = "↑" if float(r["ema20"]) > float(r["ema50"]) else "↓"
        ema_c     = GREEN if ema_cross == "↑" else RED
        pat       = str(r.get("pattern", "")).replace("↑","").replace("↓","")[:12]

        print(f"  {str(r['time']):<18} "
              f"{dc}{r['candle'][0]:>1}{RESET} "
              f"{r['close']:>8.2f} "
              f"{r['body']:>7.2f} "
              f"{r['wick_up']:>6.2f} "
              f"{r['wick_down']:>6.2f} "
              f"{float(r['rsi']):>5.1f} "
              f"{ema_c}{ema_cross:>8}{RESET} "
              f"{sc}{str(r['signal']):>6}{RESET}"
              + (f"  {CYAN}{pat}{RESET}" if pat and pat != "None" else ""))

    print()


def print_signal_candles(symbol: str, timeframe: str, n: int = 10) -> None:
    """Tampilkan hanya candle yang sinyalnya BUY atau SELL (relevan dengan indikator)."""
    path = _log_path(symbol, timeframe)
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    # Filter hanya BUY/SELL
    df_sig = df[df["signal"].isin(["BUY", "SELL"])]
    if df_sig.empty:
        return

    # Ambil N terbaru
    df_sig = df_sig.tail(n)

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    print(f"\n  {BOLD}--- SINYAL AKTIF — {symbol} {timeframe} (last {n}) ---{RESET}")
    print(f"  {'Time':<18} {'C':>1}  {'Close':>8} {'Body':>7} {'WickU':>6} {'WickD':>6} "
          f"{'RSI':>5} {'EMA':>4} {'ADX':>5} {'Signal':>6}  Pattern")
    print(f"  {'-'*80}")

    for _, r in df_sig.iterrows():
        dc        = GREEN if r["candle"] == "BULLISH" else RED if r["candle"] == "BEARISH" else YELLOW
        sc        = GREEN if r["signal"] == "BUY" else RED
        ema_cross = "↑" if float(r["ema20"]) > float(r["ema50"]) else "↓"
        ema_c     = GREEN if ema_cross == "↑" else RED
        pat       = str(r.get("pattern", "")).replace("↑","").replace("↓","")[:14]
        adx       = float(r.get("adx", 0))

        print(f"  {str(r['time']):<18} "
              f"{dc}{r['candle'][0]}{RESET}  "
              f"{r['close']:>8.2f} "
              f"{r['body']:>7.2f} "
              f"{r['wick_up']:>6.2f} "
              f"{r['wick_down']:>6.2f} "
              f"{float(r['rsi']):>5.1f} "
              f"{ema_c}{ema_cross:>4}{RESET} "
              f"{adx:>5.1f} "
              f"{sc}{BOLD}{str(r['signal']):>6}{RESET}"
              + (f"  {CYAN}{pat}{RESET}" if pat and pat != "None" else ""))
