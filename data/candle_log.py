import os
import numpy as np
import pandas as pd
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _log_path(symbol: str, timeframe: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"candles_{symbol}_{timeframe}.csv")


def log_candle(symbol: str, timeframe: str, df_ind: "pd.DataFrame",
               signal: dict = None) -> None:
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
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df_row = pd.DataFrame([record])

    if os.path.exists(path):
        existing = pd.read_csv(path)
        if len(existing) and existing["time"].iloc[-1] == record["time"]:
            existing = existing.iloc[:-1]
            existing = pd.concat([existing, df_row], ignore_index=True)
            existing.to_csv(path, index=False)
        else:
            df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)


def _candle_features(row: pd.Series) -> np.ndarray:
    rng   = float(row.get("body", 0)) + float(row.get("wick_up", 0)) + float(row.get("wick_down", 0))
    rng   = rng if rng > 0 else 1.0
    body  = float(row.get("body", 0))
    wu    = float(row.get("wick_up", 0))
    wd    = float(row.get("wick_down", 0))
    rsi   = float(row.get("rsi", 50))
    adx   = float(row.get("adx", 0))
    ema20 = float(row.get("ema20", 1))
    ema50 = float(row.get("ema50", 1))
    hist  = float(row.get("histogram", 0))
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


def find_similar_candles(symbol: str, timeframe: str,
                          current_row: pd.Series,
                          top_k: int = 15,
                          lookahead: int = 3) -> dict:
    """
    Cari candle historis yang paling mirip ke candle sekarang,
    lihat apa yang terjadi setelahnya, dan kembalikan bias + win rate.

    Returns dict:
      bias       : "BUY" | "SELL" | "NEUTRAL"
      win_rate   : float (0-100)
      buy_count  : int
      sell_count : int
      total      : int
      avg_dist   : float  (jarak rata-rata, makin kecil makin mirip)
      matches    : list of dicts
    """
    path = _log_path(symbol, timeframe)
    if not os.path.exists(path):
        return {"bias": "NEUTRAL", "win_rate": 0, "buy_count": 0,
                "sell_count": 0, "total": 0, "avg_dist": 99, "matches": []}

    df = pd.read_csv(path)
    # Butuh minimal top_k + lookahead baris
    if len(df) < top_k + lookahead + 5:
        return {"bias": "NEUTRAL", "win_rate": 0, "buy_count": 0,
                "sell_count": 0, "total": 0, "avg_dist": 99, "matches": []}

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

    print(f"\n  {BOLD}[CANDLE MEMORY]{RESET}  "
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
