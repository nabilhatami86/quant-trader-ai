import os
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
            existing.iloc[-1] = df_row.iloc[0]
            existing.to_csv(path, index=False)
        else:
            df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)


def print_recent(symbol: str, timeframe: str, n: int = 10) -> None:
    path = _log_path(symbol, timeframe)
    if not os.path.exists(path):
        return

    df = pd.read_csv(path).tail(n)
    if df.empty:
        return

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    print(f"\n  {BOLD}--- CANDLE LOG — {symbol} {timeframe} (last {n}) ---{RESET}")
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
