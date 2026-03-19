import os
import pandas as pd
from datetime import datetime

JOURNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _journal_path(symbol: str, timeframe: str) -> str:
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    return os.path.join(JOURNAL_DIR, f"journal_{symbol}_{timeframe}.csv")


def log_entry(symbol: str, timeframe: str, ticket: int, direction: str,
              entry_price: float, sl: float, tp: float,
              lot: float, source: str, atr: float = 0.0) -> None:
    path = _journal_path(symbol, timeframe)
    row  = {
        "ticket":      ticket,
        "symbol":      symbol,
        "timeframe":   timeframe,
        "direction":   direction,
        "entry_price": entry_price,
        "sl":          sl,
        "tp":          tp,
        "lot":         lot,
        "source":      source,
        "atr":         atr,
        "entry_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time":   "",
        "exit_price":  "",
        "result":      "OPEN",
        "pnl":         "",
        "pips":        "",
    }
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def log_exit(symbol: str, timeframe: str, ticket: int,
             exit_price: float, pnl: float) -> None:
    path = _journal_path(symbol, timeframe)
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    mask = (df["ticket"] == ticket) & (df["result"] == "OPEN")
    if not mask.any():
        return

    idx = df[mask].index[0]
    entry_price = float(df.at[idx, "entry_price"])
    direction   = df.at[idx, "direction"]
    sl          = float(df.at[idx, "sl"]) if df.at[idx, "sl"] else 0
    tp          = float(df.at[idx, "tp"]) if df.at[idx, "tp"] else 0

    if direction == "BUY":
        pips = exit_price - entry_price
    else:
        pips = entry_price - exit_price

    if tp and sl:
        if direction == "BUY":
            result = "WIN" if exit_price >= tp * 0.995 else ("LOSS" if exit_price <= sl * 1.005 else "MANUAL")
        else:
            result = "WIN" if exit_price <= tp * 1.005 else ("LOSS" if exit_price >= sl * 0.995 else "MANUAL")
    else:
        result = "WIN" if pnl > 0 else "LOSS"

    df.at[idx, "exit_time"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.at[idx, "exit_price"] = exit_price
    df.at[idx, "result"]     = result
    df.at[idx, "pnl"]        = round(pnl, 2)
    df.at[idx, "pips"]       = round(pips, 5)
    df.to_csv(path, index=False)


def get_stats(symbol: str, timeframe: str) -> dict:
    path = _journal_path(symbol, timeframe)
    if not os.path.exists(path):
        return {}

    df     = pd.read_csv(path)
    closed = df[df["result"].isin(["WIN", "LOSS", "MANUAL"])]
    if closed.empty:
        return {"total": 0}

    wins   = (closed["result"] == "WIN").sum()
    losses = (closed["result"] == "LOSS").sum()
    total  = len(closed)
    pnl_s  = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
    total_pnl  = pnl_s.sum()
    avg_win    = pnl_s[pnl_s > 0].mean() if (pnl_s > 0).any() else 0
    avg_loss   = pnl_s[pnl_s < 0].mean() if (pnl_s < 0).any() else 0

    return {
        "total":     total,
        "wins":      int(wins),
        "losses":    int(losses),
        "win_rate":  round(wins / total * 100, 1) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win":   round(avg_win, 2),
        "avg_loss":  round(avg_loss, 2),
        "open":      int((df["result"] == "OPEN").sum()),
    }


def print_stats(symbol: str, timeframe: str) -> None:
    s = get_stats(symbol, timeframe)
    if not s or s.get("total", 0) == 0:
        print("  [Journal] Belum ada trade tercatat.")
        return

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    wr_color = GREEN if s["win_rate"] >= 50 else RED
    pnl_color = GREEN if s["total_pnl"] >= 0 else RED

    print(f"\n  {'='*48}")
    print(f"  {BOLD}TRADE JOURNAL — {symbol} {timeframe}{RESET}")
    print(f"  {'='*48}")
    print(f"  Total Trade : {s['total']}  (Open: {s['open']})")
    print(f"  Win / Loss  : {GREEN}{s['wins']} WIN{RESET} / {RED}{s['losses']} LOSS{RESET}")
    print(f"  Win Rate    : {BOLD}{wr_color}{s['win_rate']}%{RESET}")
    print(f"  Total P&L   : {pnl_color}{BOLD}${s['total_pnl']:+.2f}{RESET}")
    print(f"  Avg Win     : {GREEN}${s['avg_win']:+.2f}{RESET}  |  "
          f"Avg Loss: {RED}${s['avg_loss']:+.2f}{RESET}")
    print(f"  {'='*48}")


def get_recent_trades(symbol: str, timeframe: str, n: int = 10) -> pd.DataFrame:
    path = _journal_path(symbol, timeframe)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.tail(n)
