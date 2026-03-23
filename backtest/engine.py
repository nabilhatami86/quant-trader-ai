"""
Backtest Engine - Simulasi trading historis
"""
import pandas as pd
import numpy as np
from config import *
from analysis.indicators import add_all_indicators
from analysis.signals import generate_signal


def run_backtest(df_raw: pd.DataFrame, symbol: str = "") -> dict:
    """
    Jalankan backtest pada DataFrame OHLCV.
    Returns dict hasil backtest lengkap.
    """
    df = add_all_indicators(df_raw, max_rows=0)  # backtest butuh semua data
    trades = []
    capital = float(INITIAL_CAPITAL)
    in_trade = False
    trade_entry = None

    for i in range(50, len(df)):
        window = df.iloc[:i + 1]
        sig = generate_signal(window)

        if not in_trade:
            if sig["direction"] in ("BUY", "SELL"):
                trade_entry = {
                    "open_idx":   df.index[i],
                    "direction":  sig["direction"],
                    "entry":      sig["close"],
                    "sl":         sig["sl"],
                    "tp":         sig["tp"],
                    "score":      sig["score"],
                }
                in_trade = True
        else:
            cur_high  = float(df.iloc[i]["High"])
            cur_low   = float(df.iloc[i]["Low"])
            cur_close = float(df.iloc[i]["Close"])
            direction = trade_entry["direction"]
            sl = trade_entry["sl"]
            tp = trade_entry["tp"]
            entry = trade_entry["entry"]

            hit_sl = hit_tp = False
            if direction == "BUY":
                if cur_low  <= sl: hit_sl = True
                if cur_high >= tp: hit_tp = True
            else:
                if cur_high >= sl: hit_sl = True
                if cur_low  <= tp: hit_tp = True

            if hit_tp or hit_sl:
                exit_price = tp if hit_tp else sl
                pnl_pips   = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
                pnl_pct    = pnl_pips / entry * 100
                result     = "WIN" if hit_tp else "LOSS"

                # Simulasi PnL sederhana: 1% risk per trade
                risk_amt = capital * 0.01
                rr       = ATR_MULTIPLIER_TP / ATR_MULTIPLIER_SL
                pnl_usd  = risk_amt * rr if hit_tp else -risk_amt
                capital += pnl_usd

                trades.append({
                    "open":       trade_entry["open_idx"],
                    "close":      df.index[i],
                    "direction":  direction,
                    "entry":      entry,
                    "exit":       exit_price,
                    "sl":         sl,
                    "tp":         tp,
                    "result":     result,
                    "pnl_pips":   round(pnl_pips * 10000, 1),
                    "pnl_pct":    round(pnl_pct, 3),
                    "pnl_usd":    round(pnl_usd, 2),
                    "capital":    round(capital, 2),
                    "score":      trade_entry["score"],
                })
                in_trade = False
                trade_entry = None

    # ─── Statistik ───────────────────────────────────────────
    if not trades:
        return {"error": "Tidak ada trade yang tereksekusi"}

    df_trades = pd.DataFrame(trades)
    wins       = df_trades[df_trades["result"] == "WIN"]
    losses     = df_trades[df_trades["result"] == "LOSS"]
    win_rate   = len(wins) / len(df_trades) * 100
    total_pnl  = df_trades["pnl_usd"].sum()
    avg_win    = wins["pnl_usd"].mean() if len(wins) else 0
    avg_loss   = losses["pnl_usd"].mean() if len(losses) else 0
    profit_factor = abs(wins["pnl_usd"].sum() / losses["pnl_usd"].sum()) if len(losses) else float("inf")

    # Max drawdown
    capital_curve = [INITIAL_CAPITAL] + list(df_trades["capital"])
    peak = INITIAL_CAPITAL
    max_dd = 0
    for c in capital_curve:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100
        if dd > max_dd:
            max_dd = dd

    buy_trades  = df_trades[df_trades["direction"] == "BUY"]
    sell_trades = df_trades[df_trades["direction"] == "SELL"]

    return {
        "symbol":         symbol,
        "total_trades":   len(df_trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(win_rate, 2),
        "total_pnl_usd":  round(total_pnl, 2),
        "profit_factor":  round(profit_factor, 2),
        "max_drawdown":   round(max_dd, 2),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "final_capital":  round(capital, 2),
        "return_pct":     round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
        "buy_trades":     len(buy_trades),
        "sell_trades":    len(sell_trades),
        "trades":         df_trades,
    }


def print_backtest_report(result: dict) -> None:
    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        return

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT - {result['symbol']}")
    print(sep)
    print(f"  Total Trades    : {result['total_trades']}")
    print(f"  Win / Loss      : {result['wins']} / {result['losses']}")
    print(f"  Win Rate        : {result['win_rate']}%")
    print(f"  Buy Trades      : {result['buy_trades']}")
    print(f"  Sell Trades     : {result['sell_trades']}")
    print(sep)
    print(f"  Total PnL       : ${result['total_pnl_usd']:+.2f}")
    print(f"  Return          : {result['return_pct']:+.2f}%")
    print(f"  Final Capital   : ${result['final_capital']:.2f}")
    print(f"  Profit Factor   : {result['profit_factor']:.2f}")
    print(f"  Max Drawdown    : {result['max_drawdown']}%")
    print(f"  Avg Win         : ${result['avg_win']:.2f}")
    print(f"  Avg Loss        : ${result['avg_loss']:.2f}")
    print(sep)
