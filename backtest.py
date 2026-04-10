#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------

def run_backtest(period: str = "60d",
                 verbose: bool = False,
                 save_csv: bool = False) -> dict:
    """
    Jalankan backtest walk-forward pada data historis XAUUSD 5m.

    Session filter dimatikan agar semua candle historis dievaluasi.
    Ini lebih konservatif: ada sinyal off-session yang akan di-reject live.
    """
    import config as cfg
    _orig_session = cfg.SESSION_FILTER
    cfg.SESSION_FILTER = False       # matikan untuk backtest (historical candle)

    try:
        return _run(period=period, verbose=verbose, save_csv=save_csv)
    finally:
        cfg.SESSION_FILTER = _orig_session


# ---------------------------------------------
# INTERNAL RUNNER
# ---------------------------------------------

def _run(period: str, verbose: bool, save_csv: bool) -> dict:
    from app.engine.signals.indicators import add_all_indicators
    from app.engine.signals.signals import generate_signal
    from config import ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP, MIN_SL_PIPS

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    print(f"\n  {BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}  BACKTEST PROFESIONAL — XAUUSD 5m  ({period}){RESET}")
    print(f"  {BOLD}{'='*60}{RESET}")
    print(f"  {DIM}Session filter: OFF (evaluasi semua jam historis){RESET}")

    # -- 1. Load data --------------------------------------------------
    print("  Loading data...", end="", flush=True)

    _csv = os.path.join(os.path.dirname(__file__), "data", "history", "XAUUSD_5m.csv")
    if os.path.exists(_csv):
        df_raw = pd.read_csv(_csv)
        df_raw = df_raw.set_index(df_raw.columns[0])
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw.columns = [c.strip() for c in df_raw.columns]
    else:
        try:
            import yfinance as yf
            df_raw = yf.download("GC=F", period=period, interval="5m",
                                 progress=False, auto_adjust=True)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            print(f" GAGAL download: {e}")
            return {}

    if df_raw is None or df_raw.empty:
        print(" GAGAL — tidak ada data")
        return {}
    print(f" OK ({len(df_raw)} candle)", flush=True)

    print("  Adding indicators...", end="", flush=True)
    df_full = add_all_indicators(df_raw)
    df_full = df_full.dropna(subset=["Close", "High", "Low"])
    print(f" OK ({len(df_full)} candle setelah dropna)", flush=True)

    LOOKBACK = 100   # candle minimum sebelum sinyal mulai dinilai

    if len(df_full) < LOOKBACK + 50:
        print(f"  Data terlalu sedikit ({len(df_full)} candle)")
        return {}

    # -- 2. Walk-forward simulation -------------------------------------
    print(f"  Simulasi walk-forward {len(df_full) - LOOKBACK} candle...", flush=True)

    trades      = []
    open_trade  = None
    equity_pips = 0.0
    equity_curve = [0.0]

    total_candles = len(df_full) - LOOKBACK
    last_pct      = -1

    for idx in range(LOOKBACK, len(df_full)):
        row = df_full.iloc[idx]
        hi  = float(row["High"])
        lo  = float(row["Low"])

        # -- Progress indicator ----------------------------------------
        pct = int((idx - LOOKBACK) / total_candles * 100)
        if pct != last_pct and pct % 10 == 0:
            print(f"    {pct}%... ", end="", flush=True)
            last_pct = pct

        # -- Cek SL/TP pada open trade ---------------------------------
        if open_trade:
            drn   = open_trade["direction"]
            sl    = open_trade["sl"]
            tp    = open_trade["tp"]
            entry = open_trade["entry"]
            sl_d  = open_trade["sl_dist"]
            tp_d  = open_trade["tp_dist"]
            atr   = open_trade["atr"]

            hit = None
            if drn == "BUY":
                if lo <= sl:       # SL kena (worst case: SL sebelum TP)
                    hit = ("LOSS", sl,  -sl_d)
                elif hi >= tp:     # TP kena
                    hit = ("WIN",  tp,  +tp_d)
            else:  # SELL
                if hi >= sl:       # SL kena
                    hit = ("LOSS", sl,  -sl_d)
                elif lo <= tp:     # TP kena
                    hit = ("WIN",  tp,  +tp_d)

            if hit:
                result, exit_px, pips = hit
                equity_pips += pips
                equity_curve.append(equity_pips)
                rec = {
                    "result":     result,
                    "direction":  drn,
                    "entry":      round(entry, 3),
                    "exit":       round(exit_px, 3),
                    "sl":         round(sl, 3),
                    "tp":         round(tp, 3),
                    "sl_dist":    round(sl_d, 3),
                    "tp_dist":    round(tp_d, 3),
                    "pips":       round(pips, 3),
                    "atr":        round(atr, 3),
                    "rr_planned": round(tp_d / sl_d, 2) if sl_d > 0 else 0,
                    "candle":     str(df_full.index[idx]),
                }
                trades.append(rec)
                open_trade = None

                if verbose:
                    c = GREEN if result == "WIN" else RED
                    print(f"\n    [{idx:5d}] {drn:<4} {c}{result:<4}{RESET} "
                          f"pips={pips:+7.2f}  "
                          f"entry={entry:.2f}  exit={exit_px:.2f}  "
                          f"atr={atr:.2f}")

        # -- Cari sinyal baru jika tidak ada open trade ----------------
        if not open_trade:
            window = df_full.iloc[max(0, idx - LOOKBACK) : idx + 1]
            try:
                sig = generate_signal(window)
            except Exception:
                continue

            drn = sig.get("direction")
            if drn not in ("BUY", "SELL"):
                continue

            close = float(row["Close"])
            atr   = sig.get("atr") or float(row.get("atr", close * 0.001))
            sl_d  = max(float(atr) * ATR_MULTIPLIER_SL, MIN_SL_PIPS)
            tp_d  = float(atr) * ATR_MULTIPLIER_TP

            if drn == "BUY":
                sl = close - sl_d
                tp = close + tp_d
            else:
                sl = close + sl_d
                tp = close - tp_d

            open_trade = {
                "direction": drn,
                "entry":     close,
                "sl":        sl,
                "tp":        tp,
                "sl_dist":   sl_d,
                "tp_dist":   tp_d,
                "atr":       float(atr),
                "entry_idx": idx,
            }

    print()  # newline setelah progress

    # -- Close sisa open trade di harga terakhir ------------------------
    if open_trade and len(df_full) > 0:
        last_close = float(df_full.iloc[-1]["Close"])
        drn  = open_trade["direction"]
        pips = (last_close - open_trade["entry"]) if drn == "BUY" \
               else (open_trade["entry"] - last_close)
        equity_pips += pips
        equity_curve.append(equity_pips)
        trades.append({
            "result":    "OPEN_CLOSE",
            "direction": drn,
            "entry":     open_trade["entry"],
            "exit":      last_close,
            "sl":        open_trade["sl"],
            "tp":        open_trade["tp"],
            "sl_dist":   open_trade["sl_dist"],
            "tp_dist":   open_trade["tp_dist"],
            "pips":      round(pips, 3),
            "atr":       open_trade["atr"],
            "rr_planned": 0,
            "candle":    str(df_full.index[-1]),
        })

    if not trades:
        print(f"  {YELLOW}Tidak ada trade — sinyal terlalu selektif "
              f"(coba naikkan ENTRY_ZONE_PCT atau turunkan MIN_SIGNAL_SCORE){RESET}")
        return {}

    # -- 3. Hitung metrik -----------------------------------------------
    closed        = [t for t in trades if t["result"] in ("WIN", "LOSS")]
    wins          = [t for t in closed if t["result"] == "WIN"]
    losses        = [t for t in closed if t["result"] == "LOSS"]
    n_total       = len(closed)
    n_wins        = len(wins)
    n_loss        = len(losses)
    win_rate      = (n_wins / n_total * 100) if n_total > 0 else 0
    total_profit  = sum(t["pips"] for t in wins)
    total_loss_a  = abs(sum(t["pips"] for t in losses))
    profit_factor = (total_profit / total_loss_a) if total_loss_a > 0 else float("inf")
    avg_win       = (total_profit / n_wins)   if n_wins  > 0 else 0
    avg_loss      = (total_loss_a / n_loss)   if n_loss  > 0 else 0
    avg_rr        = (avg_win / avg_loss)      if avg_loss > 0 else 0
    total_pips    = sum(t["pips"] for t in closed)

    # Max Drawdown
    eq      = np.array(equity_curve)
    peak    = np.maximum.accumulate(eq)
    max_dd  = float((eq - peak).min())

    # Streak
    max_cw = max_cl = cw = cl = 0
    for t in closed:
        if t["result"] == "WIN":
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)

    # -- 4. Cetak laporan -----------------------------------------------
    pf_disp = f"{profit_factor:.2f}" if profit_factor != float("inf") else "inf"
    wr_c  = GREEN  if win_rate >= 55    else (YELLOW if win_rate >= 45    else RED)
    pf_c  = GREEN  if profit_factor >= 1.5 else (YELLOW if profit_factor >= 1.0 else RED)
    dd_c  = GREEN  if abs(max_dd) < 100 else (YELLOW if abs(max_dd) < 200 else RED)
    tp_c  = GREEN  if total_pips > 0    else RED
    rr_c  = GREEN  if avg_rr >= 3.0     else (YELLOW if avg_rr >= 1.5     else RED)

    print(f"\n  {BOLD}HASIL BACKTEST:{RESET}")
    print(f"  {'-'*56}")
    print(f"  {'Total Trade':<24}: {BOLD}{n_total}{RESET}  "
          f"({GREEN}{n_wins} WIN{RESET} | {RED}{n_loss} LOSS{RESET})")
    print(f"  {'Win Rate':<24}: {wr_c}{BOLD}{win_rate:.1f}%{RESET}  "
          f"{'[OK] OK' if win_rate >= 50 else '[!!] Perlu ditingkatkan'}")
    print(f"  {'Profit Factor':<24}: {pf_c}{BOLD}{pf_disp}{RESET}  "
          f"{'[OK] Excellent' if profit_factor >= 2.0 else ('[OK] Good' if profit_factor >= 1.5 else ('~ Cukup' if profit_factor >= 1.0 else '[!!] Rugi'))}")
    print(f"  {'Max Drawdown':<24}: {dd_c}{max_dd:+.2f} pips{RESET}")
    print(f"  {'Total Pips':<24}: {tp_c}{total_pips:+.2f}{RESET}")
    print(f"  {'Avg Win / Avg Loss':<24}: {GREEN}+{avg_win:.2f}{RESET} / {RED}-{avg_loss:.2f}{RESET} pips")
    print(f"  {'Avg R (real RR)':<24}: {rr_c}{BOLD}{avg_rr:.2f}x{RESET}")
    print(f"  {'Win / Loss Streak':<24}: max {max_cw}W / {max_cl}L beruntun")
    print(f"  {'-'*56}")

    # Grade
    pts = 0
    if win_rate >= 55:         pts += 2
    elif win_rate >= 45:       pts += 1
    if profit_factor >= 1.5:   pts += 2
    elif profit_factor >= 1.0: pts += 1
    if total_pips > 0:         pts += 1
    if avg_rr >= 2.0:          pts += 1
    grade   = "A" if pts >= 5 else "B" if pts >= 3 else "C" if pts >= 2 else "D"
    grade_c = GREEN if grade in ("A", "B") else (YELLOW if grade == "C" else RED)
    print(f"  {'Grade':<24}: {grade_c}{BOLD}{grade}{RESET}  ({pts}/6 pts)")
    print(f"  {'='*56}")

    # Target check
    targets = [
        ("Win Rate >= 50%",     win_rate >= 50,     f"{win_rate:.1f}%"),
        ("Profit Factor > 1.5", profit_factor >= 1.5, pf_disp),
        ("Drawdown < 200",      abs(max_dd) < 200,  f"{max_dd:.2f}"),
        ("Total Pips > 0",      total_pips > 0,     f"{total_pips:+.2f}"),
    ]
    print(f"\n  {BOLD}TARGET CHECK:{RESET}")
    for label, ok, val in targets:
        icon = "[OK]" if ok else "[!!]"
        c    = GREEN if ok else RED
        print(f"    {c}{icon}{RESET} {label:<36} {DIM}[{val}]{RESET}")

    # -- 5. Simpan CSV jika diminta -------------------------------------
    if save_csv and trades:
        _logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(_logs_dir, exist_ok=True)
        _out = os.path.join(_logs_dir, "backtest_trades.csv")
        pd.DataFrame(trades).to_csv(_out, index=False)
        print(f"\n  Trade log disimpan: {_out}")

    print()

    return {
        "n_total":          n_total,
        "n_wins":           n_wins,
        "n_loss":           n_loss,
        "win_rate":         round(win_rate, 1),
        "profit_factor":    round(profit_factor, 2) if profit_factor != float("inf") else 999,
        "max_drawdown":     round(max_dd, 2),
        "total_pips":       round(total_pips, 2),
        "avg_win_pips":     round(avg_win, 2),
        "avg_loss_pips":    round(avg_loss, 2),
        "avg_rr":           round(avg_rr, 2),
        "max_win_streak":   max_cw,
        "max_loss_streak":  max_cl,
        "grade":            grade,
    }


# ---------------------------------------------
# CLI
# ---------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest profesional XAUUSD trading bot")
    parser.add_argument("--period",  default="60d",
                        help="Periode data (default: 60d). Contoh: 30d, 90d")
    parser.add_argument("--verbose", action="store_true",
                        help="Tampilkan setiap trade secara detail")
    parser.add_argument("--csv",     action="store_true",
                        help="Simpan trade log ke logs/backtest_trades.csv")
    args = parser.parse_args()

    run_backtest(period=args.period, verbose=args.verbose, save_csv=args.csv)
