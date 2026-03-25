"""
trade_journal.py — Simpan & tampilkan riwayat trade bot.

Storage: logs/trade_journal.csv (satu file terpusat, semua symbol)

Kolom CSV:
  ticket, symbol, timeframe, direction, entry_price, sl, tp,
  lot, source, atr, entry_time, exit_time, exit_price,
  result (WIN/LOSS/MANUAL/OPEN), pnl, pips, note

Fungsi utama:
  log_entry(symbol, tf, ticket, direction, entry_price, sl, tp, lot, source)
      Catat trade baru saat order dibuka. result="OPEN", pnl=0.

  log_exit(symbol, tf, ticket, exit_price, result, pnl, pips)
      Update baris trade saat posisi ditutup. Set result=WIN/LOSS.

  is_open(ticket) -> bool
      Cek apakah ticket masih berstatus OPEN di jurnal.

  get_stats(symbol, timeframe) -> dict
      Statistik: total, wins, losses, win_rate, total_pnl, avg_pnl,
                 max_win, max_loss, avg_rr, sumber sinyal breakdown.

  print_stats(symbol, timeframe)
      Tampilkan laporan lengkap di terminal termasuk tabel LOSS trades.

  get_recent_trades(symbol, n) -> DataFrame
      N trade terakhir yang sudah tutup (WIN/LOSS).

Catatan P&L:
  P&L diisi saat log_exit dipanggil dari sync_closed_positions() di
  mt5_connector.py. Net P&L = profit + swap + fee dari MT5 deal history.
  Key mapping menggunakan position_id (bukan order ticket) untuk akurasi.
"""
import os
import pandas as pd
from datetime import datetime

JOURNAL_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
JOURNAL_PATH = os.path.join(JOURNAL_DIR, "trade_journal.csv")

COLS = [
    "ticket", "symbol", "timeframe", "direction",
    "entry_price", "sl", "tp", "lot", "source", "atr",
    "entry_time", "exit_time", "exit_price",
    "result", "pnl", "pips", "note",
]


def _ensure_dir():
    os.makedirs(JOURNAL_DIR, exist_ok=True)


def log_entry(symbol: str, timeframe: str, ticket: int, direction: str,
              entry_price: float, sl: float, tp: float,
              lot: float, source: str, atr: float = 0.0) -> None:
    _ensure_dir()
    row = {
        "ticket":      ticket,
        "symbol":      symbol,
        "timeframe":   timeframe,
        "direction":   direction,
        "entry_price": round(entry_price, 5),
        "sl":          round(sl, 5) if sl else 0,
        "tp":          round(tp, 5) if tp else 0,
        "lot":         lot,
        "source":      source,
        "atr":         round(atr, 5),
        "entry_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time":   "",
        "exit_price":  "",
        "result":      "OPEN",
        "pnl":         "",
        "pips":        "",
        "note":        "",
    }
    df = pd.DataFrame([row])
    if os.path.exists(JOURNAL_PATH):
        df.to_csv(JOURNAL_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(JOURNAL_PATH, index=False)


def log_exit(symbol: str, timeframe: str, ticket: int,
             exit_price: float, pnl: float, note: str = "") -> None:
    if not os.path.exists(JOURNAL_PATH):
        return

    df = pd.read_csv(JOURNAL_PATH, dtype=str)   # baca semua sebagai string
    mask = (df["ticket"] == str(ticket)) & (df["result"] == "OPEN")
    if not mask.any():
        return

    idx         = df[mask].index[0]
    entry_price = float(df.at[idx, "entry_price"] or 0)
    direction   = df.at[idx, "direction"]
    sl_v        = float(df.at[idx, "sl"]  or 0)
    tp_v        = float(df.at[idx, "tp"]  or 0)

    pips = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)

    if tp_v and sl_v:
        if direction == "BUY":
            result = "WIN"  if exit_price >= tp_v * 0.995 else \
                     "LOSS" if exit_price <= sl_v * 1.005 else "MANUAL"
        else:
            result = "WIN"  if exit_price <= tp_v * 1.005 else \
                     "LOSS" if exit_price >= sl_v * 0.995 else "MANUAL"
    else:
        result = "WIN" if pnl > 0 else "LOSS"

    df.at[idx, "exit_time"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.at[idx, "exit_price"] = str(round(exit_price, 5))
    df.at[idx, "result"]     = result
    df.at[idx, "pnl"]        = str(round(pnl, 2))
    df.at[idx, "pips"]       = str(round(pips, 5))
    df.at[idx, "note"]       = note
    df.to_csv(JOURNAL_PATH, index=False)


def is_open(ticket: int) -> bool:
    """Cek apakah ticket masih berstatus OPEN di journal."""
    if not os.path.exists(JOURNAL_PATH):
        return False
    df = pd.read_csv(JOURNAL_PATH, dtype=str)
    return bool(((df["ticket"] == str(ticket)) & (df["result"] == "OPEN")).any())


def get_stats(symbol: str = "", timeframe: str = "") -> dict:
    if not os.path.exists(JOURNAL_PATH):
        return {}
    df     = pd.read_csv(JOURNAL_PATH)
    if symbol:
        df = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
    closed = df[df["result"].isin(["WIN", "LOSS", "MANUAL"])]
    if closed.empty:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "open": int((df["result"] == "OPEN").sum()),
        }

    wins       = int((closed["result"] == "WIN").sum())
    losses     = int((closed["result"] == "LOSS").sum())
    manuals    = int((closed["result"] == "MANUAL").sum())
    total      = len(closed)
    pnl_s      = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
    total_pnl  = pnl_s.sum()
    avg_win    = pnl_s[pnl_s > 0].mean()  if (pnl_s > 0).any() else 0
    avg_loss   = pnl_s[pnl_s < 0].mean()  if (pnl_s < 0).any() else 0
    max_win    = pnl_s.max() if not pnl_s.empty else 0
    max_loss   = pnl_s.min() if not pnl_s.empty else 0
    return {
        "total":     total,
        "wins":      wins,
        "losses":    losses,
        "manuals":   manuals,
        "win_rate":  round(wins / total * 100, 1) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win":   round(avg_win, 2),
        "avg_loss":  round(avg_loss, 2),
        "max_win":   round(max_win, 2),
        "max_loss":  round(max_loss, 2),
        "open":      int((df["result"] == "OPEN").sum()),
    }


def get_recent_trades(symbol: str = "", n: int = 10) -> pd.DataFrame:
    if not os.path.exists(JOURNAL_PATH):
        return pd.DataFrame()
    df = pd.read_csv(JOURNAL_PATH)
    if symbol:
        df = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
    return df.tail(n).reset_index(drop=True)


def _merge_with_db(symbol: str = "", limit: int = 30) -> pd.DataFrame:
    """
    Gabungkan data dari DB (lebih lengkap) + CSV journal (fallback/merge).
    DB-first: ambil semua dari DB, tambahkan dari CSV yang tidak ada di DB.
    """
    # ── DB ───────────────────────────────────────────────────────────
    df_db = pd.DataFrame()
    try:
        from services.db_logger import get_trade_journal_df
        df_db = get_trade_journal_df(symbol, limit)
    except Exception:
        pass

    # ── CSV ──────────────────────────────────────────────────────────
    df_csv = pd.DataFrame()
    if os.path.exists(JOURNAL_PATH):
        try:
            df_csv = pd.read_csv(JOURNAL_PATH)
            if symbol:
                df_csv = df_csv[df_csv["symbol"].astype(str).str.upper() == symbol.upper()]
        except Exception:
            pass

    if df_db.empty and df_csv.empty:
        return pd.DataFrame()
    if df_db.empty:
        return df_csv.tail(limit).reset_index(drop=True)
    if df_csv.empty:
        return df_db.head(limit).reset_index(drop=True)

    # Merge: gabungkan DB + CSV, deduplikasi per ticket, prioritaskan yang closed
    try:
        db_tickets = set(df_db["ticket"].astype(str).dropna())
        csv_extra  = df_csv[~df_csv["ticket"].astype(str).isin(db_tickets)]
        merged     = pd.concat([df_db, csv_extra], ignore_index=True)
        # Urutkan: closed (WIN/LOSS/MANUAL) duluan, lalu OPEN, tiap grup by entry_time desc
        merged["_sort_result"] = merged["result"].map(
            lambda r: 0 if r in ("WIN", "LOSS", "MANUAL") else 1
        )
        merged = merged.sort_values(["_sort_result", "entry_time"],
                                    ascending=[True, False],
                                    na_position="last").drop(columns=["_sort_result"])
        return merged.head(limit).reset_index(drop=True)
    except Exception:
        return df_db.head(limit)


def print_stats(symbol: str = "", timeframe: str = "") -> None:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"

    # ── Ambil data gabungan DB + CSV ──────────────────────────────────────
    all_trades = _merge_with_db(symbol, limit=50)

    if all_trades.empty:
        print(f"  [Journal] Belum ada trade tercatat.")
        return

    closed = all_trades[all_trades["result"].isin(["WIN", "LOSS", "MANUAL"])]
    open_c = all_trades[all_trades["result"] == "OPEN"]

    if closed.empty:
        print(f"  [Journal] {len(open_c)} posisi OPEN — belum ada yang ditutup.")
        return

    pnl_s    = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
    wins     = int((closed["result"] == "WIN").sum())
    losses   = int((closed["result"] == "LOSS").sum())
    manuals  = int((closed["result"] == "MANUAL").sum())
    total    = len(closed)
    tot_pnl  = round(pnl_s.sum(), 2)
    avg_win  = round(pnl_s[pnl_s > 0].mean(), 2) if (pnl_s > 0).any() else 0.0
    avg_loss = round(pnl_s[pnl_s < 0].mean(), 2) if (pnl_s < 0).any() else 0.0
    max_win  = round(pnl_s.max(), 2) if not pnl_s.empty else 0.0
    max_loss = round(pnl_s.min(), 2) if not pnl_s.empty else 0.0
    win_rate = round(wins / total * 100, 1) if total else 0

    lbl      = f"{symbol} {timeframe}".strip() or "Semua Symbol"
    wr_color = GREEN if win_rate >= 55 else (YELLOW if win_rate >= 45 else RED)
    pc_color = GREEN if tot_pnl >= 0 else RED
    sep52    = "=" * 56
    src_tag  = f"{DIM}[DB+CSV]{RESET}"

    print(f"\n  {sep52}")
    print(f"  {BOLD}TRADE JOURNAL — {lbl}{RESET}  {src_tag}")
    print(f"  {sep52}")
    print(f"  Total Trade : {BOLD}{total}{RESET}  (Open: {YELLOW}{len(open_c)}{RESET})")
    print(f"  Win / Loss  : {GREEN}{BOLD}{wins} WIN{RESET} / {RED}{BOLD}{losses} LOSS{RESET}"
          + (f" / {DIM}{manuals} MANUAL{RESET}" if manuals else ""))
    print(f"  Win Rate    : {wr_color}{BOLD}{win_rate}%{RESET}")
    print(f"  Total P&L   : {pc_color}{BOLD}${tot_pnl:+.2f}{RESET}")
    print(f"  Avg Win     : {GREEN}${avg_win:+.2f}{RESET}  |  "
          f"Avg Loss : {RED}${avg_loss:+.2f}{RESET}")
    print(f"  Best Trade  : {GREEN}${max_win:+.2f}{RESET}  |  "
          f"Worst    : {RED}${max_loss:+.2f}{RESET}")

    # ── Breakdown LOSS trades ─────────────────────────────────────────────
    loss_trades = closed[closed["result"] == "LOSS"].copy()
    loss_pnl    = pd.to_numeric(loss_trades["pnl"], errors="coerce").fillna(0)
    total_loss  = round(loss_pnl.sum(), 2)

    if not loss_trades.empty:
        print(f"\n  {RED}{BOLD}LOSS TRADES ({losses} trade) — Total Rugi: ${total_loss:+.2f}{RESET}")
        print(f"  {'-'*82}")
        print(f"  {'#Ticket':<13} {'Dir':<5} {'Entry':>10} {'Exit':>10} "
              f"{'Lot':>5}  {'P&L':>8}  {'Waktu Entry':<20}")
        print(f"  {'-'*82}")
        for _, row in loss_trades.iterrows():
            pnl_v = float(pd.to_numeric(row.get("pnl", ""), errors="coerce") or 0)
            ep    = pd.to_numeric(row.get("entry_price", 0), errors="coerce") or 0
            xp    = pd.to_numeric(row.get("exit_price", ""), errors="coerce")
            xp_s  = f"{float(xp):.2f}" if not pd.isna(xp) and float(xp) != 0 else "--"
            lot_v = pd.to_numeric(row.get("lot", 0), errors="coerce") or 0
            etime = str(row.get("entry_time", ""))[:19]
            dc    = GREEN if str(row.get("direction","")) == "BUY" else RED
            print(f"  #{str(row.get('ticket','--')):<12} "
                  f"{dc}{str(row.get('direction','--')):<4}{RESET} "
                  f"{float(ep):>10.2f} {xp_s:>10} "
                  f"{float(lot_v):>5.2f}  "
                  f"{RED}${pnl_v:>+7.2f}{RESET}  "
                  f"{DIM}{etime}{RESET}")

    # ── Tabel semua trade terakhir (max 10) ───────────────────────────────
    recent_closed = closed.sort_values("entry_time", ascending=False).head(10)
    if not recent_closed.empty:
        print(f"\n  {BOLD}10 Trade Terakhir (closed):{RESET}")
        print(f"  {'#Ticket':<13} {'Dir':<5} {'Entry':>10} {'Exit':>10} "
              f"{'Lot':>5}  {'P&L':>8}  {'Hasil':<8}  Catatan")
        print(f"  {'-'*82}")
        for _, row in recent_closed.iterrows():
            res   = str(row.get("result", ""))
            pnl_v = pd.to_numeric(row.get("pnl", ""), errors="coerce")
            pnl_v = float(pnl_v) if not pd.isna(pnl_v) else 0.0
            rc    = GREEN if res == "WIN" else (RED if res == "LOSS" else YELLOW)
            pc    = GREEN if pnl_v >= 0 else RED
            dc    = GREEN if str(row.get("direction","")) == "BUY" else RED
            ep    = pd.to_numeric(row.get("entry_price", 0), errors="coerce") or 0
            xp    = pd.to_numeric(row.get("exit_price", ""), errors="coerce")
            xp_s  = f"{float(xp):.2f}" if not pd.isna(xp) and float(xp) != 0 else "--"
            note  = str(row.get("note", "") or row.get("source", ""))[:18]
            lot_v = pd.to_numeric(row.get("lot", 0), errors="coerce") or 0
            print(f"  #{str(row.get('ticket','--')):<12} "
                  f"{dc}{str(row.get('direction','--')):<4}{RESET} "
                  f"{float(ep):>10.2f} {xp_s:>10} "
                  f"{float(lot_v):>5.2f}  "
                  f"{pc}${pnl_v:>+7.2f}{RESET}  "
                  f"{rc}{res:<8}{RESET}  "
                  f"{DIM}{note}{RESET}")

    print(f"  {sep52}")
