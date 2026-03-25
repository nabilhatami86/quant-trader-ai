"""
fix_journal_pnl.py — Patch trade_journal.csv yang P&L-nya 0.0
Jalankan SEKALI: python fix_journal_pnl.py
"""
import os
import datetime as dt
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 tidak terinstall")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "logs", "trade_journal.csv")

# ── Connect MT5 ───────────────────────────────────────────────────────────────
mt5_path = os.getenv("MT5_PATH", "")
login    = int(os.getenv("MT5_LOGIN", "0"))
password = os.getenv("MT5_PASSWORD", "")
server   = os.getenv("MT5_SERVER", "")

AUTO_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
use_path  = mt5_path or AUTO_PATH

if not mt5.initialize(path=use_path, login=login, password=password, server=server):
    print(f"[ERROR] MT5 init gagal: {mt5.last_error()}")
    exit(1)

print("[OK] MT5 terhubung")

# ── Ambil history deals 30 hari terakhir ──────────────────────────────────────
from_date = dt.datetime.now() - dt.timedelta(days=30)
deals     = mt5.history_deals_get(from_date, dt.datetime.now()) or []
print(f"[OK] {len(deals)} deals ditemukan dari MT5 history")

# Map position_id -> deal (entry=1 = close deal)
close_map = {}
for d in deals:
    if d.entry == 1:
        pid = int(getattr(d, "position_id", 0))
        close_map[pid] = {
            "price":  float(d.price),
            "profit": float(d.profit) + float(getattr(d, "swap", 0)) + float(getattr(d, "fee", 0)),
            "comment": getattr(d, "comment", ""),
        }

print(f"[OK] {len(close_map)} close deals di-map")

# ── Baca CSV ──────────────────────────────────────────────────────────────────
if not os.path.exists(JOURNAL_PATH):
    print("[ERROR] trade_journal.csv tidak ditemukan")
    mt5.shutdown()
    exit(1)

df = pd.read_csv(JOURNAL_PATH, dtype=str)
print(f"[OK] {len(df)} baris dibaca dari CSV")

# ── Patch baris yang exit_price=0 atau pnl=0 tapi result bukan OPEN ──────────
patched = 0
for idx, row in df.iterrows():
    result = str(row.get("result", ""))
    if result == "OPEN":
        continue

    try:
        ticket = int(str(row.get("ticket", "0")))
    except ValueError:
        continue

    pnl_cur   = float(row.get("pnl", "0") or "0")
    exit_cur  = float(row.get("exit_price", "0") or "0")

    deal = close_map.get(ticket)
    if deal and (pnl_cur == 0.0 or exit_cur == 0.0):
        new_pnl   = round(deal["profit"], 2)
        new_price = round(deal["price"], 5)
        new_result = "WIN" if new_pnl > 0 else ("LOSS" if new_pnl < 0 else "MANUAL")

        df.at[idx, "pnl"]        = str(new_pnl)
        df.at[idx, "exit_price"] = str(new_price)
        df.at[idx, "result"]     = new_result
        if not str(row.get("note", "")):
            df.at[idx, "note"]   = deal.get("comment", "") or ("sl" if new_pnl < 0 else "tp")

        # Hitung pips
        try:
            ep  = float(str(row.get("entry_price", "0")))
            dir_ = str(row.get("direction", ""))
            pips = (new_price - ep) if dir_ == "BUY" else (ep - new_price)
            df.at[idx, "pips"] = str(round(pips, 5))
        except Exception:
            pass

        patched += 1
        print(f"  Patch #{ticket}: exit={new_price} pnl=${new_pnl:+.2f} -> {new_result}")

# ── Simpan ────────────────────────────────────────────────────────────────────
df.to_csv(JOURNAL_PATH, index=False)
print(f"\n[OK] {patched} baris diperbarui -> {JOURNAL_PATH}")

# ── Ringkasan ─────────────────────────────────────────────────────────────────
closed = df[df["result"].isin(["WIN", "LOSS", "MANUAL"])]
pnl_s  = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
wins   = int((closed["result"] == "WIN").sum())
losses = int((closed["result"] == "LOSS").sum())
total_pnl = round(pnl_s.sum(), 2)

print(f"\n  ━━━ RINGKASAN ━━━")
print(f"  Total closed : {len(closed)}")
print(f"  WIN  : {wins}")
print(f"  LOSS : {losses}")
print(f"  Total P&L : ${total_pnl:+.2f}")

if not closed.empty:
    loss_df  = closed[closed["result"] == "LOSS"]
    loss_pnl = pd.to_numeric(loss_df["pnl"], errors="coerce").fillna(0)
    print(f"\n  Breakdown LOSS ({losses} trade):")
    for _, row in loss_df.iterrows():
        p = pd.to_numeric(row.get("pnl", "0"), errors="coerce") or 0
        print(f"    #{row.get('ticket','--')}  {row.get('direction','--')}  "
              f"entry={row.get('entry_price','--')}  "
              f"exit={row.get('exit_price','--')}  P&L=${float(p):+.2f}")

mt5.shutdown()
print("\n[OK] Selesai.")
