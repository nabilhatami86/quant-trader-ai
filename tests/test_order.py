"""
TEST ORDER - Cek apakah bot bisa masang order ke MT5
=====================================================
Jalankan:
  python test_order.py --find-symbols     <- cari nama symbol gold di broker
  python test_order.py --symbol XAUUSDm  <- pakai nama exact dari broker
  python test_order.py --direction SELL
  python test_order.py --close-all
"""
import argparse
import sys
import os

# Tambahkan root project ke sys.path agar bisa dijalankan dari tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as _mt5
except ImportError:
    _mt5 = None

from backend.broker.mt5_connector import MT5Connector, MT5_AVAILABLE

LOT = 0.01


def find_gold_symbols(mt5_conn: MT5Connector):
    """Cari semua symbol yang mengandung 'XAU' atau 'GOLD' di broker."""
    if not _mt5:
        return []
    symbols = _mt5.symbols_get()
    if not symbols:
        return []
    keywords = ("XAU", "GOLD", "xau", "gold")
    found = [s.name for s in symbols if any(k in s.name for k in keywords)]
    return sorted(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",       default="GOLD",       help="Symbol key atau nama exact broker")
    parser.add_argument("--direction",    default="BUY",        help="BUY atau SELL")
    parser.add_argument("--close-all",    action="store_true",  help="Tutup semua posisi bot")
    parser.add_argument("--status",       action="store_true",  help="Tampilkan status akun saja")
    parser.add_argument("--find-symbols", action="store_true",  help="Cari nama symbol XAU/GOLD di broker")
    args = parser.parse_args()

    if not MT5_AVAILABLE:
        print("[ERROR] Library MetaTrader5 tidak terinstall.")
        print("        pip install MetaTrader5")
        sys.exit(1)

    conn = MT5Connector()
    if not conn.connect():
        print(f"[ERROR] Gagal connect: {conn.last_error}")
        sys.exit(1)

    # ─── Cari symbol ──────────────────────────────────────
    if args.find_symbols:
        print("\n[~] Mencari symbol XAU/GOLD di broker...\n")
        found = find_gold_symbols(conn)
        if found:
            print(f"  Ditemukan {len(found)} symbol:\n")
            for s in found:
                info = _mt5.symbol_info(s)
                status = "✓ aktif" if info and info.visible else "  (tidak aktif)"
                print(f"    {s:20s}  {status}")
            print()
            print("  Salin nama yang aktif ke --symbol, contoh:")
            print(f"  python test_order.py --symbol {found[0]}")
            print()
            print("  Lalu update SYMBOL_MAP di broker/mt5_connector.py:")
            print(f'  "GOLD": "{found[0]}",')
        else:
            print("  Tidak ada symbol XAU/GOLD ditemukan di broker ini.")
        conn.disconnect()
        return

    # ─── Status saja ──────────────────────────────────────
    if args.status:
        conn.print_status()
        conn.disconnect()
        return

    # ─── Close all ────────────────────────────────────────
    if args.close_all:
        print(f"\n[~] Menutup semua posisi bot pada {args.symbol}...")
        results = conn.close_all(args.symbol)
        if results:
            for r in results:
                if r["success"]:
                    print(f"[OK] Closed #{r['ticket']} @ {r['close_price']}")
                else:
                    print(f"[ERROR] {r['error']}")
        else:
            print("[~] Tidak ada posisi untuk ditutup")
        conn.print_status()
        conn.disconnect()
        return

    # ─── Test order ───────────────────────────────────────
    direction = args.direction.upper()
    symbol    = args.symbol

    # Jika bukan key di SYMBOL_MAP, inject langsung ke MT5Connector
    from backend.broker.mt5_connector import SYMBOL_MAP
    if symbol not in SYMBOL_MAP:
        SYMBOL_MAP[symbol] = symbol   # pakai nama exact

    print(f"\n{'='*50}")
    print(f"  TEST ORDER: {direction} {LOT} lot {symbol}")
    print(f"  (tanpa SL/TP — hanya untuk test koneksi)")
    print(f"{'='*50}\n")

    si = conn.get_symbol_info(symbol)
    if not si:
        print(f"[ERROR] Symbol '{si or symbol}' tidak ditemukan di broker.")
        print()
        print("  Cari nama yang benar dengan:")
        print("  python test_order.py --find-symbols")
        conn.disconnect()
        sys.exit(1)

    price = si["ask"] if direction == "BUY" else si["bid"]
    print(f"  Symbol : {si['symbol']}")
    print(f"  Bid    : {si['bid']:.5f}")
    print(f"  Ask    : {si['ask']:.5f}")
    print(f"  Spread : {si['spread']} points")
    print(f"  Price  : {price:.5f}\n")

    result = conn.place_order(
        symbol_key=symbol,
        direction=direction,
        lot=LOT,
        sl=None,
        tp=None,
        comment="TEST-TraderAI",
    )

    print()
    if result["success"]:
        print(f"[✓] ORDER BERHASIL MASUK!")
        print(f"    Ticket    : #{result['ticket']}")
        print(f"    Direction : {result['direction']}")
        print(f"    Lot       : {result['lot']}")
        print(f"    Price     : {result['price']:.5f}")
        print(f"    Time      : {result['time']}")
        print()
        print(f"  Tutup dengan: python test_order.py --close-all --symbol {symbol}")
    else:
        print(f"[✗] ORDER GAGAL!")
        print(f"    Error   : {result['error']}")
        if "retcode" in result:
            RETCODE_MSG = {
                10004: "Requote — harga berubah, coba lagi",
                10006: "Request ditolak broker",
                10007: "Request dibatalkan trader",
                10013: "Request tidak valid",
                10014: "Volume tidak valid (cek lot min/step)",
                10015: "Harga tidak valid",
                10018: "Market tutup",
                10019: "Tidak ada margin cukup",
                10030: "Filling mode tidak support — ganti ke ORDER_FILLING_FOK di mt5_connector.py",
            }
            msg = RETCODE_MSG.get(result["retcode"], "Lihat dokumentasi MT5 retcode")
            print(f"    Retcode : {result['retcode']} — {msg}")

    print()
    conn.print_status()
    conn.disconnect()


if __name__ == "__main__":
    main()
