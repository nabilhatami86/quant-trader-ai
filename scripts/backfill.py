"""
Backfill candle log dari data historis.
Jalankan sekali sebelum bot live — candle memory langsung siap.

Contoh:
    python backfill.py                              # XAUUSD 5m default
    python backfill.py --symbol XAUUSD --tf 1h
    python backfill.py --symbol XAUUSD --tf 5m --mt5
    python backfill.py --symbol EURUSD --tf 1h --tf 4h
"""
import argparse
import sys
import os

# Tambahkan root project ke sys.path agar bisa dijalankan dari scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, SYMBOLS


def backfill(symbol: str, timeframe: str, use_mt5: bool = False) -> None:
    print(f"\n[~] Backfill  {symbol} {timeframe}  ({'MT5' if use_mt5 else 'Yahoo Finance'})")

    # 1. Fetch data
    df_raw = None

    if use_mt5:
        try:
            from app.engine.broker.mt5_connector import MT5Connector
            mt5 = MT5Connector()
            if mt5.connect():
                count_map = {
                    "1m": 5000, "5m": 5000, "15m": 3000,
                    "1h": 2000, "4h": 1000, "1d": 500,
                }
                df_raw = mt5.get_ohlcv(symbol, timeframe,
                                       count=count_map.get(timeframe, 2000))
                mt5.disconnect()
            else:
                print("[!] MT5 gagal connect — fallback ke Yahoo Finance")
        except Exception as e:
            print(f"[!] MT5 error: {e} — fallback ke Yahoo Finance")

    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
        from app.engine.bot import fetch_data
        period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d",
            "1h": "730d", "4h": "730d", "1d": "5y",
        }
        df_raw = fetch_data(symbol, timeframe, period=period_map.get(timeframe, "60d"))

    if df_raw is None or df_raw.empty:
        print(f"[ERROR] Tidak bisa fetch data {symbol} {timeframe}")
        return

    print(f"  Data: {len(df_raw)} candle  "
          f"({str(df_raw.index[0])[:10]} → {str(df_raw.index[-1])[:10]})")

    # 2. Hitung indikator
    from app.engine.signals.indicators import add_all_indicators
    try:
        df_ind = add_all_indicators(df_raw)
    except Exception as e:
        print(f"[ERROR] Indikator gagal: {e}")
        return

    print(f"  Indikator: OK ({len(df_ind)} candle valid)")

    # 3. Simpan ke candle DB (CSV)
    from app.services.candle_db import save_candles
    saved_db = save_candles(df_raw, symbol, timeframe)
    if saved_db > 0:
        print(f"  Candle DB : +{saved_db} candle baru  (CSV)")

    # 4. Backfill candle log (CSV)
    from app.services.candle_log import backfill_candle_log
    added = backfill_candle_log(symbol, timeframe, df_ind)
    if added > 0:
        print(f"  Candle Log: +{added} candle ditulis  (CSV)")
    else:
        print(f"  Candle Log: sudah up-to-date (CSV)")

    # 5 + 6 + 7. Simpan semua ke PostgreSQL dalam satu event loop
    import asyncio
    pg_candles, pg_logs = asyncio.run(_save_all_pg(df_raw, df_ind, symbol, timeframe))
    print(f"  PostgreSQL candles    : +{pg_candles} baris")
    print(f"  PostgreSQL candle_logs: +{pg_logs} baris")

    print(f"[OK] {symbol} {timeframe} selesai\n")


# ── Helper DB (satu event loop untuk semua operasi) ───────────────────────────

async def _save_all_pg(df_raw, df_ind, symbol: str, timeframe: str):
    """Satu coroutine untuk semua DB operations — hindari multi asyncio.run()."""
    from app.database.session import AsyncSessionLocal, engine
    from app.database.crud.candles import bulk_upsert_candles
    from app.database.crud.candle_logs import bulk_upsert_candle_logs
    from app.database.crud.tx_log import log_event

    pg_candles = 0
    pg_logs    = 0

    try:
        async with AsyncSessionLocal() as db:
            # Candles (raw OHLCV)
            try:
                pg_candles = await bulk_upsert_candles(db, symbol, timeframe, df_raw)
            except Exception as e:
                print(f"  [!] candles error: {e}")

            # Candle logs (OHLC + indikator)
            try:
                df = df_ind.copy()
                df.columns = [c.lower() for c in df.columns]
                pg_logs = await bulk_upsert_candle_logs(db, symbol, timeframe, df)
            except Exception as e:
                print(f"  [!] candle_logs error: {e}")

            # tx_log BACKFILL event
            try:
                await log_event(
                    db,
                    event_type="BACKFILL",
                    symbol=symbol,
                    timeframe=timeframe,
                    message=f"Backfill {symbol} {timeframe}: {len(df_raw)} candles",
                    meta={
                        "total_fetched":  len(df_raw),
                        "pg_candles":     pg_candles,
                        "pg_candle_logs": pg_logs,
                    },
                )
            except Exception as e:
                print(f"  [!] tx_log error: {e}")
    except Exception as e:
        print(f"  [!] PostgreSQL connection error: {e}")
    finally:
        # Tutup semua koneksi agar event loop bisa ditutup bersih
        await engine.dispose()

    return pg_candles, pg_logs


def main():
    parser = argparse.ArgumentParser(description="Backfill candle log dari data historis")
    parser.add_argument("--symbol",  nargs="+", default=[DEFAULT_SYMBOL],
                        help="Symbol (bisa lebih dari 1, misal: XAUUSD EURUSD)")
    parser.add_argument("--tf",      nargs="+", default=[DEFAULT_TIMEFRAME],
                        help="Timeframe (bisa lebih dari 1, misal: 5m 1h)")
    parser.add_argument("--mt5",     action="store_true",
                        help="Pakai MT5 sebagai sumber data (harus MT5 terbuka)")
    args = parser.parse_args()

    pairs = [(s, t) for s in args.symbol for t in args.tf]

    print(f"\n{'='*50}")
    print(f"  BACKFILL CANDLE LOG")
    print(f"  {len(pairs)} kombinasi: {', '.join(f'{s}/{t}' for s, t in pairs)}")
    print(f"{'='*50}")

    for symbol, tf in pairs:
        if symbol not in SYMBOLS and symbol not in [v for v in SYMBOLS.values()]:
            print(f"[!] Symbol {symbol} tidak dikenal, skip")
            continue
        backfill(symbol, tf, use_mt5=args.mt5)

    print("Backfill selesai. Sekarang bisa jalankan bot normal.")
    print("  python main.py --symbol XAUUSD --tf 5m --mt5 --live\n")


if __name__ == "__main__":
    main()
