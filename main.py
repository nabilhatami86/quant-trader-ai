"""
=============================================================
  TRADING ROBOT - EUR/USD & GOLD/USD
  Yahoo Finance Data | ML + Rule-Based | Candle Predictor
=============================================================

Usage:
  python main.py                         # Default (EURUSD, 1h)
  python main.py --symbol GOLD           # Gold/USD
  python main.py --symbol EURUSD --tf 15m
  python main.py --symbol EURUSD --backtest
  python main.py --symbol GOLD --live    # Live mode (auto refresh)
  python main.py --tune                  # Tampilkan info tuning
=============================================================
"""
import argparse
import time
import sys
import os

from config import *
from bot import TradingBot, fetch_data
from backtest.engine import run_backtest, print_backtest_report
from analysis.indicators import add_all_indicators
from broker.mt5_connector import MT5Connector, SignalExecutor, MT5_AVAILABLE
from broker.mt4_bridge import MT4Bridge


def parse_args():
    parser = argparse.ArgumentParser(description="Trading Robot - EUR/USD & GOLD")
    parser.add_argument("--symbol",    default=DEFAULT_SYMBOL,    help="EURUSD atau GOLD")
    parser.add_argument("--tf",        default=DEFAULT_TIMEFRAME, help="Timeframe: 1m,5m,15m,1h,1d")
    parser.add_argument("--backtest",  action="store_true",       help="Jalankan backtest")
    parser.add_argument("--live",      action="store_true",       help="Live mode (refresh otomatis)")
    parser.add_argument("--tune",      action="store_true",       help="Tampilkan parameter tuning")
    parser.add_argument("--lstm",      action="store_true",       help="Aktifkan LSTM (TensorFlow)")
    parser.add_argument("--no-news",   action="store_true",       help="Nonaktifkan news filter")
    parser.add_argument("--no-ml",     action="store_true",       help="Nonaktifkan ML prediction")
    parser.add_argument("--mt5",       action="store_true",       help="Connect ke MetaTrader 5 (auto-execute)")
    parser.add_argument("--mt4",       action="store_true",       help="Connect ke MetaTrader 4 (file bridge)")
    parser.add_argument("--mt-login",  type=int,   default=0,     help="Nomor akun MT5")
    parser.add_argument("--mt-pass",   default="",                help="Password MT5")
    parser.add_argument("--mt-server", default="",                help="Server broker MT5")
    parser.add_argument("--mt-setup",  action="store_true",       help="Setup & generate file EA untuk MT4")
    parser.add_argument("--mt-status", action="store_true",       help="Tampilkan status akun MT5")
    parser.add_argument("--trail",       type=float, default=0.0,   help="Aktifkan trailing stop (pips), misal: --trail 15")
    parser.add_argument("--period",      default=None,              help="Override data period (misal: 3mo)")
    parser.add_argument("--force-trade", default=None,              help="Bypass semua filter, paksa pasang order. Nilai: BUY atau SELL")
    parser.add_argument("--dca",         type=float, default=0.0,   help="Jeda antar order dalam menit — tiap X menit pasang lagi jika sinyal masih sama (misal: --dca 15)")
    return parser.parse_args()


def show_tuning_guide():
    print("""
+============================================================+
|              PANDUAN TUNING PARAMETER                      |
+============================================================+
|                                                            |
|  File: config.py                                           |
|                                                            |
|  -- SINYAL -----------------------------------------       |
|  MIN_SIGNAL_SCORE  : Threshold sinyal (0-10)               |
|    Rendah (3-4)  = lebih banyak sinyal, lebih berisiko     |
|    Tinggi (6-7)  = sinyal langka, lebih akurat             |
|                                                            |
|  -- BOBOT INDIKATOR --------------------------------       |
|  WEIGHTS["rsi"]        = 2.0  (pengaruh RSI)               |
|  WEIGHTS["macd"]       = 2.0  (pengaruh MACD)              |
|  WEIGHTS["ema_cross"]  = 2.0  (pengaruh EMA)               |
|  WEIGHTS["stoch"]      = 1.5  (pengaruh Stochastic)        |
|  WEIGHTS["bb"]         = 1.0  (pengaruh Bollinger)         |
|  WEIGHTS["adx"]        = 1.0  (pengaruh ADX/Trend)         |
|                                                            |
|  -- RISK MANAGEMENT --------------------------------       |
|  ATR_MULTIPLIER_SL = 1.5  (Stop Loss = 1.5x ATR)          |
|  ATR_MULTIPLIER_TP = 3.0  (Take Profit = 3.0x ATR)        |
|                                                            |
|  -- INDIKATOR --------------------------------------       |
|  RSI_PERIOD        = 14  (default standar)                 |
|  RSI_OVERSOLD      = 30  (turunkan = BUY lebih jarang)     |
|  RSI_OVERBOUGHT    = 70  (naikkan = SELL lebih jarang)     |
|  EMA_FAST/SLOW     = 9/21 (crossover sensitivity)          |
|  BB_STD            = 2.0  (bandwidth Bollinger)            |
|                                                            |
|  -- ML MODEL ----------------------------------------      |
|  ML_MODEL_TYPE     = "rf" / "gb"                           |
|  ML_LOOKBACK       = 20  (candle sebelumnya sbg fitur)     |
|                                                            |
+============================================================+
""")


def run_analysis(bot: TradingBot, executor=None, mt4: MT4Bridge = None,
                 force_trade: str = None):
    """Satu siklus analisis + eksekusi ke MT5/MT4 jika terhubung"""
    if not bot.load_data():
        print("[!] Gagal load data. Cek koneksi internet.")
        return False, None

    bot.train_model()
    bot.fetch_news()
    result = bot.analyze()
    bot.print_analysis(result)

    # Tampilkan news report jika ada
    if bot.news_filter and bot.news_sentiment:
        bot.news_filter.print_news_report(bot.news_sentiment)

    if executor:
        sig       = result.get("signal", {})
        ml_pred   = result.get("ml_pred", {})
        news_risk = result.get("news_risk", "LOW")

        if force_trade and force_trade.upper() in ("BUY", "SELL"):
            force_dir = force_trade.upper()
            print(f"\n[MT5] ⚡ FORCE-TRADE aktif — paksa {force_dir} tanpa filter")

            close = result.get("close", 0)
            atr   = result.get("atr", close * 0.001) if result.get("atr") else close * 0.001

            if sig.get("sl") is None:
                from analysis.signals import calculate_smart_tp_sl
                from analysis.indicators import add_all_indicators
                tp_sl = calculate_smart_tp_sl(force_dir, close, atr,
                                              bot.df_ind, 5.0)
                sig = dict(sig)
                sig["direction"] = force_dir
                sig["sl"]        = tp_sl["sl"]
                sig["tp"]        = tp_sl["tp"]
            else:
                sig = dict(sig)
                sig["direction"] = force_dir

            result_order = executor.mt5.place_order(
                symbol_key=bot.symbol,
                direction=force_dir,
                sl=sig.get("sl"),
                tp=sig.get("tp"),
            )
            if result_order.get("success"):
                print(f"[MT5] ✓ Order masuk! Ticket: #{result_order.get('ticket')}")
                print(f"      {force_dir}  SL:{sig.get('sl')}  TP:{sig.get('tp')}")
            else:
                print(f"[MT5] ✗ Order gagal: {result_order.get('error')}")
            return True, result

        exec_dir = result.get("exec_direction", "WAIT")
        exec_src = result.get("exec_source", "")

        print(f"\n[MT5] Keputusan: {exec_dir}  "
              f"(sumber: {exec_src})  |  News Risk: {news_risk}")

        executor.manage_positions(sig)

        if exec_dir in ("BUY", "SELL"):
            exec_result = executor.execute(sig, ml_pred, news_risk)
            if exec_result.get("multi"):
                n = exec_result.get("count", 0)
                total = len(exec_result.get("results", []))
                print(f"[MT5] ✓ {n}/{total} order masuk  SL:{sig.get('sl')}  TP:{sig.get('tp')}")
            elif exec_result.get("success"):
                print(f"[MT5] ✓ Order masuk! Ticket: #{exec_result.get('ticket')}  "
                      f"SL:{sig.get('sl')}  TP:{sig.get('tp')}")
            elif exec_result.get("skipped"):
                print(f"[MT5] Order di-skip: {exec_result.get('reason','posisi sudah ada / DCA menunggu')}")
            else:
                print(f"[MT5] Order gagal: {exec_result.get('error')}")
        else:
            print(f"[MT5] Tidak eksekusi — menunggu sinyal lebih kuat")

    if mt4:
        sig       = result.get("signal", {})
        news_risk = result.get("news_risk", "LOW")
        direction = sig.get("direction", "WAIT")

        if direction in ("BUY", "SELL") and news_risk != "HIGH":
            from broker.mt5_connector import SYMBOL_MAP
            sym = SYMBOL_MAP.get(bot.symbol, bot.symbol)
            mt4.write_signal(
                direction=direction,
                symbol=sym,
                sl=sig.get("sl", 0.0),
                tp=sig.get("tp", 0.0),
                lot=DEFAULT_LOT,
            )

    return True, result


def run_backtest_mode(symbol: str, timeframe: str, period: str = None):
    print(f"\n[~] Running backtest for {symbol} ({timeframe})...")
    from bot import fetch_data
    from analysis.indicators import add_all_indicators

    bt_period = period or BACKTEST_PERIOD
    raw = fetch_data(symbol, timeframe, bt_period)
    if raw.empty:
        print("[!] Tidak ada data untuk backtest")
        return

    result = run_backtest(raw, symbol=symbol)
    print_backtest_report(result)

    if isinstance(result.get("trades"), __import__("pandas").DataFrame):
        last_5 = result["trades"].tail(5)
        print("\n  Last 5 Trades:")
        for _, t in last_5.iterrows():
            icon = "✓" if t["result"] == "WIN" else "✗"
            print(f"    {icon} {t['direction']:4s} | Entry: {t['entry']:.5f} | Exit: {t['exit']:.5f} | PnL: ${t['pnl_usd']:+.2f}")


def main():
    args = parse_args()

    if args.no_ml:
        import config
        config.ML_ENABLED = False

    print("""
+======================================================+
|      TRADING ROBOT - EUR/USD & GOLD/USD              |
|      Powered by Yahoo Finance + ML Predictor         |
+======================================================+""")

    if args.tune:
        show_tuning_guide()
        return

    symbol = args.symbol.upper()
    if symbol not in SYMBOLS:
        print(f"[!] Symbol '{symbol}' tidak dikenal. Pilih: {list(SYMBOLS.keys())}")
        sys.exit(1)

    timeframe = args.tf
    if timeframe not in TIMEFRAMES:
        print(f"[!] Timeframe '{timeframe}' tidak valid. Pilih: {list(TIMEFRAMES.keys())}")
        sys.exit(1)

    from ml.deep_model import TF_AVAILABLE
    use_lstm = args.lstm
    use_news = not args.no_news

    # ─── MT4 SETUP ───────────────────────────────────────────
    if args.mt_setup:
        bridge = MT4Bridge()
        bridge.save_ea_file()
        print("\n  Langkah setup MT4:")
        print("  1. Copy EA_TraderAI.mq4 ke: MT4/MQL4/Experts/")
        print("  2. Buka MetaEditor, compile file tersebut")
        print("  3. Di MT4: pasang EA ke chart EURUSD/XAUUSD")
        print("  4. Aktifkan 'Allow live trading' di pengaturan EA")
        print("  5. Jalankan bot: python main.py --symbol EURUSD --mt4 --live")
        return

    data_src = "MetaTrader 5 (live)" if args.mt5 else "Yahoo Finance (delay)"
    print(f"  Symbol    : {symbol} ({SYMBOLS[symbol]})")
    print(f"  Timeframe : {timeframe}")
    print(f"  Data      : {data_src}")
    print(f"  ML (RF)   : {'ON - ' + ML_MODEL_TYPE.upper() if ML_ENABLED else 'OFF'}")
    print(f"  LSTM (DL) : {'ON (TensorFlow)' if use_lstm and TF_AVAILABLE else 'OFF (--lstm untuk aktifkan)'}")
    print(f"  News      : {'ON' if use_news else 'OFF'}")
    print(f"  MT5       : {'ON' if args.mt5 else 'OFF'}")
    print(f"  MT4       : {'ON (file bridge)' if args.mt4 else 'OFF'}")
    if args.dca > 0:
        print(f"  DCA       : pasang order tiap {args.dca:.0f} menit selama sinyal sama")
    print(f"  Min Score : {MIN_SIGNAL_SCORE}/10")
    print()

    # ─── MT5 CONNECT ─────────────────────────────────────────
    mt5_conn  = None
    executor  = None
    if args.mt5:
        if not MT5_AVAILABLE:
            print("[!] MetaTrader5 library tidak ada. Install: pip install MetaTrader5")
        else:
            mt5_conn = MT5Connector()
            ok = mt5_conn.connect(
                login=args.mt_login or None,
                password=args.mt_pass or None,
                server=args.mt_server or None,
            )
            if ok:
                executor = SignalExecutor(mt5_conn, symbol,
                                         trailing_pips=args.trail,
                                         dca_minutes=args.dca)
                if args.mt_status:
                    mt5_conn.print_status()
                    return
            else:
                print("[!] MT5 gagal connect. Lanjut tanpa eksekusi order.")
                mt5_conn = None

    mt4_bridge = MT4Bridge() if args.mt4 else None
    if mt4_bridge:
        print(f"[OK] MT4 Bridge aktif. Signal file: {mt4_bridge.signal_path}")

    if args.backtest:
        run_backtest_mode(symbol, timeframe, args.period)
        return

    bot = TradingBot(symbol=symbol, timeframe=timeframe,
                     use_lstm=use_lstm, use_news=use_news,
                     mt5_connector=mt5_conn)

    if args.live:
        mt_mode = "MT5" if executor else ("MT4" if mt4_bridge else "ANALYSIS ONLY")
        print(f"[~] Live mode aktif [{mt_mode}]. Refresh setiap {REFRESH_INTERVAL}s (Ctrl+C untuk stop)\n")
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n{'-'*60}")
                print(f"  Cycle #{cycle}  |  {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
                print(f"{'-'*60}")
                run_analysis(bot, executor=executor, mt4=mt4_bridge,
                             force_trade=args.force_trade)
                if mt5_conn:
                    mt5_conn.print_status()
                print(f"  [~] Menunggu {REFRESH_INTERVAL}s...")
                time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            print("\n[OK] Live mode dihentikan.")
            if mt5_conn:
                mt5_conn.disconnect()
    else:
        run_analysis(bot, executor=executor, mt4=mt4_bridge,
                     force_trade=args.force_trade)
        if mt5_conn:
            mt5_conn.print_status()
            mt5_conn.disconnect()


if __name__ == "__main__":
    main()
