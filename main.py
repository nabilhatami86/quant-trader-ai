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
    parser.add_argument("--tv",        action="store_true",       help="Ambil data dari TradingView (tvdatafeed)")
    parser.add_argument("--tv-user",   default="",                help="Username TradingView (opsional, untuk akun premium)")
    parser.add_argument("--tv-pass",   default="",                help="Password TradingView (opsional)")
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
    parser.add_argument("--orders",      type=int,   default=15,    help="Jumlah order sekaligus saat sinyal masuk (default: 15)")
    parser.add_argument("--lot",         type=float, default=0.01,  help="Lot size per order (default: 0.01)")
    parser.add_argument("--micro",       action="store_true",       help="Mode akun mikro (<1 juta IDR) — 1 order, lot 0.01, filter ketat, ML wajib setuju")
    parser.add_argument("--real",        action="store_true",       help="Mode akun real kecil (~$60) — 1 order, lot 0.10, target $15-20/trade, multi-TP + trailing otomatis")
    parser.add_argument("--multi-tp",   action="store_true",       help="Partial close: kunci profit 50%% di TP1, sisanya jalan ke TP2")
    parser.add_argument("--risk",        type=float, default=0.0,   help="Maksimal kerugian per order dalam USD (misal: --risk 1.0). 0 = pakai ATR")
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


def _print_current_candle(df_ind, signal: dict = None) -> None:
    from config import EMA_SLOW, EMA_TREND
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    row    = df_ind.iloc[-1]
    ts     = str(df_ind.index[-1])[:16]
    open_  = float(row.get("Open",  0))
    close  = float(row.get("Close", 0))
    high   = float(row.get("High",  0))
    low    = float(row.get("Low",   0))
    body   = round(abs(close - open_), 2)
    wick_u = round(high - max(open_, close), 2)
    wick_d = round(min(open_, close) - low, 2)
    rsi    = round(float(row.get("rsi", 50)), 1)
    adx    = round(float(row.get("adx",  0)), 1)
    atr    = round(float(row.get("atr",  0)), 2)
    macd   = round(float(row.get("macd", 0)), 3)
    hist   = round(float(row.get("histogram", 0)), 3)
    ema20  = float(row.get(f"ema_{EMA_SLOW}",  close))
    ema50  = float(row.get(f"ema_{EMA_TREND}", close))
    pat    = str(row.get("candle_name", "")).replace("↑","").replace("↓","")[:16]

    dir_   = "BULLISH" if close > open_ else "BEARISH" if close < open_ else "DOJI"
    dc     = GREEN if dir_ == "BULLISH" else RED if dir_ == "BEARISH" else YELLOW
    ema_dir = "↑ BULL" if ema20 > ema50 else "↓ BEAR"
    ema_c   = GREEN if ema20 > ema50 else RED

    sig_dir = signal.get("direction", "WAIT") if signal else "WAIT"
    sc      = GREEN if sig_dir == "BUY" else RED if sig_dir == "SELL" else YELLOW
    sl      = signal.get("sl", "-") if signal else "-"
    tp      = signal.get("tp", "-") if signal else "-"

    print(f"\n  {BOLD}┌─ CANDLE SEKARANG ─ {ts} ─────────────────────────┐{RESET}")
    print(f"  │  {dc}{dir_:<8}{RESET}  Close:{close:>8.2f}  Body:{body:>6.2f}  "
          f"WickU:{wick_u:>5.2f}  WickD:{wick_d:>5.2f}")
    print(f"  │  RSI:{rsi:>5.1f}  ADX:{adx:>5.1f}  ATR:{atr:>6.2f}  "
          f"MACD:{macd:>+7.3f}  Hist:{hist:>+7.3f}")
    print(f"  │  EMA20>50: {ema_c}{ema_dir}{RESET}  "
          + (f"Pattern: {CYAN}{pat}{RESET}" if pat and pat != "nan" else "Pattern: -"))
    print(f"  │  Signal: {sc}{BOLD}{sig_dir}{RESET}"
          + (f"  SL:{sl}  TP:{tp}" if sig_dir in ("BUY","SELL") else ""))
    print(f"  {BOLD}└──────────────────────────────────────────────────────┘{RESET}")


def run_analysis(bot: TradingBot, executor=None, mt4: MT4Bridge = None,
                 force_trade: str = None):
    if not bot.load_data():
        print("[!] Gagal load data. Cek koneksi internet.")
        return False, None

    bot.train_model()

    # Candle memory — cari pola serupa dari histori sebelum generate sinyal
    from data.candle_log import find_similar_candles, print_similar_report, log_candle, print_recent, print_signal_candles
    candle_memory = None
    if bot.df_ind is not None and not bot.df_ind.empty:
        candle_memory = find_similar_candles(bot.symbol, bot.timeframe, bot.df_ind.iloc[-1])
        print_similar_report(candle_memory)

    result = bot.analyze(candle_memory=candle_memory)
    bot.print_analysis(result)

    from analysis.signals import print_filter_log
    sig = result.get("signal", {})
    if sig:
        print_filter_log(sig.get("filters", {}), sig.get("direction", "WAIT"))

    log_candle(bot.symbol, bot.timeframe, bot.df_ind, result.get("signal"))

    # Tampilkan hanya candle yang sedang berjalan
    if bot.df_ind is not None and not bot.df_ind.empty:
        _print_current_candle(bot.df_ind, result.get("signal"))

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
            if exec_result.get("bulk"):
                pass  # output sudah ditangani di executor.execute()
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

    if args.micro:
        import config
        args.lot    = config.MICRO_LOT
        args.orders = config.MICRO_MAX_ORDERS
        args.dca    = 0.0
        config.MIN_SIGNAL_SCORE = config.MICRO_MIN_SCORE
        config.ADX_TREND_MIN    = config.MICRO_ADX_MIN

    if args.real:
        import config
        args.lot       = config.REAL_LOT
        args.orders    = config.REAL_MAX_ORDERS
        args.dca       = 0.0
        args.trail     = config.REAL_TRAIL_PIPS
        args.multi_tp  = True
        config.MULTI_TP_ENABLED    = True
        config.ADX_TREND_MIN       = config.REAL_ADX_MIN
        config.MIN_SIGNAL_SCORE    = 6
        config.ATR_MULTIPLIER_SL   = config.REAL_ATR_SL
        config.ATR_MULTIPLIER_TP   = config.REAL_ATR_TP

    if args.multi_tp:
        import config
        config.MULTI_TP_ENABLED = True

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

    data_src = ("MetaTrader 5 (live)" if args.mt5
                else "TradingView (tvdatafeed)" if args.tv
                else "Yahoo Finance (delay)")
    desc = SYMBOL_DESC.get(symbol, SYMBOLS.get(symbol, ""))

    if args.micro:
        print("""
  ╔══════════════════════════════════════════════════╗
  ║       MODE AKUN MIKRO  (<1 JUTA IDR)            ║
  ║  Filter ketat — 1 order — ML wajib setuju       ║
  ╚══════════════════════════════════════════════════╝""")
        print(f"  Modal estimasi : <Rp 1.000.000  (~$65 USD)")
        print(f"  Risk per trade : {MICRO_RISK_PCT}% modal")
        print(f"  Lot            : {MICRO_LOT} lot/trade")
        print(f"  Max order aktif: {MICRO_MAX_ORDERS} posisi (tidak tumpuk)")
        print(f"  ML min conf    : {MICRO_ML_CONF}%")
        print(f"  ADX minimum    : {MICRO_ADX_MIN}")
        print(f"  Min skor sinyal: {MIN_SIGNAL_SCORE}/10")
        print(f"  DCA            : OFF")
        print()

    if args.real:
        print("""
  ╔══════════════════════════════════════════════════╗
  ║       MODE AKUN REAL  (~$60 USD)                ║
  ║  Target $15-20/trade — Multi-TP + Trailing      ║
  ╚══════════════════════════════════════════════════╝""")
        print(f"  Modal estimasi : ~$60 USD")
        print(f"  Lot            : {REAL_LOT} lot/trade  (~$1/pip XAUUSD)")
        print(f"  Max order aktif: {REAL_MAX_ORDERS} posisi (tidak tumpuk)")
        print(f"  SL             : {REAL_ATR_SL}x ATR  (RR 1:{int(REAL_ATR_TP / REAL_ATR_SL)})")
        print(f"  TP             : {REAL_ATR_TP}x ATR")
        print(f"  Partial close  : 50% tutup di TP1 → profit dikunci")
        print(f"  Trailing stop  : {REAL_TRAIL_PIPS:.0f} pips aktif setelah TP1")
        print(f"  Breakeven      : otomatis setelah profit = jarak SL")
        print(f"  ML min conf    : {REAL_ML_CONF}%")
        print(f"  ADX minimum    : {REAL_ADX_MIN}")
        print(f"  Min skor sinyal: 6/10")
        print(f"  DCA            : OFF")
        print()

    print(f"  Symbol    : {symbol}  —  {desc}")
    print(f"  Timeframe : {timeframe}")
    print(f"  Data      : {data_src}")
    print(f"  ML (RF)   : {'ON - ' + ML_MODEL_TYPE.upper() if ML_ENABLED else 'OFF'}")
    print(f"  LSTM (DL) : {'ON (TensorFlow)' if use_lstm and TF_AVAILABLE else 'OFF (--lstm untuk aktifkan)'}")
    print(f"  News      : {'ON' if use_news else 'OFF'}")
    print(f"  MT5       : {'ON' if args.mt5 else 'OFF'}")
    print(f"  MT4       : {'ON (file bridge)' if args.mt4 else 'OFF'}")
    total_lot = round(args.lot * args.orders, 2)
    print(f"  Lot       : {args.lot} × {args.orders} order  =  {total_lot} lot total")

    if args.risk > 0:
        total_risk = round(args.risk * args.orders, 2)
        print(f"  Risk/order: max ${args.risk:.2f} loss  (total: max ${total_risk:.2f} jika semua SL)")
    if args.dca > 0:
        print(f"  DCA       : pasang order tiap {args.dca:.0f} menit selama sinyal sama")
    if args.multi_tp:
        print(f"  Multi-TP  : ON — tutup {TP1_CLOSE_PCT:.0f}% di TP1, sisa jalan ke TP2, SL → breakeven")
    print(f"  Min Score : {MIN_SIGNAL_SCORE}/10")
    print()

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
                                         dca_minutes=args.dca,
                                         bulk_orders=args.orders,
                                         fixed_lot=args.lot,
                                         strict_mode=args.micro,
                                         risk_per_trade=args.risk,
                                         real_mode=args.real)
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

    from datetime import date

    bot = TradingBot(symbol=symbol, timeframe=timeframe,
                     use_lstm=use_lstm, use_news=use_news,
                     mt5_connector=mt5_conn,
                     use_tv=args.tv,
                     tv_user=args.tv_user or None,
                     tv_pass=args.tv_pass or None)

    bot.fetch_news()
    if bot.news_filter and bot.news_sentiment:
        bot.news_filter.print_news_report(bot.news_sentiment)

    if args.live:
        mt_mode = "MT5" if executor else ("MT4" if mt4_bridge else "ANALYSIS ONLY")
        print(f"[~] Live mode aktif [{mt_mode}]. Refresh setiap {REFRESH_INTERVAL}s (Ctrl+C untuk stop)\n")
        cycle       = 0
        news_date   = date.today()
        try:
            while True:
                cycle += 1

                if date.today() != news_date:
                    print("\n[~] Hari baru — refresh bias berita...")
                    bot.fetch_news()
                    if bot.news_filter and bot.news_sentiment:
                        bot.news_filter.print_news_report(bot.news_sentiment)
                    news_date = date.today()

                print(f"\n{'-'*60}")
                print(f"  Cycle #{cycle}  |  {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
                print(f"{'-'*60}")
                run_analysis(bot, executor=executor, mt4=mt4_bridge,
                             force_trade=args.force_trade)
                if mt5_conn:
                    mt5_conn.print_status()
                from data.trade_journal import print_stats
                print_stats(symbol, timeframe)
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
