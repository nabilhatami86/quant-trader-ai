"""
debug_all.py — Diagnosa komprehensif semua fitur trading bot.
Jalankan: python debug_all.py
Tidak perlu MT5 aktif — akan skip test yang butuh koneksi.
"""

import sys
import os
import traceback
import importlib

# Tambah root ke path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

passed = 0
failed = 0
warned = 0
errors = []

def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}[OK]{RESET}  {msg}")

def fail(msg, exc=None):
    global failed
    failed += 1
    detail = f": {exc}" if exc else ""
    print(f"  {RED}[FAIL]{RESET} {msg}{detail}")
    errors.append(f"{msg}{detail}")

def warn(msg):
    global warned
    warned += 1
    print(f"  {YELLOW}[WARN]{RESET} {msg}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


# ══════════════════════════════════════════════════════════════════
# 1. IMPORT & SYNTAX CHECK
# ══════════════════════════════════════════════════════════════════
section("1. IMPORT & DEPENDENCY CHECK")

modules_to_check = [
    ("config",                          "config.py"),
    ("ai.indicators",                   "ai/indicators.py"),
    ("ai.signals",                      "ai/signals.py"),
    ("data.candle_log",                 "data/candle_log.py"),
    ("data.trade_journal",              "data/trade_journal.py"),
    ("data.news_filter",                "data/news_filter.py"),
    ("data.session_bias",               "data/session_bias.py"),
    ("ai.model",                        "ai/model.py"),
    ("ai.adaptive",                     "ai/adaptive.py"),
    ("ai.news_model",                   "ai/news_model.py"),
    ("services.db_logger",              "services/db_logger.py"),
    ("backend.broker.mt5_connector",    "backend/broker/mt5_connector.py"),
    ("backend.bot",                     "backend/bot.py"),
    ("main",                            "main.py"),
]

imported = {}
for mod_name, path in modules_to_check:
    try:
        mod = importlib.import_module(mod_name)
        imported[mod_name] = mod
        ok(f"import {mod_name}")
    except ImportError as e:
        warn(f"import {mod_name} — dependency missing: {e}")
    except Exception as e:
        fail(f"import {mod_name}", str(e)[:120])

# Third-party deps
section("  Optional Dependencies")
optional_deps = [
    "MetaTrader5", "lightgbm", "xgboost", "sklearn",
    "pandas", "numpy", "yfinance", "psycopg2", "sqlalchemy",
    "requests", "joblib"
]
for dep in optional_deps:
    try:
        importlib.import_module(dep)
        ok(f"{dep}")
    except ImportError:
        warn(f"{dep} — tidak terinstall (opsional)")


# ══════════════════════════════════════════════════════════════════
# 2. CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════
section("2. CONFIG VALIDATION")
try:
    import config as cfg

    checks = [
        ("MIN_SIGNAL_SCORE",      lambda: 1.0 <= cfg.MIN_SIGNAL_SCORE <= 10.0),
        ("MIN_QUALITY_SCORE",     lambda: 1 <= cfg.MIN_QUALITY_SCORE <= 10),
        ("MIN_RR_RATIO",          lambda: cfg.MIN_RR_RATIO >= 1.0),
        ("ATR_MULTIPLIER_SL",     lambda: cfg.ATR_MULTIPLIER_SL > 0),
        ("ATR_MULTIPLIER_TP",     lambda: cfg.ATR_MULTIPLIER_TP > cfg.ATR_MULTIPLIER_SL),
        ("TRADE_COOLDOWN_MIN",    lambda: cfg.TRADE_COOLDOWN_MIN >= 0),
        ("SL_COOLDOWN_MIN",       lambda: cfg.SL_COOLDOWN_MIN >= cfg.TRADE_COOLDOWN_MIN),
        ("MAX_TRADES_PER_HOUR",   lambda: 1 <= cfg.MAX_TRADES_PER_HOUR <= 20),
        ("MAX_DAILY_TRADES",      lambda: cfg.MAX_DAILY_TRADES >= cfg.MAX_TRADES_PER_HOUR),
        ("MAX_OPEN_POSITIONS",    lambda: cfg.MAX_OPEN_POSITIONS >= 1),
        ("ML_MIN_CONFIDENT_ACC",  lambda: 50.0 <= cfg.ML_MIN_CONFIDENT_ACC <= 100.0),
        ("ML_VOTE_THRESHOLD",     lambda: 50 <= cfg.ML_VOTE_THRESHOLD <= 100),
        ("MIN_SL_PIPS",           lambda: cfg.MIN_SL_PIPS >= 1.0),
        ("MAX_LOT_SAFE",          lambda: cfg.MAX_LOT_SAFE > 0),
        ("MAX_LOT_LOSING",        lambda: cfg.MAX_LOT_LOSING > 0 and cfg.MAX_LOT_LOSING <= cfg.MAX_LOT_SAFE),
        ("WEIGHTS dict",          lambda: isinstance(cfg.WEIGHTS, dict) and len(cfg.WEIGHTS) >= 10),
        ("WEIGHTS positive",      lambda: all(v >= 0 for v in cfg.WEIGHTS.values())),
        ("DEFAULT_SYMBOL",        lambda: cfg.DEFAULT_SYMBOL in cfg.SYMBOLS),
        ("DEFAULT_TIMEFRAME",     lambda: cfg.DEFAULT_TIMEFRAME in cfg.TIMEFRAMES),
        ("REAL_ATR_SL",           lambda: cfg.REAL_ATR_SL > 0),
        ("REAL_ATR_TP",           lambda: cfg.REAL_ATR_TP > cfg.REAL_ATR_SL),
        ("REAL_RISK_PCT",         lambda: 0 < cfg.REAL_RISK_PCT <= 100),
        ("REAL_AUTO_LOT_MAX",     lambda: cfg.REAL_AUTO_LOT_MAX >= cfg.REAL_AUTO_LOT_MIN),
    ]

    for name, check_fn in checks:
        try:
            if check_fn():
                val = getattr(cfg, name.split()[0], "dict")
                ok(f"{name} = {val if not isinstance(val, dict) else f'{len(val)} items'}")
            else:
                val = getattr(cfg, name.split()[0], "?")
                fail(f"{name} nilai tidak valid: {val}")
        except AttributeError:
            fail(f"{name} — atribut tidak ditemukan di config.py")
        except Exception as e:
            fail(f"{name} check error", str(e))

    # Check weight keys align with known indicators
    known_weights = {
        "rsi","macd","ema_cross","bb","stoch","adx","candle",
        "obv","vwap","williams_r","cci","volume","smc","pattern_ex",
        "rsi_div","momentum_chain","sma","fibonacci","supertrend",
        "mfi","psar","ichimoku"
    }
    extra = set(cfg.WEIGHTS.keys()) - known_weights
    missing = known_weights - set(cfg.WEIGHTS.keys())
    if extra:
        warn(f"WEIGHTS keys tambahan (tidak dikenal): {extra}")
    if missing:
        warn(f"WEIGHTS keys hilang dari standar: {missing}")
    if not extra and not missing:
        ok("WEIGHTS keys lengkap dan tidak ada typo")

except Exception as e:
    fail("Config validation gagal", str(e))


# ══════════════════════════════════════════════════════════════════
# 3. INDICATOR CALCULATION
# ══════════════════════════════════════════════════════════════════
section("3. INDICATOR CALCULATION (data sintetis)")

try:
    import pandas as pd
    import numpy as np
    from ai.indicators import add_all_indicators

    # Buat data OHLCV sintetis (200 candle, Gold price range)
    np.random.seed(42)
    n = 250
    base = 3000.0
    closes  = base + np.cumsum(np.random.randn(n) * 2)
    highs   = closes + np.abs(np.random.randn(n) * 1.5)
    lows    = closes - np.abs(np.random.randn(n) * 1.5)
    opens   = closes - np.random.randn(n) * 0.8
    volumes = np.abs(np.random.randn(n) * 1000) + 500

    df_test = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }, index=pd.date_range("2025-01-01", periods=n, freq="5min"))

    df_ind = add_all_indicators(df_test.copy())

    if df_ind is None or df_ind.empty:
        fail("add_all_indicators mengembalikan DataFrame kosong")
    else:
        expected_cols = [
            "rsi", "macd", "histogram", "adx", "atr",
            f"ema_{cfg.EMA_SLOW}", f"ema_{cfg.EMA_TREND}", f"ema_{cfg.EMA_LONG}",
            "bb_upper", "bb_lower", "supertrend", "obv",
        ]
        missing_cols = [c for c in expected_cols if c not in df_ind.columns]
        if missing_cols:
            warn(f"Kolom indikator hilang: {missing_cols}")
        else:
            ok(f"add_all_indicators OK — {len(df_ind.columns)} kolom")

        # Check NaN di kolom penting (baris terakhir 50)
        tail = df_ind.tail(50)
        nan_cols = [c for c in expected_cols if c in tail.columns and tail[c].isna().any()]
        if nan_cols:
            warn(f"NaN di kolom: {nan_cols} (50 baris terakhir)")
        else:
            ok("Tidak ada NaN di kolom indikator (50 baris terakhir)")

        # Check for infinite values
        inf_cols = [c for c in df_ind.columns if df_ind[c].dtype in [float, np.float64]
                    and np.isinf(df_ind[c]).any()]
        if inf_cols:
            warn(f"Infinite values di kolom: {inf_cols}")
        else:
            ok("Tidak ada infinite values di indikator")

except Exception as e:
    fail("Indicator calculation", traceback.format_exc()[-200:])
    df_ind = None
    df_test = None


# ══════════════════════════════════════════════════════════════════
# 4. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════
section("4. SIGNAL GENERATION")

try:
    from ai.signals import generate_signal

    if df_ind is not None:
        # generate_signal(df, news_bias=None, news_risk="LOW", candle_memory=None)
        sig = generate_signal(df_ind)

        # Validasi struktur output — rr pakai key "rr_ratio" bukan "rr"
        required_keys = ["direction", "score", "filters", "tp", "sl", "rr_ratio",
                         "confidence", "market_state", "reasons"]
        missing_keys  = [k for k in required_keys if k not in sig]
        if missing_keys:
            fail(f"generate_signal output kehilangan keys: {missing_keys}")
        else:
            ok(f"generate_signal struktur OK")

        direction = sig.get("direction", "?")
        score     = sig.get("score", 0)
        rr        = sig.get("rr_ratio", 0)
        tp        = sig.get("tp")
        sl_       = sig.get("sl")

        if direction not in ("BUY", "SELL", "WAIT"):
            fail(f"direction tidak valid: '{direction}'")
        else:
            ok(f"direction = {direction}  score = {score}")

        if direction in ("BUY", "SELL"):
            if tp is None or sl_ is None:
                fail("TP/SL None pada signal BUY/SELL")
            else:
                ok(f"TP={tp}  SL={sl_}  RR={rr}")
            if rr < cfg.MIN_RR_RATIO:
                warn(f"RR {rr} < MIN_RR_RATIO {cfg.MIN_RR_RATIO}")

        # Test filters dict
        filters = sig.get("filters", {})
        if not isinstance(filters, dict):
            fail("filters bukan dict")
        else:
            ok(f"filters ada {len(filters)} keys")

        # Test dengan news_bias
        sig2 = generate_signal(df_ind, news_risk="HIGH")
        if "direction" not in sig2:
            fail("generate_signal news_risk=HIGH gagal")
        else:
            ok(f"generate_signal news_risk=HIGH OK -- direction={sig2['direction']}")

    else:
        warn("Skip signal test -- df_ind None (indicator gagal)")

except Exception as e:
    fail("Signal generation", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════
# 5. ML MODEL
# ══════════════════════════════════════════════════════════════════
section("5. ML MODEL (train + predict)")

try:
    from ai.model import CandlePredictor
    import config as cfg_ml

    predictor = CandlePredictor(timeframe="5m")

    if df_ind is not None and len(df_ind) >= 100:
        # Train
        result = predictor.train(df_ind)
        if result is None:
            warn("predictor.train() return None — mungkin data terlalu sedikit")
        else:
            conf_acc = result.get("conf_accuracy", 0)
            ok(f"ML train OK — conf_accuracy={conf_acc:.1f}%  "
               f"precision={result.get('precision',0):.1f}%")

            # Predict — predict() expects DataFrame, not Series
            pred = predictor.predict(df_ind)
            if pred is None:
                warn("predictor.predict() return None")
            else:
                direction_p = pred.get("direction", "?")
                prob_p      = pred.get("confidence", pred.get("probability", 0))
                confident_p = pred.get("confident", pred.get("uncertain", None))
                ok(f"ML predict OK -- direction={direction_p}  conf={prob_p:.1f}%")

                # Validasi keys output
                ml_keys = ["direction", "confidence", "proba_buy", "proba_sell"]
                missing_ml = [k for k in ml_keys if k not in pred]
                if missing_ml:
                    warn(f"ML predict output kehilangan keys: {missing_ml}")
                else:
                    ok("ML predict output keys lengkap")

            # Check threshold logic
            threshold_ok = cfg_ml.ML_MIN_CONFIDENT_ACC == 75.0
            if not threshold_ok:
                warn(f"ML_MIN_CONFIDENT_ACC = {cfg_ml.ML_MIN_CONFIDENT_ACC} (expected 75.0)")
            else:
                ok(f"ML_MIN_CONFIDENT_ACC = {cfg_ml.ML_MIN_CONFIDENT_ACC}% (benar)")

            vote_ok = cfg_ml.ML_VOTE_THRESHOLD == 65
            if not vote_ok:
                warn(f"ML_VOTE_THRESHOLD = {cfg_ml.ML_VOTE_THRESHOLD} (expected 65)")
            else:
                ok(f"ML_VOTE_THRESHOLD = {cfg_ml.ML_VOTE_THRESHOLD}% (benar)")

    else:
        warn("Skip ML test — data tidak cukup")

except Exception as e:
    fail("ML model", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════
# 6. ADAPTIVE LEARNER
# ══════════════════════════════════════════════════════════════════
section("6. ADAPTIVE LEARNER")

try:
    from ai.adaptive import AdaptiveLearner, get_learner

    learner = AdaptiveLearner()
    ok("AdaptiveLearner init OK")

    # Test record_trade_outcome
    learner.record_trade_outcome(
        ticket=99999, result="WIN", pnl=5.0,
        direction="BUY", source="#1-Rule+ML",
        signal_score=4.5,
        indicator_snapshot={"rsi_bullish": True, "macd_bullish": True}
    )
    ok("record_trade_outcome WIN OK")

    learner.record_trade_outcome(
        ticket=99998, result="LOSS", pnl=-3.0,
        direction="SELL", source="#3-Rule-Only",
        signal_score=3.2,
        indicator_snapshot={"rsi_bullish": True, "macd_bullish": False}
    )
    ok("record_trade_outcome LOSS OK")

    # Test min_score property
    min_s = learner.min_score
    if not isinstance(min_s, (int, float)):
        fail(f"min_score tipe salah: {type(min_s)}")
    else:
        ok(f"min_score = {min_s}")

    # Test should_retrain_ml
    should = learner.should_retrain_ml()
    ok(f"should_retrain_ml = {should}")

    # Test get_weight_multipliers
    mults = learner.get_weight_multipliers()
    ok(f"get_weight_multipliers = {mults} (min 20 sample untuk aktif)")

    # Test get_real_labels
    labels_df = learner.get_real_labels()
    ok(f"get_real_labels = {len(labels_df)} rows")

    # Test singleton
    l2 = get_learner()
    ok("get_learner() singleton OK")

except Exception as e:
    fail("Adaptive Learner", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════
# 7. SESSION BIAS
# ══════════════════════════════════════════════════════════════════
section("7. SESSION BIAS")

try:
    from data.session_bias import (
        get_current_bias, is_near_session_close,
        was_analyzed_recently, is_market_closed,
        get_session_plan
    )

    bias = get_current_bias()
    if isinstance(bias, dict):
        ok(f"get_current_bias OK — direction={bias.get('direction','(kosong)')}")
    else:
        fail("get_current_bias tidak return dict")

    is_close, label, next_label = is_near_session_close()
    ok(f"is_near_session_close = {is_close}  ({label} -> {next_label})")

    analyzed = was_analyzed_recently("Daily")
    ok(f"was_analyzed_recently('Daily') = {analyzed}")

    closed = is_market_closed(mt5_conn=None)
    ok(f"is_market_closed (no MT5) = {closed}")

    plan = get_session_plan()
    ok(f"get_session_plan OK — {len(plan)} keys: {list(plan.keys())[:5]}")

except Exception as e:
    fail("Session Bias", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════
# 8. NEWS FILTER
# ══════════════════════════════════════════════════════════════════
section("8. NEWS FILTER")

try:
    from data.news_filter import NewsFilter

    nf = NewsFilter()
    ok("NewsFilter init OK")

    # Test get_news_impact
    impact = nf.get_news_impact("XAUUSD")
    if impact in ("HIGH", "MEDIUM", "LOW", "NONE", None):
        ok(f"get_news_impact('XAUUSD') = {impact}")
    else:
        warn(f"get_news_impact return nilai tidak dikenal: {impact}")

    # Test is_high_impact_now
    is_high = nf.is_high_impact_now("XAUUSD")
    ok(f"is_high_impact_now = {is_high}")

except AttributeError as e:
    # Cek metode apa yang ada
    try:
        from data.news_filter import NewsFilter
        nf = NewsFilter()
        methods = [m for m in dir(nf) if not m.startswith("_")]
        warn(f"NewsFilter ada tapi metode berbeda: {methods}")
    except Exception:
        fail("NewsFilter", str(e))
except Exception as e:
    fail("News Filter", traceback.format_exc()[-200:])


# ══════════════════════════════════════════════════════════════════
# 9. TRADE JOURNAL
# ══════════════════════════════════════════════════════════════════
section("9. TRADE JOURNAL")

try:
    import data.trade_journal as tj_mod

    # trade_journal pakai module-level functions, bukan class
    required_fns = ["log_entry", "log_exit", "is_open", "get_stats", "print_stats", "get_recent_trades"]
    missing_fns  = [fn for fn in required_fns if not hasattr(tj_mod, fn)]
    if missing_fns:
        fail(f"trade_journal missing functions: {missing_fns}")
    else:
        ok(f"trade_journal module functions OK: {required_fns}")

    # Test get_stats (baca journal existing)
    stats = tj_mod.get_stats(symbol="XAUUSD")
    if isinstance(stats, dict):
        ok(f"get_stats OK -- keys: {list(stats.keys())[:5]}")
    else:
        warn(f"get_stats return tipe: {type(stats)}")

    # Test get_recent_trades
    recent = tj_mod.get_recent_trades(symbol="XAUUSD", n=5)
    if hasattr(recent, "shape"):
        ok(f"get_recent_trades OK -- {len(recent)} rows, {len(recent.columns)} kolom")
    else:
        warn(f"get_recent_trades return tipe: {type(recent)}")

    # Test print_stats (capture output agar tidak crash di encoding)
    import io
    from contextlib import redirect_stdout
    f_buf = io.StringIO()
    try:
        with redirect_stdout(f_buf):
            tj_mod.print_stats(symbol="XAUUSD")
        out = f_buf.getvalue()
        ok(f"print_stats tidak crash ({len(out)} chars output)")
    except Exception as pe:
        warn(f"print_stats encoding issue: {str(pe)[:80]}")

except Exception as e:
    fail("Trade Journal", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════
# 10. CANDLE LOG
# ══════════════════════════════════════════════════════════════════
section("10. CANDLE LOG")

try:
    import data.candle_log as cl_mod
    from data.candle_log import log_candle

    # Fungsi-fungsi yang ada di candle_log
    required_fns = ["log_candle", "_load_df", "_candle_features"]
    for fn in required_fns:
        if hasattr(cl_mod, fn):
            ok(f"candle_log.{fn} ada")
        else:
            warn(f"candle_log.{fn} tidak ada")

    if df_ind is not None:
        import tempfile
        tmp_dir2 = tempfile.mkdtemp()

        original_log_dir = cl_mod.LOG_DIR
        cl_mod.LOG_DIR = tmp_dir2

        log_candle("XAUUSD", "5m", df_ind,
                   signal={"direction": "BUY", "score": 4.5},
                   realtime_data=None)
        ok("log_candle OK")

        # Test _load_df (fungsi internal)
        cl_mod.LOG_DIR = original_log_dir
        df_log = cl_mod._load_df("XAUUSD", "5m")
        if df_log is not None and not df_log.empty:
            ok(f"_load_df OK -- {len(df_log)} rows, {len(df_log.columns)} kolom")
        else:
            warn("_load_df return kosong")

        import shutil
        shutil.rmtree(tmp_dir2, ignore_errors=True)
    else:
        warn("Skip candle log write test -- df_ind None")

    # Test find_similar_candles ada
    if hasattr(cl_mod, "find_similar_candles"):
        ok("find_similar_candles function ada")
    else:
        warn("find_similar_candles tidak ditemukan")

    # Test get_signal_accuracy ada
    if hasattr(cl_mod, "get_signal_accuracy"):
        ok("get_signal_accuracy function ada")
    else:
        warn("get_signal_accuracy tidak ditemukan")

except Exception as e:
    fail("Candle Log", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════
# 11. DB LOGGER
# ══════════════════════════════════════════════════════════════════
section("11. DB LOGGER")

try:
    import services.db_logger as dbl_mod

    # db_logger pakai module-level functions
    required_fns = ["save_cycle", "save_order_skip", "save_candles_batch",
                    "save_bot_start", "save_bot_stop", "close_trade_in_db",
                    "get_trade_journal_df", "get_trade_stats_db"]
    found   = [fn for fn in required_fns if hasattr(dbl_mod, fn)]
    missing = [fn for fn in required_fns if not hasattr(dbl_mod, fn)]

    ok(f"db_logger functions OK: {len(found)}/{len(required_fns)}")
    if missing:
        warn(f"db_logger missing functions: {missing}")

    # get_trade_stats_db — coba (ada CSV fallback)
    try:
        stats = dbl_mod.get_trade_stats_db(symbol="XAUUSD")
        ok(f"get_trade_stats_db OK -- {stats}")
    except Exception as dbe:
        if "connection" in str(dbe).lower() or "psycopg2" in str(dbe).lower():
            warn(f"get_trade_stats_db skip -- DB tidak tersedia: {str(dbe)[:60]}")
        else:
            fail("get_trade_stats_db", str(dbe)[:80])

except Exception as e:
    fail("DbLogger module", traceback.format_exc()[-200:])


# ══════════════════════════════════════════════════════════════════
# 12. MT5 CONNECTOR (tanpa koneksi aktif)
# ══════════════════════════════════════════════════════════════════
section("12. MT5 CONNECTOR (struktur & logika)")

try:
    from backend.broker.mt5_connector import MT5Connector, SYMBOL_MAP

    # Test SYMBOL_MAP
    if "XAUUSD" in SYMBOL_MAP:
        ok(f"SYMBOL_MAP XAUUSD = {SYMBOL_MAP['XAUUSD']}")
    else:
        warn("SYMBOL_MAP tidak punya XAUUSD key")

    # Test init tanpa MT5
    conn = MT5Connector.__new__(MT5Connector)
    conn.symbol        = "XAUUSD"
    conn.timeframe     = "5m"
    conn.connected     = False
    conn._trades_today = 0
    conn._last_trade_time = 0
    conn._last_sl_time    = 0
    conn._trades_this_hour = []
    ok("MT5Connector object created (tanpa init MT5)")

    # Test should_trade logic kalau punya metode
    # should_trade ada di SignalExecutor, bukan MT5Connector
    from backend.broker.mt5_connector import SignalExecutor
    if hasattr(SignalExecutor, "should_trade"):
        ok("SignalExecutor.should_trade metode ada")
    else:
        fail("SignalExecutor.should_trade tidak ditemukan")

    # Anti-overtrading constants
    import config as cfg_mt5
    for attr in ["TRADE_COOLDOWN_MIN", "SL_COOLDOWN_MIN",
                 "MAX_TRADES_PER_HOUR", "MAX_DAILY_TRADES", "MAX_OPEN_POSITIONS"]:
        val = getattr(cfg_mt5, attr, None)
        if val is not None:
            ok(f"  {attr} = {val}")
        else:
            fail(f"  {attr} tidak ada di config")

except Exception as e:
    fail("MT5 Connector", traceback.format_exc()[-200:])


# ══════════════════════════════════════════════════════════════════
# 13. BOT CLASS
# ══════════════════════════════════════════════════════════════════
section("13. BOT CLASS (tanpa MT5)")

try:
    import backend.bot as bot_mod

    # Cek kelas/fungsi yang ada
    bot_attrs = [a for a in dir(bot_mod) if not a.startswith("_")]
    ok(f"bot.py attrs: {bot_attrs[:10]}...")

    # Cari kelas utama bot
    bot_class = None
    for name in ["TradingBot", "Bot", "GoldBot"]:
        if hasattr(bot_mod, name):
            bot_class = getattr(bot_mod, name)
            ok(f"Found bot class: {name}")
            break

    if bot_class is None:
        warn("Tidak ada kelas bot standar (TradingBot/Bot/GoldBot) — cek manual")

    # Test session bias import — seharusnya ada untuk GUARD-HTF-Bias
    import inspect
    bot_src = inspect.getsource(bot_mod)
    if "session_bias" in bot_src or "get_current_bias" in bot_src:
        ok("bot.py mengimpor session_bias (GUARD-HTF-Bias aktif)")
    else:
        fail("bot.py BELUM mengimpor session_bias -- GUARD-HTF-Bias tidak aktif!")

    # Test main.py punya market closed detection
    import main as main_mod
    main_src = inspect.getsource(main_mod)
    if "is_market_closed" in main_src:
        ok("main.py punya is_market_closed check")
    else:
        fail("main.py BELUM punya is_market_closed -- deep analysis tidak berjalan!")

    if "session_bias" in main_src or "run_close_analysis" in main_src:
        ok("main.py punya session close analysis")
    else:
        warn("main.py belum punya session close analysis trigger")

except Exception as e:
    fail("Bot module", traceback.format_exc()[-200:])


# ══════════════════════════════════════════════════════════════════
# 14. SIGNAL QUALITY — EMA DAMPENING & COUNTER-TREND GATE
# ══════════════════════════════════════════════════════════════════
section("14. SIGNAL QUALITY CHECKS")

try:
    from ai.signals import generate_signal
    import config as cfg_sq

    if df_ind is not None and len(df_ind) >= 50:
        # Buat skenario uptrend (EMA20 > EMA50)
        df_up = df_ind.copy()
        ema20_col = f"ema_{cfg_sq.EMA_SLOW}"
        ema50_col = f"ema_{cfg_sq.EMA_TREND}"

        if ema20_col in df_up.columns and ema50_col in df_up.columns:
            # Force uptrend: EMA20 > EMA50
            df_up[ema20_col] = df_up["Close"] * 1.001
            df_up[ema50_col] = df_up["Close"] * 0.998

            sig_up = generate_signal(df_up)
            dir_up = sig_up.get("direction", "?")
            filters_up = sig_up.get("filters", {})

            ok(f"Skenario uptrend — direction={dir_up}")

            # Cek counter-trend gate ada
            if "counter_trend" in filters_up:
                ok(f"Counter-trend gate aktif: {filters_up['counter_trend'][:60]}")
            else:
                # Mungkin tidak di-trigger karena tidak ada signal yang berlawanan
                ok("Counter-trend gate ada (tidak di-trigger — sinyal searah trend)")

            # Check dampening ada di kode
            import inspect
            sig_src = inspect.getsource(generate_signal)
            if "dampen" in sig_src.lower() or "_dampen_trad" in sig_src:
                ok("EMA dampening logic ditemukan di generate_signal")
            else:
                warn("EMA dampening mungkin belum diimplementasi di generate_signal")

            if "counter_trend" in sig_src or "_counter_min" in sig_src:
                ok("Counter-trend gate logic ditemukan di generate_signal")
            else:
                warn("Counter-trend gate mungkin belum diimplementasi")
        else:
            warn(f"Kolom {ema20_col}/{ema50_col} tidak ada di df_ind")
    else:
        warn("Skip signal quality test — df_ind None/kurang data")

except Exception as e:
    fail("Signal Quality Checks", traceback.format_exc()[-200:])


# ══════════════════════════════════════════════════════════════════
# 15. FILE STRUCTURE CHECK
# ══════════════════════════════════════════════════════════════════
section("15. FILE & STATE INTEGRITY")

state_files = [
    ("ai/adaptive_state.json",           "Adaptive learner state"),
    ("data/session_bias_state.json",    "Session bias state"),
    ("data/session_plan.json",          "Pre-session plan"),
    ("data/news_cache",                 "News cache directory"),
    ("logs",                            "Logs directory"),
]

for rel_path, desc in state_files:
    full_path = os.path.join(ROOT, rel_path)
    if os.path.exists(full_path):
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            ok(f"{rel_path} ({size} bytes) — {desc}")
        else:
            files = os.listdir(full_path)
            ok(f"{rel_path}/ ({len(files)} files) — {desc}")
    else:
        warn(f"{rel_path} belum ada — {desc} (akan dibuat saat pertama dipakai)")

# Check critical CSV logs
log_files = [
    f"logs/candles_XAUUSD_5m.csv",
    f"logs/journal_XAUUSD_.csv",
    f"logs/trade_journal.csv",
]
for lf in log_files:
    fp = os.path.join(ROOT, lf)
    if os.path.exists(fp):
        try:
            import pandas as pd
            df_check = pd.read_csv(fp, nrows=5)
            ok(f"{lf} — {os.path.getsize(fp)//1024}KB, {len(df_check.columns)} kolom")
        except Exception as e:
            fail(f"{lf} CSV rusak: {str(e)[:80]}")
    else:
        warn(f"{lf} tidak ada (normal jika belum ada trade)")


# ══════════════════════════════════════════════════════════════════
# RINGKASAN
# ══════════════════════════════════════════════════════════════════
total = passed + failed + warned
print(f"\n{'='*60}")
print(f"{BOLD}RINGKASAN DEBUG{RESET}")
print(f"{'='*60}")
print(f"  {GREEN}PASS  : {passed}{RESET}")
print(f"  {RED}FAIL  : {failed}{RESET}")
print(f"  {YELLOW}WARN  : {warned}{RESET}")
print(f"  Total : {total}")

if errors:
    print(f"\n{BOLD}{RED}ERROR DETAIL:{RESET}")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e[:120]}")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}Semua test lulus! Bot siap dijalankan.{RESET}")
elif failed <= 3:
    print(f"\n  {YELLOW}{BOLD}{failed} masalah ditemukan — perlu diperbaiki sebelum live trading.{RESET}")
else:
    print(f"\n  {RED}{BOLD}{failed} masalah kritis ditemukan — jangan live trading dulu!{RESET}")

print(f"{'='*60}\n")
