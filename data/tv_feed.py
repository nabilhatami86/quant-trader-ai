import pandas as pd

try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False

# Mapping timeframe string → TradingView Interval
TV_INTERVAL_MAP = {
    "1m":  "in_1_minute",
    "5m":  "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "1h":  "in_1_hour",
    "2h":  "in_2_hour",
    "4h":  "in_4_hour",
    "1d":  "in_daily",
    "1w":  "in_weekly",
    "1M":  "in_monthly",
}

# Symbol TradingView: (symbol, exchange)
TV_SYMBOL_MAP = {
    # Emas
    "GOLD":   ("XAUUSD", "OANDA"),
    "XAUUSD": ("XAUUSD", "OANDA"),
    "XAUEUR": ("XAUEUR", "OANDA"),
    # Dollar & Forex
    "EURUSD": ("EURUSD", "OANDA"),
    "GBPUSD": ("GBPUSD", "OANDA"),
    "USDJPY": ("USDJPY", "OANDA"),
    "DXY":    ("DXY",    "TVC"),
    # Index
    "NASDAQ": ("NDX",    "NASDAQ"),
    "SP500":  ("SPX",    "SP"),
    # Crypto
    "BTCUSD": ("BTCUSD", "COINBASE"),
    "ETHUSD": ("ETHUSD", "COINBASE"),
}

_tv_instance = None


def _get_tv(username: str = None, password: str = None) -> "TvDatafeed":
    global _tv_instance
    if _tv_instance is None:
        if username and password:
            _tv_instance = TvDatafeed(username=username, password=password)
        else:
            _tv_instance = TvDatafeed()  # anonymous (rate limit lebih rendah)
    return _tv_instance


def get_tv_ohlcv(symbol_key: str, timeframe: str = "1h",
                 count: int = 1000,
                 username: str = None, password: str = None) -> pd.DataFrame:
    if not TV_AVAILABLE:
        print("[ERROR] tvdatafeed belum terinstall. Jalankan: pip install tvdatafeed")
        return pd.DataFrame()

    tv_sym, exchange = TV_SYMBOL_MAP.get(symbol_key.upper(), (symbol_key, "OANDA"))
    interval_name    = TV_INTERVAL_MAP.get(timeframe, "in_1_hour")

    try:
        tv       = _get_tv(username, password)
        interval = getattr(Interval, interval_name)
        df       = tv.get_hist(symbol=tv_sym, exchange=exchange,
                               interval=interval, n_bars=count)

        if df is None or df.empty:
            print(f"[TV] Tidak ada data untuk {tv_sym} ({exchange})")
            return pd.DataFrame()

        df = df.rename(columns={
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        print(f"[TV] {len(df)} candles dari TradingView — "
              f"{tv_sym}:{exchange} ({timeframe})")
        return df

    except Exception as e:
        print(f"[ERROR] TradingView fetch gagal: {e}")
        return pd.DataFrame()
