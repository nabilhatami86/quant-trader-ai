import os
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(__file__), "history")


def _db_path(symbol: str, timeframe: str) -> str:
    os.makedirs(DB_DIR, exist_ok=True)
    key = symbol.upper().replace("/", "")
    return os.path.join(DB_DIR, f"{key}_{timeframe}.csv")


def save_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    if df is None or df.empty:
        return 0

    path = _db_path(symbol, timeframe)

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Normalisasi timezone — strip tz supaya bisa digabung dengan CSV (tz-naive)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]

    if os.path.exists(path):
        existing  = pd.read_csv(path, index_col=0, parse_dates=True)
        if existing.index.tz is not None:
            existing.index = existing.index.tz_localize(None)
        merged    = pd.concat([existing, df])
        merged    = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        new_count = len(merged) - len(existing)
        merged.to_csv(path)
    else:
        df.sort_index(inplace=True)
        df.to_csv(path)
        new_count = len(df)

    return max(new_count, 0)


def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    path = _db_path(symbol, timeframe)
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def get_db_stats(symbol: str, timeframe: str) -> dict:
    path = _db_path(symbol, timeframe)
    if not os.path.exists(path):
        return {"exists": False, "count": 0}

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return {
        "exists":   True,
        "count":    len(df),
        "from":     str(df.index[0])[:10] if len(df) else "-",
        "to":       str(df.index[-1])[:10] if len(df) else "-",
        "file":     path,
        "size_kb":  round(os.path.getsize(path) / 1024, 1),
    }


def merge_with_db(df_new: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    df_old = load_candles(symbol, timeframe)

    if df_old.empty:
        return df_new
    if df_new is None or df_new.empty:
        return df_old

    df_new = df_new.copy()
    if not isinstance(df_new.index, pd.DatetimeIndex):
        df_new.index = pd.to_datetime(df_new.index)
    if df_new.index.tz is not None:
        df_new.index = df_new.index.tz_localize(None)
    if df_old.index.tz is not None:
        df_old.index = df_old.index.tz_localize(None)

    cols   = [c for c in ["Open", "High", "Low", "Close", "Volume"]
              if c in df_new.columns and c in df_old.columns]
    merged = pd.concat([df_old[cols], df_new[cols]])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)
    return merged
