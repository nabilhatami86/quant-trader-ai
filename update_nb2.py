import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('ai/ml/notebooks/xauusd_scalping_ml.ipynb', encoding='utf-8'))

# ── Cell 05: Load + Merge data (termasuk crash data) ──────────────────────────
new_cell05 = (
    'def load_history(path):\n'
    '    """Load CSV dari data/history/ (format standard pandas)."""\n'
    '    df = pd.read_csv(path, index_col=0, parse_dates=True)\n'
    '    df.columns = [c.strip().lower() for c in df.columns]\n'
    '    for col in ["open","high","low","close"]:\n'
    '        df[col] = pd.to_numeric(df[col], errors="coerce")\n'
    '    if "volume" not in df.columns:\n'
    '        df["volume"] = 0.0\n'
    '    df = df[["open","high","low","close","volume"]].dropna()\n'
    '    df = df[~df.index.duplicated(keep="last")].sort_index()\n'
    '    return df\n'
    '\n'
    '# ── Path data ──────────────────────────────────────────────\n'
    'BASE     = Path("C:/Users/muham/Desktop/trader-ai")\n'
    'M5_PATH  = BASE / "data/history/XAUUSD_5m.csv"\n'
    'H1_PATH  = BASE / "data/history/XAUUSD_1h.csv"\n'
    'GOLD_M5  = BASE / "data/history/GOLD_5m.csv"\n'
    'GOLD_H1  = BASE / "data/history/GOLD_1h.csv"\n'
    '\n'
    '# Load M5 utama\n'
    'df_raw = load_history(M5_PATH)\n'
    'print(f"XAUUSD M5 : {len(df_raw):,} candles | {df_raw.index[0]} -> {df_raw.index[-1]}")\n'
    '\n'
    '# Gabung GOLD_5m (candle tambahan, beda sumber/periode)\n'
    'if GOLD_M5.exists():\n'
    '    df_gold = load_history(GOLD_M5)\n'
    '    before  = len(df_raw)\n'
    '    df_raw  = pd.concat([df_raw, df_gold])\n'
    '    df_raw  = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()\n'
    '    added   = len(df_raw) - before\n'
    '    print(f"GOLD  M5  : {len(df_gold):,} candles  (+{added:,} baru setelah dedup)")\n'
    'print(f"Total M5  : {len(df_raw):,} candles | {df_raw.index[0]} -> {df_raw.index[-1]}")\n'
    '\n'
    '# ── Tandai periode crash / high-volatility ─────────────────\n'
    '# ATR spike: candle dengan range > 2x rata-rata range 50 candle sebelumnya\n'
    '_rng   = df_raw["high"] - df_raw["low"]\n'
    '_rng50 = _rng.rolling(50).mean()\n'
    'df_raw["is_crash"] = (_rng > _rng50 * 2.0).astype(int)\n'
    'n_crash = df_raw["is_crash"].sum()\n'
    'print(f"Crash candle: {n_crash:,} ({n_crash/len(df_raw)*100:.1f}%) — ATR spike > 2x median")\n'
    '\n'
    '# Load H1 untuk trend konteks\n'
    'df_h1 = None\n'
    'h1_candidates = [p for p in [H1_PATH, GOLD_H1] if p.exists()]\n'
    'if h1_candidates:\n'
    '    frames = [load_history(p) for p in h1_candidates]\n'
    '    df_h1  = pd.concat(frames)\n'
    '    df_h1  = df_h1[~df_h1.index.duplicated(keep="last")].sort_index()\n'
    '    df_h1["h1_ema50"]  = df_h1["close"].ewm(span=50,  adjust=False).mean()\n'
    '    df_h1["h1_ema200"] = df_h1["close"].ewm(span=200, adjust=False).mean()\n'
    '    df_h1["h1_bull"]   = (df_h1["h1_ema50"] > df_h1["h1_ema200"]).astype(int)\n'
    '    df_h1["h1_rsi"]    = 100 - 100/(1 + df_h1["close"].diff().clip(lower=0).ewm(span=14).mean() /\n'
    '                          (-df_h1["close"].diff().clip(upper=0).ewm(span=14).mean() + 1e-9))\n'
    '    df_h1["h1_adx"]    = df_h1["close"].diff().abs().ewm(span=14, adjust=False).mean()\n'
    '    print(f"H1 (gabung): {len(df_h1):,} candles | {df_h1.index[0]} -> {df_h1.index[-1]}")\n'
    'else:\n'
    '    print("H1 tidak ditemukan — training tanpa HTF context")\n'
    '\n'
    'df_raw.tail(3)'
)

# ── Baca cell 11 untuk ditambah sample_weight ─────────────────────────────────
print('Cell 11 lama:')
print(''.join(nb['cells'][11]['source'])[:800])
print('---')

nb['cells'][5]['source']  = [new_cell05]
nb['cells'][5]['outputs'] = []

# ── Cell 11: Tambah sample_weight ke dataset ──────────────────────────────────
old_c11 = ''.join(nb['cells'][11]['source'])

# Cek apakah is_crash sudah di EXCLUDE
if 'is_crash' not in old_c11:
    # Tambah is_crash ke EXCLUDE dan buat sample_weight setelah y dibuat
    old_c11 = old_c11.replace(
        '"volume",',
        '"volume","is_crash",'
    )
    # Tambah sample_weight setelah baris y = ...
    weight_code = (
        '\n\n# ── Sample weight: crash candle dapat bobot 3x ──────────────\n'
        '# Supaya model belajar lebih hati-hati di kondisi volatile\n'
        '_crash_col = df_clean.get("is_crash", pd.Series(0, index=df_clean.index))\n'
        'sample_weight = _crash_col.reindex(df_clean.index).fillna(0).astype(float)\n'
        'sample_weight = 1.0 + sample_weight * 2.0   # crash=3.0, normal=1.0\n'
        'sw_train = sample_weight.iloc[:SPLIT].values\n'
        'sw_test  = sample_weight.iloc[SPLIT:].values\n'
        'n_crash_train = int((_crash_col.iloc[:SPLIT] == 1).sum())\n'
        'print(f"  sample_weight: crash={n_crash_train} candles bobot 3x, normal bobot 1x")\n'
    )
    # Sisipkan setelah baris split
    old_c11 = old_c11.replace(
        'y_tr=y_all.iloc[:SPLIT]; y_te=y_all.iloc[SPLIT:]',
        'y_tr=y_all.iloc[:SPLIT]; y_te=y_all.iloc[SPLIT:]\n' + weight_code
    )
    if weight_code not in old_c11:
        old_c11 += weight_code

    nb['cells'][11]['source']  = [old_c11]
    nb['cells'][11]['outputs'] = []
    print('Cell 11 diupdate: tambah sample_weight')
else:
    print('Cell 11: is_crash sudah ada, skip')

# ── Cell 13 (Optuna): tambah sample_weight ke fit ────────────────────────────
old_c13 = ''.join(nb['cells'][13]['source'])
print()
print('Cell 13 preview:')
print(old_c13[:400])

with open('ai/ml/notebooks/xauusd_scalping_ml.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print('Notebook updated!')
