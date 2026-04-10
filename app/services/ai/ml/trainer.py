import sys
import warnings
import json
import time

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, classification_report)
import optuna
import shap
import joblib

from app.services.ai.ml.features import load_mt5, add_m5_features, merge_m1_m5, add_m1_features

optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use('dark_background')
sns.set_palette('husl')

SEED = 42
np.random.seed(SEED)

HERE      = Path(__file__).parent
# Project root: app/services/ai/ml/ -> app/services/ai/ -> app/services/ -> app/ -> root
_ROOT     = HERE.parent.parent.parent.parent
DATA_DIR  = _ROOT / 'ai' / 'ml' / 'data'
MODEL_DIR = _ROOT / 'ai' / 'ml' / 'models'
OUT_DIR   = _ROOT / 'ai' / 'ml' / 'results'

MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
TP_USD       = 2.0
SL_USD       = 1.0
LOOKAHEAD    = 10
SPREAD_LIMIT = 500
K_FEATURES   = 50
N_TRIALS     = 5      # raise to 60+ for better tuning
PROB_THRESH  = 0.58

t0 = time.time()

# ==================================================
# STEP 1: LOAD DATA
# ==================================================
print("=" * 60)
print("STEP 1: Loading MT5 data...")
print("=" * 60)

m1 = load_mt5(DATA_DIR / 'XAUUSDm_M1.csv')
m5 = load_mt5(DATA_DIR / 'XAUUSDm_M5.csv')

print(f"M1: {len(m1):,} candles | {m1.index[0]} -> {m1.index[-1]}")
print(f"M5: {len(m5):,} candles | {m5.index[0]} -> {m5.index[-1]}")
print(f"M1 spread range: {m1['spread_pt'].min()} – {m1['spread_pt'].max()} poin")
print(f"M1 spread mean : {m1['spread_pt'].mean():.1f} poin = {m1['spread_pt'].mean()*0.001:.3f} USD")

# ==================================================
# STEP 2: EDA CHARTS
# ==================================================
print("\nSTEP 2: EDA charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(m5['close'], lw=0.5, color='#00d4ff')
axes[0, 0].set_title('XAUUSD Close Price (M5)')
axes[0, 0].set_ylabel('USD')

axes[0, 1].hist(m1['spread_pt'], bins=60, color='#ff6b35', edgecolor='none', alpha=0.8)
axes[0, 1].axvline(SPREAD_LIMIT, color='yellow', lw=2, ls='--', label=f'Limit={SPREAD_LIMIT}')
pct_ok = (m1['spread_pt'] <= SPREAD_LIMIT).mean() * 100
axes[0, 1].set_title(f'Spread Distribution ({pct_ok:.1f}% candles lolos filter)')
axes[0, 1].legend()

ret = m1['close'].pct_change().dropna() * 100
axes[0, 2].hist(ret, bins=150, color='#a8edea', edgecolor='none', alpha=0.8)
axes[0, 2].set_title('M1 Return Distribution (%)')
axes[0, 2].set_xlim(-1, 1)

m1_h = m1.copy()
m1_h['hour']  = m1_h.index.hour
m1_h['range'] = m1_h['high'] - m1_h['low']

spread_h = m1_h.groupby('hour')['spread_pt'].mean()
bar_c = ['#ff4466' if v > SPREAD_LIMIT else '#00ff88' for v in spread_h]
axes[1, 0].bar(spread_h.index, spread_h.values, color=bar_c)
axes[1, 0].axhline(SPREAD_LIMIT, color='yellow', lw=1.5, ls='--')
axes[1, 0].set_title('Avg Spread per Hour (UTC)')

range_h = m1_h.groupby('hour')['range'].mean()
axes[1, 1].bar(range_h.index, range_h.values, color='#f7971e')
axes[1, 1].set_title('Avg Candle Range per Hour (USD)')

vol_h = m1_h.groupby('hour')['volume'].mean()
axes[1, 2].bar(vol_h.index, vol_h.values, color='#c3f0ca')
axes[1, 2].set_title('Avg Volume per Hour')

plt.suptitle('XAUUSD M1+M5 — EDA Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/eda_overview.png")

# ==================================================
# STEP 3: M5 FEATURES + MERGE
# ==================================================
print("\nSTEP 3: Computing M5 features + merge...")

m5_feat = add_m5_features(m5)
df      = merge_m1_m5(m1, m5_feat)
print(f"  Merged: {df.shape}")

# ==================================================
# STEP 4: M1 FEATURE ENGINEERING
# ==================================================
print("\nSTEP 4: Feature engineering M1...")

df = add_m1_features(df)
print(f"  Total features: {len(df.columns)}")

# ==================================================
# STEP 5: TARGET CREATION
# ==================================================
print(f"\nSTEP 5: Target creation (TP={TP_USD}$ SL={SL_USD}$ look={LOOKAHEAD})...")

t5 = time.time()
close_arr = df['close'].values
high_arr  = df['high'].values
low_arr   = df['low'].values
n         = len(df)
labels    = np.full(n, np.nan)

# 3-class: 2=BUY win, 0=SELL win, 1=WAIT (no clear direction)
for i in range(n - LOOKAHEAD):
    entry  = close_arr[i]
    buy_tp = entry + TP_USD;  buy_sl  = entry - SL_USD
    sel_tp = entry - TP_USD;  sel_sl  = entry + SL_USD

    buy_win = buy_loss = sel_win = sel_loss = False

    for j in range(i + 1, i + 1 + LOOKAHEAD):
        h = high_arr[j]; l = low_arr[j]
        if not buy_win and not buy_loss:
            if h >= buy_tp:  buy_win  = True
            elif l <= buy_sl: buy_loss = True
        if not sel_win and not sel_loss:
            if l <= sel_tp:  sel_win  = True
            elif h >= sel_sl: sel_loss = True

    if buy_win and not buy_loss:
        labels[i] = 2
    elif sel_win and not sel_loss:
        labels[i] = 0
    else:
        labels[i] = 1

df['target'] = labels
n_valid = df['target'].notna().sum()
n_buy   = (df['target'] == 2).sum()
n_sell  = (df['target'] == 0).sum()
n_wait  = (df['target'] == 1).sum()
print(f"  Done in {time.time()-t5:.1f}s")
print(f"  BUY(2) : {n_buy:,} ({n_buy/n_valid*100:.1f}%)")
print(f"  SELL(0): {n_sell:,} ({n_sell/n_valid*100:.1f}%)")
print(f"  WAIT(1): {n_wait:,} ({n_wait/n_valid*100:.1f}%)")

# ==================================================
# STEP 6: PREPROCESSING
# ==================================================
print("\nSTEP 6: Preprocessing...")

df_clean = df[
    df['target'].notna() &
    (df['spread_pt'] <= SPREAD_LIMIT)
].copy()
print(f"  After spread filter ≤{SPREAD_LIMIT}: {len(df_clean):,} rows")

EXCLUDE   = ['target', 'open', 'high', 'low', 'close', 'volume', 'vol_real', 'spread_pt']
feat_cols = [
    c for c in df_clean.columns
    if c not in EXCLUDE
    and df_clean[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]
]

X_all = df_clean[feat_cols].copy()
y_all = df_clean['target'].astype(int)

null_pct = X_all.isnull().mean()
bad_cols = null_pct[null_pct > 0.2].index.tolist()
X_all.drop(columns=bad_cols, inplace=True)
X_all = X_all.ffill().fillna(0).replace([np.inf, -np.inf], 0)

N     = len(X_all)
SPLIT = int(N * 0.80)
X_train = X_all.iloc[:SPLIT];  X_test = X_all.iloc[SPLIT:]
y_train = y_all.iloc[:SPLIT];  y_test = y_all.iloc[SPLIT:]

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train: {X_train.index[0]} -> {X_train.index[-1]}")
print(f"  Test : {X_test.index[0]} -> {X_test.index[-1]}")
vc = y_train.value_counts()
print(f"  Label (train): BUY={vc.get(2,0):,} SELL={vc.get(0,0):,} WAIT={vc.get(1,0):,}")

scaler       = RobustScaler()
X_train_s    = scaler.fit_transform(X_train)
X_test_s     = scaler.transform(X_test)

selector     = SelectKBest(mutual_info_classif, k=K_FEATURES)
X_train_sel  = selector.fit_transform(X_train_s, y_train)
X_test_sel   = selector.transform(X_test_s)
sel_feats    = np.array(X_train.columns)[selector.get_support()].tolist()

mi_scores = pd.Series(
    selector.scores_[selector.get_support()], index=sel_feats
).sort_values(ascending=False)

print(f"\n  Top 10 features (MI):")
for f, s in mi_scores.head(10).items():
    print(f"    {f:<30} {s:.4f}")

# ==================================================
# STEP 7: BASELINE MODELS
# ==================================================
print("\nSTEP 7: Training baseline models...")

RESULTS = {}

def eval_model(name, model, Xtr, ytr, Xte, yte):
    t = time.time()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred  = model.predict(Xte)

    classes  = list(model.classes_)
    buy_idx  = classes.index(2) if 2 in classes else -1
    sell_idx = classes.index(0) if 0 in classes else -1

    prob_buy  = proba[:, buy_idx]  if buy_idx  >= 0 else np.zeros(len(yte))
    prob_sell = proba[:, sell_idx] if sell_idx >= 0 else np.zeros(len(yte))

    acc  = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average='macro', zero_division=0)
    rec  = recall_score(yte, pred, average='macro', zero_division=0)
    f1   = f1_score(yte, pred, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(yte, proba, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    conf_mask = (prob_buy >= 0.55) | (prob_sell >= 0.55)
    conf_acc  = accuracy_score(yte[conf_mask], pred[conf_mask]) if conf_mask.sum() > 5 else 0
    conf_cov  = conf_mask.mean() * 100

    RESULTS[name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                         conf_acc=conf_acc, conf_cov=conf_cov)
    print(f"  [{name:<30}] Acc={acc*100:5.1f}%  Prec={prec*100:5.1f}%  "
          f"F1={f1*100:5.1f}%  AUC={auc:.3f}  ConfAcc={conf_acc*100:5.1f}%  "
          f"({time.time()-t:.0f}s)")
    return model

rf_m  = eval_model('RandomForest',
    RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                           n_jobs=-1, random_state=SEED),
    X_train_sel, y_train, X_test_sel, y_test)

lgb_m = eval_model('LightGBM',
    lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                       random_state=SEED, verbose=-1, n_jobs=-1),
    X_train_sel, y_train, X_test_sel, y_test)

xgb_m = eval_model('XGBoost',
    xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=10, eval_metric='logloss',
                      use_label_encoder=False, random_state=SEED, verbosity=0, n_jobs=-1),
    X_train_sel, y_train, X_test_sel, y_test)

# ==================================================
# STEP 8: OPTUNA TUNING
# ==================================================
print(f"\nSTEP 8: Optuna tuning ({N_TRIALS} trials each)...")

tscv = TimeSeriesSplit(n_splits=5)

def lgb_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 1000),
        learning_rate    = trial.suggest_float('lr', 0.005, 0.15, log=True),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        num_leaves       = trial.suggest_int('num_leaves', 15, 80),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('col', 0.5, 1.0),
        min_child_samples= trial.suggest_int('min_cs', 10, 100),
        reg_alpha        = trial.suggest_float('reg_a', 1e-8, 5.0, log=True),
        reg_lambda       = trial.suggest_float('reg_l', 1e-8, 5.0, log=True),
        random_state=SEED, verbose=-1, n_jobs=-1
    )
    return cross_val_score(lgb.LGBMClassifier(**p), X_train_sel, y_train,
                           cv=tscv, scoring='f1_macro', n_jobs=-1).mean()

def xgb_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 1000),
        learning_rate    = trial.suggest_float('lr', 0.005, 0.15, log=True),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('col', 0.5, 1.0),
        min_child_weight = trial.suggest_int('min_cw', 1, 50),
        gamma            = trial.suggest_float('gamma', 0, 5),
        eval_metric='mlogloss', use_label_encoder=False,
        random_state=SEED, verbosity=0, n_jobs=-1
    )
    return cross_val_score(xgb.XGBClassifier(**p), X_train_sel, y_train,
                           cv=tscv, scoring='f1_macro', n_jobs=-1).mean()

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
print(f"  LGB best F1={study_lgb.best_value:.4f}")

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
print(f"  XGB best F1={study_xgb.best_value:.4f}")

_remap_lgb = {'lr': 'learning_rate', 'col': 'colsample_bytree',
              'min_cs': 'min_child_samples', 'reg_a': 'reg_alpha', 'reg_l': 'reg_lambda'}
_remap_xgb = {'lr': 'learning_rate', 'col': 'colsample_bytree', 'min_cw': 'min_child_weight'}

lgb_best = {_remap_lgb.get(k, k): v for k, v in study_lgb.best_params.items()}
xgb_best = {_remap_xgb.get(k, k): v for k, v in study_xgb.best_params.items()}

lgb_tuned = eval_model('LGB_Tuned',
    lgb.LGBMClassifier(**lgb_best, random_state=SEED, verbose=-1, n_jobs=-1),
    X_train_sel, y_train, X_test_sel, y_test)

xgb_tuned = eval_model('XGB_Tuned',
    xgb.XGBClassifier(**xgb_best, eval_metric='logloss', use_label_encoder=False,
                      random_state=SEED, verbosity=0, n_jobs=-1),
    X_train_sel, y_train, X_test_sel, y_test)

voting = eval_model('Voting_Tuned(LGB+XGB+RF)',
    VotingClassifier(estimators=[
        ('lgb', lgb.LGBMClassifier(**lgb_best, random_state=SEED, verbose=-1, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(**xgb_best, eval_metric='logloss', use_label_encoder=False,
                                   random_state=SEED, verbosity=0, n_jobs=-1)),
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                        n_jobs=-1, random_state=SEED))
    ], voting='soft', weights=[2, 2, 1]),
    X_train_sel, y_train, X_test_sel, y_test)

stacking = eval_model('Stacking(LGB+XGB+RF->LR)',
    StackingClassifier(estimators=[
        ('lgb', lgb.LGBMClassifier(**lgb_best, random_state=SEED, verbose=-1)),
        ('xgb', xgb.XGBClassifier(**xgb_best, eval_metric='logloss', use_label_encoder=False,
                                   random_state=SEED, verbosity=0)),
        ('rf',  ExtraTreesClassifier(n_estimators=200, max_depth=7, n_jobs=-1, random_state=SEED))
    ],
    final_estimator=CalibratedClassifierCV(LogisticRegression(C=1, max_iter=500), method='isotonic'),
    cv=3, stack_method='predict_proba', n_jobs=-1),
    X_train_sel, y_train, X_test_sel, y_test)

# ==================================================
# STEP 9: WALK-FORWARD VALIDATION
# ==================================================
print("\nSTEP 9: Walk-forward validation (5-fold)...")

tscv5 = TimeSeriesSplit(n_splits=5)
wf_lgb_rows = []; wf_xgb_rows = []

for fold, (tr_i, te_i) in enumerate(tscv5.split(X_train_sel)):
    for rows, ModelCls, params, name in [
        (wf_lgb_rows, lgb.LGBMClassifier,
         dict(**lgb_best, random_state=SEED, verbose=-1, n_jobs=-1), 'LGB'),
        (wf_xgb_rows, xgb.XGBClassifier,
         dict(**xgb_best, eval_metric='logloss', use_label_encoder=False,
              random_state=SEED, verbosity=0, n_jobs=-1), 'XGB'),
    ]:
        m = ModelCls(**params)
        m.fit(X_train_sel[tr_i], y_train.iloc[tr_i])
        proba = m.predict_proba(X_train_sel[te_i])
        pred  = m.predict(X_train_sel[te_i])
        yt    = y_train.iloc[te_i]
        cls   = list(m.classes_)
        bi    = cls.index(2) if 2 in cls else -1
        si    = cls.index(0) if 0 in cls else -1
        pb    = proba[:, bi] if bi >= 0 else np.zeros(len(yt))
        ps    = proba[:, si] if si >= 0 else np.zeros(len(yt))
        cm    = (pb >= 0.55) | (ps >= 0.55)
        rows.append({
            'fold': fold + 1,
            'acc':      accuracy_score(yt, pred),
            'f1':       f1_score(yt, pred, average='macro', zero_division=0),
            'prec':     precision_score(yt, pred, average='macro', zero_division=0),
            'conf_acc': accuracy_score(yt[cm], pred[cm]) if cm.sum() > 5 else 0,
            'n': len(te_i),
        })

wf_lgb_df = pd.DataFrame(wf_lgb_rows)
wf_xgb_df = pd.DataFrame(wf_xgb_rows)

print(f"  LGB WF | Acc={wf_lgb_df.acc.mean()*100:.1f}% ± {wf_lgb_df.acc.std()*100:.1f}%  "
      f"F1={wf_lgb_df.f1.mean()*100:.1f}%  ConfAcc={wf_lgb_df.conf_acc.mean()*100:.1f}%")
print(f"  XGB WF | Acc={wf_xgb_df.acc.mean()*100:.1f}% ± {wf_xgb_df.acc.std()*100:.1f}%  "
      f"F1={wf_xgb_df.f1.mean()*100:.1f}%  ConfAcc={wf_xgb_df.conf_acc.mean()*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
folds = range(1, 6)
for ax, col, title in zip(axes, ['acc', 'f1', 'conf_acc'],
                          ['Accuracy', 'F1 Score', 'Confident Accuracy (≥55%)']):
    ax.plot(folds, wf_lgb_df[col], 'o-', color='#00d4ff', lw=2, ms=8, label='LGB Tuned')
    ax.plot(folds, wf_xgb_df[col], 's-', color='#ff6b35', lw=2, ms=8, label='XGB Tuned')
    ax.set_title(title); ax.set_xlabel('Fold'); ax.legend()
    ax.set_ylim(0.3, 1.0); ax.grid(True, alpha=0.2)
    ax.axhline(0.6, color='yellow', lw=1, ls='--', alpha=0.7)
plt.suptitle('Walk-Forward Validation (5-Fold)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'walkforward.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/walkforward.png")

# ==================================================
# STEP 10: SHAP
# ==================================================
print("\nSTEP 10: SHAP analysis...")

lgb_final = lgb.LGBMClassifier(**lgb_best, random_state=SEED, verbose=-1, n_jobs=-1)
lgb_final.fit(X_train_sel, y_train)

n_shap    = min(2000, len(X_test_sel))
explainer = shap.TreeExplainer(lgb_final)
sv_raw    = explainer.shap_values(X_test_sel[:n_shap])

if isinstance(sv_raw, list):
    sv = np.mean([np.abs(s) for s in sv_raw], axis=0)
elif sv_raw.ndim == 3:
    sv = np.abs(sv_raw).mean(axis=2)
else:
    sv = sv_raw

mean_shap = np.abs(sv).mean(axis=0)
shap_df   = (pd.DataFrame({'feature': sel_feats, 'mean_shap': mean_shap})
               .sort_values('mean_shap', ascending=False)
               .reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
top25  = shap_df.head(25)
clr_map = {'m5_': '#ff6b35', 'rsi': '#c3f0ca', 'macd': '#f7971e', 'ema': '#a8edea', 'ret_': '#ffd89b'}
colors = []
for f in top25['feature']:
    c = '#00d4ff'
    for k, v in clr_map.items():
        if k in f: c = v; break
    colors.append(c)

axes[0].barh(top25['feature'][::-1], top25['mean_shap'][::-1], color=colors[::-1])
axes[0].set_title('Top 25 Features — Mean |SHAP| (colour by group)', fontsize=11)
axes[0].set_xlabel('Mean |SHAP|')

from matplotlib.patches import Patch
axes[0].legend(handles=[
    Patch(color='#ff6b35', label='M5 trend'),
    Patch(color='#c3f0ca', label='RSI'),
    Patch(color='#f7971e', label='MACD'),
    Patch(color='#a8edea', label='EMA'),
    Patch(color='#ffd89b', label='Returns'),
    Patch(color='#00d4ff', label='Other'),
], loc='lower right', fontsize=9)

top10_idx = [sel_feats.index(f) for f in shap_df.head(10)['feature']]
for i, (idx, feat) in enumerate(zip(top10_idx, shap_df.head(10)['feature'])):
    axes[1].scatter(sv[:n_shap, idx], np.full(n_shap, i),
                    c=X_test_sel[:n_shap, idx], cmap='RdYlGn', alpha=0.3, s=6)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(shap_df.head(10)['feature'], fontsize=9)
axes[1].axvline(0, color='white', lw=1)
axes[1].set_title('SHAP Scatter — Top 10\n(hijau=nilai tinggi, merah=rendah)', fontsize=11)
axes[1].set_xlabel('SHAP value (-> BUY, <- SELL)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'shap_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/shap_analysis.png")
print(f"\n  Top 10 SHAP features:")
for _, r in shap_df.head(10).iterrows():
    print(f"    {r['feature']:<30} {r['mean_shap']:.4f}")

# ==================================================
# STEP 11: BACKTEST
# ==================================================
print("\nSTEP 11: Backtest...")

voting.fit(X_train_sel, y_train)
proba_test  = voting.predict_proba(X_test_sel)
v_classes   = list(voting.classes_)
buy_idx_v   = v_classes.index(2) if 2 in v_classes else -1
sell_idx_v  = v_classes.index(0) if 0 in v_classes else -1

prob_buy_arr  = proba_test[:, buy_idx_v]  if buy_idx_v  >= 0 else np.zeros(len(proba_test))
prob_sell_arr = proba_test[:, sell_idx_v] if sell_idx_v >= 0 else np.zeros(len(proba_test))

test_ohlcv = df_clean.iloc[SPLIT:].copy()
min_len    = min(len(prob_buy_arr), len(test_ohlcv))
test_ohlcv = test_ohlcv.iloc[:min_len].copy()
prob_buy_arr  = prob_buy_arr[:min_len]
prob_sell_arr = prob_sell_arr[:min_len]

test_ohlcv['prob_buy']  = prob_buy_arr
test_ohlcv['prob_sell'] = prob_sell_arr
test_ohlcv['signal']    = np.where(prob_buy_arr  >= PROB_THRESH,  1,
                          np.where(prob_sell_arr >= PROB_THRESH, -1, 0))

n_buy_sig  = (test_ohlcv['signal'] ==  1).sum()
n_sell_sig = (test_ohlcv['signal'] == -1).sum()
print(f"  BUY signals : {n_buy_sig:,} ({n_buy_sig/len(test_ohlcv)*100:.1f}%)")
print(f"  SELL signals: {n_sell_sig:,} ({n_sell_sig/len(test_ohlcv)*100:.1f}%)")

INIT_CAP = 1000.0
RISK_PCT = 0.01
SPREAD_C = 0.20

cap = INIT_CAP; peak = cap; max_dd = 0
trades = []
c_arr  = test_ohlcv['close'].values
h_arr  = test_ohlcv['high'].values
l_arr  = test_ohlcv['low'].values
s_arr  = test_ohlcv['signal'].values
pb_arr = test_ohlcv['prob_buy'].values
ps_arr = test_ohlcv['prob_sell'].values
idx    = test_ohlcv.index

i = 0
while i < len(test_ohlcv) - LOOKAHEAD:
    sig = s_arr[i]
    if sig == 0:
        i += 1; continue

    entry    = c_arr[i]
    is_buy   = (sig == 1)
    tp_lvl   = entry + TP_USD if is_buy else entry - TP_USD
    sl_lvl   = entry - SL_USD if is_buy else entry + SL_USD
    risk     = INIT_CAP * RISK_PCT
    direction = 'BUY' if is_buy else 'SELL'
    prob_val  = float(pb_arr[i]) if is_buy else float(ps_arr[i])

    result = 'OPEN'; ep = entry; ei = i + 1
    for j in range(i + 1, min(i + 1 + LOOKAHEAD, len(test_ohlcv))):
        if is_buy:
            if h_arr[j] >= tp_lvl: result = 'WIN';  ep = tp_lvl; ei = j; break
            if l_arr[j] <= sl_lvl: result = 'LOSS'; ep = sl_lvl; ei = j; break
        else:
            if l_arr[j] <= tp_lvl: result = 'WIN';  ep = tp_lvl; ei = j; break
            if h_arr[j] >= sl_lvl: result = 'LOSS'; ep = sl_lvl; ei = j; break

    if result == 'OPEN':
        ep = c_arr[min(i + LOOKAHEAD, len(test_ohlcv) - 1)]
        result = ('WIN' if (is_buy and ep > entry) or (not is_buy and ep < entry) else 'LOSS')

    pnl_pts = (ep - entry if result == 'WIN' else -(entry - ep)) if is_buy \
              else (entry - ep if result == 'WIN' else -(ep - entry))
    pnl_usd = risk * (pnl_pts / SL_USD) - SPREAD_C

    cap  += pnl_usd
    peak  = max(peak, cap)
    dd    = (peak - cap) / peak * 100
    max_dd = max(max_dd, dd)

    trades.append({'open': idx[i], 'close': idx[ei], 'dir': direction,
                   'entry': round(entry, 3), 'exit': round(ep, 3),
                   'result': result, 'pnl': round(pnl_usd, 2),
                   'prob': round(prob_val, 4), 'cap': round(cap, 2), 'dd': round(dd, 2)})
    i = ei + 1

bt = pd.DataFrame(trades)
if len(bt) == 0:
    print("  No trades!")
    stats = {}
else:
    wins   = (bt.result == 'WIN').sum(); total = len(bt)
    sum_w  = bt.loc[bt.result == 'WIN', 'pnl'].sum()
    sum_l  = bt.loc[bt.result == 'LOSS', 'pnl'].abs().sum()
    stats  = {
        'total_trades'  : total,
        'wins'          : int(wins),
        'losses'        : int(total - wins),
        'win_rate_%'    : round(wins / total * 100, 2),
        'total_pnl_usd' : round(bt.pnl.sum(), 2),
        'profit_factor' : round(sum_w / (sum_l + 1e-9), 3),
        'max_drawdown_%': round(max_dd, 2),
        'final_capital' : round(cap, 2),
        'return_%'      : round((cap - INIT_CAP) / INIT_CAP * 100, 2),
        'avg_win_usd'   : round(bt.loc[bt.result == 'WIN', 'pnl'].mean(), 2),
        'avg_loss_usd'  : round(bt.loc[bt.result == 'LOSS', 'pnl'].mean(), 2),
    }

    print(f"  Trades={total}  WinRate={stats['win_rate_%']}%  "
          f"PF={stats['profit_factor']}  Return={stats['return_%']}%  "
          f"MaxDD={stats['max_drawdown_%']}%  Capital=${stats['final_capital']}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [0.55, 0.25, 0.20]})
    axes[0].plot(bt.cap.values, lw=1.5, color='#00d4ff', label='Equity')
    axes[0].axhline(INIT_CAP, color='gray', lw=1, ls='--')
    wins_idx = bt[bt.result == 'WIN'].index
    loss_idx = bt[bt.result == 'LOSS'].index
    axes[0].scatter(wins_idx, bt.loc[wins_idx, 'cap'], color='#00ff88', s=20, zorder=5, label='WIN')
    axes[0].scatter(loss_idx, bt.loc[loss_idx, 'cap'], color='#ff4466', s=20, zorder=5, label='LOSS')
    axes[0].fill_between(range(len(bt)), bt.cap, INIT_CAP,
                         where=bt.cap >= INIT_CAP, alpha=0.15, color='#00ff88')
    axes[0].fill_between(range(len(bt)), bt.cap, INIT_CAP,
                         where=bt.cap < INIT_CAP, alpha=0.15, color='#ff4466')
    axes[0].set_title(f"Equity | WR={stats['win_rate_%']}%  PF={stats['profit_factor']}  "
                      f"Return={stats['return_%']}%  MaxDD={stats['max_drawdown_%']}%")
    axes[0].legend()
    axes[1].fill_between(range(len(bt)), -bt.dd, 0, color='#ff4466', alpha=0.7)
    axes[1].set_ylabel('Drawdown %'); axes[1].set_ylim(-max_dd * 1.2, 1)
    clrs_p = ['#00ff88' if r == 'WIN' else '#ff4466' for r in bt.result]
    axes[2].scatter(range(len(bt)), bt.prob, c=clrs_p, s=8, alpha=0.6)
    axes[2].axhline(PROB_THRESH, color='yellow', lw=1.5, ls='--', label=f'Thr={PROB_THRESH}')
    axes[2].set_ylabel('Signal Prob'); axes[2].legend(fontsize=8)
    plt.suptitle('XAUUSD M1 Scalping — Backtest', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    clrs = ['#00ff88' if v > 0 else '#ff4466' for v in bt.pnl]
    axes[0, 0].bar(range(len(bt)), bt.pnl, color=clrs, width=1)
    axes[0, 0].axhline(0, color='white', lw=1)
    axes[0, 0].set_title('PnL per Trade (USD)')

    axes[0, 1].hist(bt.loc[bt.result == 'WIN', 'pnl'],  bins=30, color='#00ff88', alpha=0.7, label='WIN')
    axes[0, 1].hist(bt.loc[bt.result == 'LOSS', 'pnl'], bins=30, color='#ff4466', alpha=0.7, label='LOSS')
    axes[0, 1].set_title('PnL Distribution'); axes[0, 1].legend()

    bt['hour2'] = pd.to_datetime(bt['open']).dt.hour
    wr_h = bt.groupby('hour2')['result'].apply(lambda x: (x == 'WIN').mean() * 100)
    axes[0, 2].bar(wr_h.index, wr_h.values,
                   color=['#00ff88' if v >= 55 else '#ff4466' for v in wr_h])
    axes[0, 2].axhline(50, color='yellow', lw=1, ls='--')
    axes[0, 2].set_title('Win Rate per Hour (UTC)')

    axes[1, 0].plot(bt.cap.values, color='#00d4ff', lw=1.5)
    axes[1, 0].axhline(INIT_CAP, color='gray', lw=1, ls='--')
    axes[1, 0].set_title('Cumulative Capital')

    axes[1, 1].hist(bt.loc[bt.result == 'WIN', 'prob'],  bins=20, color='#00ff88', alpha=0.7, label='WIN')
    axes[1, 1].hist(bt.loc[bt.result == 'LOSS', 'prob'], bins=20, color='#ff4466', alpha=0.7, label='LOSS')
    axes[1, 1].axvline(PROB_THRESH, color='yellow', lw=2, ls='--')
    axes[1, 1].set_title('Prob Distribution WIN vs LOSS'); axes[1, 1].legend()

    bt['month'] = pd.to_datetime(bt['close']).dt.to_period('M')
    mnth = bt.groupby('month')['pnl'].sum()
    axes[1, 2].bar(mnth.index.astype(str), mnth.values,
                   color=['#00ff88' if v > 0 else '#ff4466' for v in mnth])
    axes[1, 2].set_title('Monthly PnL (USD)')
    axes[1, 2].tick_params(axis='x', rotation=30)

    plt.suptitle(f"Trade Analysis  |  {total} trades  |  WR={stats['win_rate_%']}%  "
                 f"PF={stats['profit_factor']}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'trade_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/equity_curve.png + trade_analysis.png")

    bt.to_csv(OUT_DIR / 'trades.csv', index=False)
    print("  Saved: results/trades.csv")

# ==================================================
# STEP 12: MODEL COMPARISON
# ==================================================
print("\nSTEP 12: Model comparison chart...")

res_df = pd.DataFrame(RESULTS).T.round(4).sort_values('conf_acc', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
metr = ['acc', 'prec', 'f1', 'auc', 'conf_acc']
pal  = ['#00d4ff', '#ff6b35', '#c3f0ca', '#f7971e', '#ff9ff3']
x = np.arange(len(res_df)); w = 0.17

for i, (m, col) in enumerate(zip(metr, pal)):
    axes[0].bar(x + i * w, res_df[m], width=w, label=m.upper(), color=col, alpha=0.85)
axes[0].set_xticks(x + w * 2)
axes[0].set_xticklabels(res_df.index, rotation=30, ha='right', fontsize=8)
axes[0].legend(fontsize=9); axes[0].set_title('Model Performance Comparison')
axes[0].axhline(0.6, color='white', lw=1, ls='--', alpha=0.5); axes[0].set_ylim(0, 1.05)

axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
for name, color, m_obj in [
    ('LGB_Tuned',               '#00d4ff', lgb_tuned),
    ('XGB_Tuned',               '#ff6b35', xgb_tuned),
    ('Voting(LGB+XGB+RF)',      '#c3f0ca', voting),
    ('Stacking(LGB+XGB+RF->LR)','#f7971e', stacking),
]:
    try:
        pp     = m_obj.predict_proba(X_test_sel)
        bi     = list(m_obj.classes_).index(2) if 2 in list(m_obj.classes_) else 0
        y_bin  = (y_test == 2).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, pp[:, bi])
        axes[1].plot(fpr, tpr, color=color, lw=2,
                     label=f'{name} BUY ({roc_auc_score(y_bin, pp[:, bi]):.3f})')
    except Exception:
        continue

axes[1].set_title('ROC Curves'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/model_comparison.png")

# ==================================================
# STEP 13: EXPORT MODEL
# ==================================================
print("\nSTEP 13: Exporting model...")

X_full_s   = scaler.fit_transform(X_all)
X_full_sel = selector.fit_transform(X_full_s, y_all)

final = VotingClassifier(estimators=[
    ('lgb', lgb.LGBMClassifier(**lgb_best, random_state=SEED, verbose=-1, n_jobs=-1)),
    ('xgb', xgb.XGBClassifier(**xgb_best, eval_metric='logloss', use_label_encoder=False,
                               random_state=SEED, verbosity=0, n_jobs=-1)),
    ('rf',  RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                    n_jobs=-1, random_state=SEED))
], voting='soft', weights=[2, 2, 1])
final.fit(X_full_sel, y_all)

bundle = dict(model=final, scaler=scaler, selector=selector,
              feature_cols=list(X_all.columns), selected_feats=sel_feats,
              tp=TP_USD, sl=SL_USD, spread_limit=SPREAD_LIMIT, prob_threshold=PROB_THRESH)
joblib.dump(bundle, MODEL_DIR / 'xauusd_scalping_model.joblib', compress=3)

meta = dict(
    symbol='XAUUSDm', timeframes=['M1', 'M5'],
    train_candles=len(df_clean),
    date_range=f"{df_clean.index[0]} -> {df_clean.index[-1]}",
    tp_usd=TP_USD, sl_usd=SL_USD, lookahead=LOOKAHEAD,
    spread_filter=SPREAD_LIMIT, n_features=K_FEATURES,
    best_model=res_df.index[0],
    accuracy=float(res_df.iloc[0]['acc']),
    conf_accuracy=float(res_df.iloc[0]['conf_acc']),
    f1_score=float(res_df.iloc[0]['f1']),
    auc=float(res_df.iloc[0]['auc']),
    backtest=stats if len(bt) > 0 else {},
    lgb_best_params=study_lgb.best_params,
    xgb_best_params=study_xgb.best_params,
    top10_shap=shap_df.head(10)['feature'].tolist(),
    walkforward_lgb=dict(acc_mean=float(wf_lgb_df.acc.mean()),
                         f1_mean=float(wf_lgb_df.f1.mean()),
                         conf_acc_mean=float(wf_lgb_df.conf_acc.mean())),
)
with open(MODEL_DIR / 'xauusd_scalping_model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, default=str)

print(f"  Saved: models/xauusd_scalping_model.joblib")
print(f"  Saved: models/xauusd_scalping_model_meta.json")

# ── Simpan semua model metrics ke DB ─────────────────────────────────────
try:
    import sys as _sys
    _sys.path.insert(0, str(_ROOT))
    from app.services.db_logger import save_ml_result as _save_ml

    _buy_cls = 2  # label BUY = 2
    for _mname, _mrow in res_df.iterrows():
        _save_ml(
            symbol='XAUUSDm',
            timeframe='M5',
            model_type=str(_mname),
            accuracy=float(_mrow.get('acc', 0)),
            conf_accuracy=float(_mrow.get('conf_acc', 0)),
            precision_buy=float(_mrow.get('prec', 0)),
            recall_buy=float(_mrow.get('rec', 0)),
            f1=float(_mrow.get('f1', 0)),
            n_features=int(K_FEATURES),
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_sideways_removed=0,
        )
    print(f"  Saved: {len(res_df)} model results → DB (ml_results)")
except Exception as _e:
    print(f"  [!] DB save ml_results gagal: {_e}")

# ==================================================
# FINAL SUMMARY
# ==================================================
total_time = time.time() - t0
print("\n" + "=" * 65)
print("  XAUUSD SCALPING ML — FINAL SUMMARY")
print("=" * 65)
print(f"  Runtime      : {total_time/60:.1f} menit")
print(f"  Data         : M1 {len(m1):,} | M5 {len(m5):,} candles")
print(f"  Train/Test   : {len(X_train):,} / {len(X_test):,}")
print(f"  Features     : {K_FEATURES} dari {len(X_all.columns)} total")
print(f"  TP/SL        : {TP_USD}$ / {SL_USD}$ (RR 1:{int(TP_USD/SL_USD)})")
print()
print("  Model Rankings (Confident Accuracy):")
for i, (name, row) in enumerate(res_df.iterrows()):
    m = f'#{i+1}'
    print(f"  {m} {name:<32} Acc={row['acc']*100:.1f}%  "
          f"ConfAcc={row['conf_acc']*100:.1f}%  F1={row['f1']*100:.1f}%  AUC={row['auc']:.3f}")
print()
if len(bt) > 0:
    print("  Backtest:")
    for k, v in stats.items():
        print(f"    {k:<22}: {v}")
print()
print("  Output files in results/:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"    {f.name}")
print("=" * 65)
