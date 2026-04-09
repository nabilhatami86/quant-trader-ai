"""
/settings — Baca & update config live (TP/SL, target harian, lot, dll)
Perubahan langsung apply ke runtime — tidak perlu restart bot.
Disimpan ke runtime_settings.json agar persist saat restart.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from app.api.deps import get_bot
from app.services.bot_service import BotService
from app.utils.response import ok, err

logger = logging.getLogger("trader_ai.routes.settings")
router = APIRouter(prefix="/settings", tags=["Settings"])

RUNTIME_PATH = Path(__file__).parent.parent.parent / "runtime_settings.json"


# ── Schema ────────────────────────────────────────────────────────────────────

class TpSlSettings(BaseModel):
    fixed_sl_pips: Optional[float] = Field(None, ge=1.0, le=50.0, description="SL dalam points, misal 6.0")
    fixed_tp_pips: Optional[float] = Field(None, ge=1.0, le=100.0, description="TP dalam points, misal 12.0")
    use_swing_sl_tp: Optional[bool] = Field(None, description="True = pakai swing-based SL/TP dari predictor")
    min_rr_ratio: Optional[float] = Field(None, ge=1.0, le=5.0, description="Minimum RR ratio, misal 1.5")
    adaptive_tp_by_prob: Optional[bool] = Field(None, description="TP adaptif sesuai probabilitas ML")
    tp_pips_weak: Optional[float] = Field(None, ge=1.0, le=50.0)
    tp_pips_medium: Optional[float] = Field(None, ge=1.0, le=50.0)
    tp_pips_strong: Optional[float] = Field(None, ge=1.0, le=50.0)


class DailyTargetSettings(BaseModel):
    daily_profit_target_pct: Optional[float] = Field(None, ge=1.0, le=100.0, description="Target profit harian % dari balance, misal 10.0")
    daily_loss_limit_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Batas rugi harian % dari balance, 0 = nonaktif")
    daily_profit_target_usd: Optional[float] = Field(None, ge=0.0, description="Target profit harian dalam USD, 0 = pakai %, misal 5.0")
    daily_loss_limit_usd: Optional[float] = Field(None, ge=0.0, description="Batas rugi harian dalam USD, 0 = nonaktif")
    circuit_breaker_losses: Optional[int] = Field(None, ge=1, le=20, description="Jumlah loss harian sebelum circuit breaker aktif")
    circuit_breaker_cooldown_hours: Optional[float] = Field(None, ge=0.5, le=24.0)


class LotSettings(BaseModel):
    fixed_lot: Optional[float] = Field(None, ge=0.01, le=10.0)
    real_auto_lot: Optional[bool] = None
    real_auto_lot_min: Optional[float] = Field(None, ge=0.01, le=1.0)
    real_auto_lot_max: Optional[float] = Field(None, ge=0.01, le=10.0)
    real_risk_pct: Optional[float] = Field(None, ge=0.1, le=50.0, description="Risk per trade dalam % balance")
    max_lot_safe: Optional[float] = Field(None, ge=0.01, le=10.0)


class FilterSettings(BaseModel):
    min_signal_score: Optional[float] = Field(None, ge=1.0, le=20.0)
    ml_prob_threshold: Optional[float] = Field(None, ge=0.5, le=0.99, description="Minimum ML probability untuk entry")
    max_spread_pips: Optional[float] = Field(None, ge=0.1, le=10.0)
    session_filter: Optional[bool] = None
    trend_filter_enabled: Optional[bool] = None
    max_trades_per_hour: Optional[int] = Field(None, ge=1, le=20)
    trade_cooldown_min: Optional[float] = Field(None, ge=1.0, le=120.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_runtime() -> dict:
    """Load runtime overrides dari file."""
    if RUNTIME_PATH.exists():
        try:
            return json.loads(RUNTIME_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def _save_runtime(data: dict):
    """Simpan runtime overrides ke file."""
    RUNTIME_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def _apply_to_config(overrides: dict):
    """Apply dict ke config module secara langsung (live)."""
    import config as cfg
    applied = []
    for key, val in overrides.items():
        cfg_key = key.upper()
        if hasattr(cfg, cfg_key):
            setattr(cfg, cfg_key, val)
            applied.append(f"{cfg_key}={val}")
    return applied


def _apply_runtime_on_startup():
    """Dipanggil saat startup api_main.py untuk restore overrides."""
    overrides = _load_runtime()
    if overrides:
        applied = _apply_to_config(overrides)
        logger.info(f"Runtime settings restored: {applied}")


def _get_current_values() -> dict:
    """Ambil nilai config saat ini."""
    import config as cfg
    return {
        # TP/SL
        "fixed_sl_pips"          : getattr(cfg, 'FIXED_SL_PIPS', 6.0),
        "fixed_tp_pips"          : getattr(cfg, 'FIXED_TP_PIPS', 12.0),
        "use_swing_sl_tp"        : getattr(cfg, 'USE_SWING_SL_TP', False),
        "min_rr_ratio"           : getattr(cfg, 'MIN_RR_RATIO', 1.5),
        "adaptive_tp_by_prob"    : getattr(cfg, 'ADAPTIVE_TP_BY_PROB', True),
        "tp_pips_weak"           : getattr(cfg, 'FIXED_TP_PIPS_WEAK', 8.0),
        "tp_pips_medium"         : getattr(cfg, 'FIXED_TP_PIPS_MEDIUM', 10.0),
        "tp_pips_strong"         : getattr(cfg, 'FIXED_TP_PIPS_STRONG', 12.0),
        # Daily target
        "daily_profit_target_pct": getattr(cfg, 'REAL_DAILY_PROFIT_PCT', 0.20) * 100,
        "daily_loss_limit_pct"   : getattr(cfg, 'REAL_DAILY_LIMIT_PCT', 0.0) * 100,
        "daily_profit_target_usd": getattr(cfg, 'DAILY_PROFIT_TARGET_USD', 0.0),
        "daily_loss_limit_usd"   : getattr(cfg, 'DAILY_LOSS_LIMIT_USD', 0.0),
        "circuit_breaker_losses" : getattr(cfg, 'CIRCUIT_BREAKER_LOSSES', 5),
        "circuit_breaker_cooldown_hours": getattr(cfg, 'CIRCUIT_BREAKER_COOLDOWN_H', 2.0),
        # Lot
        "fixed_lot"              : getattr(cfg, 'REAL_LOT', 0.01),
        "real_auto_lot"          : getattr(cfg, 'REAL_AUTO_LOT', False),
        "real_auto_lot_min"      : getattr(cfg, 'REAL_AUTO_LOT_MIN', 0.01),
        "real_auto_lot_max"      : getattr(cfg, 'REAL_AUTO_LOT_MAX', 0.10),
        "real_risk_pct"          : getattr(cfg, 'REAL_RISK_PCT', 10.0),
        "max_lot_safe"           : getattr(cfg, 'MAX_LOT_SAFE', 0.03),
        # Filter
        "min_signal_score"       : getattr(cfg, 'MIN_SIGNAL_SCORE', 5),
        "ml_prob_threshold"      : getattr(cfg, 'ML_PROB_THRESHOLD', 0.62),
        "max_spread_pips"        : getattr(cfg, 'MAX_SPREAD_PIPS', 1.5),
        "session_filter"         : getattr(cfg, 'SESSION_FILTER', True),
        "trend_filter_enabled"   : getattr(cfg, 'TREND_FILTER_ENABLED', True),
        "max_trades_per_hour"    : getattr(cfg, 'MAX_TRADES_PER_HOUR', 2),
        "trade_cooldown_min"     : getattr(cfg, 'TRADE_COOLDOWN_MIN', 5.0),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", summary="Baca semua setting saat ini")
async def get_settings():
    current  = _get_current_values()
    runtime  = _load_runtime()
    return ok({
        "current"       : current,
        "runtime_overrides": runtime,
        "note"          : "runtime_overrides = yang sudah diubah via API (persist saat restart)",
    })


@router.put("/tp-sl", summary="Update TP/SL settings")
async def update_tp_sl(body: TpSlSettings, bot: BotService = Depends(get_bot)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Tidak ada perubahan")

    # Map ke nama config
    cfg_map = {
        "fixed_sl_pips"      : "FIXED_SL_PIPS",
        "fixed_tp_pips"      : "FIXED_TP_PIPS",
        "use_swing_sl_tp"    : "USE_SWING_SL_TP",
        "min_rr_ratio"       : "MIN_RR_RATIO",
        "adaptive_tp_by_prob": "ADAPTIVE_TP_BY_PROB",
        "tp_pips_weak"       : "FIXED_TP_PIPS_WEAK",
        "tp_pips_medium"     : "FIXED_TP_PIPS_MEDIUM",
        "tp_pips_strong"     : "FIXED_TP_PIPS_STRONG",
    }

    cfg_updates = {cfg_map[k]: v for k, v in updates.items() if k in cfg_map}
    applied = _apply_to_config({k.lower(): v for k, v in cfg_updates.items()})

    # Persist
    runtime = _load_runtime()
    runtime.update({k.lower(): v for k, v in cfg_updates.items()})
    _save_runtime(runtime)

    logger.info(f"TP/SL updated: {applied}")
    return ok({"applied": applied, "current": _get_current_values()})


@router.put("/daily-target", summary="Update target harian & circuit breaker")
async def update_daily_target(body: DailyTargetSettings, bot: BotService = Depends(get_bot)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Tidak ada perubahan")

    import config as cfg

    applied = []
    runtime = _load_runtime()

    if "daily_profit_target_pct" in updates:
        cfg.REAL_DAILY_PROFIT_PCT = updates["daily_profit_target_pct"] / 100.0
        runtime["real_daily_profit_pct"] = cfg.REAL_DAILY_PROFIT_PCT
        applied.append(f"REAL_DAILY_PROFIT_PCT={cfg.REAL_DAILY_PROFIT_PCT:.2f} ({updates['daily_profit_target_pct']:.1f}%)")

    if "daily_loss_limit_pct" in updates:
        cfg.REAL_DAILY_LIMIT_PCT = updates["daily_loss_limit_pct"] / 100.0
        runtime["real_daily_limit_pct"] = cfg.REAL_DAILY_LIMIT_PCT
        applied.append(f"REAL_DAILY_LIMIT_PCT={cfg.REAL_DAILY_LIMIT_PCT:.2f} ({updates['daily_loss_limit_pct']:.1f}%)")

    if "daily_profit_target_usd" in updates:
        cfg.DAILY_PROFIT_TARGET_USD = updates["daily_profit_target_usd"]
        runtime["daily_profit_target_usd"] = cfg.DAILY_PROFIT_TARGET_USD
        applied.append(f"DAILY_PROFIT_TARGET_USD=${updates['daily_profit_target_usd']:.2f}")

    if "daily_loss_limit_usd" in updates:
        cfg.DAILY_LOSS_LIMIT_USD = updates["daily_loss_limit_usd"]
        runtime["daily_loss_limit_usd"] = cfg.DAILY_LOSS_LIMIT_USD
        applied.append(f"DAILY_LOSS_LIMIT_USD=${updates['daily_loss_limit_usd']:.2f}")

    if "circuit_breaker_losses" in updates:
        cfg.CIRCUIT_BREAKER_LOSSES = updates["circuit_breaker_losses"]
        runtime["circuit_breaker_losses"] = cfg.CIRCUIT_BREAKER_LOSSES
        applied.append(f"CIRCUIT_BREAKER_LOSSES={updates['circuit_breaker_losses']}")

    if "circuit_breaker_cooldown_hours" in updates:
        cfg.CIRCUIT_BREAKER_COOLDOWN_H = updates["circuit_breaker_cooldown_hours"]
        runtime["circuit_breaker_cooldown_h"] = cfg.CIRCUIT_BREAKER_COOLDOWN_H
        applied.append(f"CIRCUIT_BREAKER_COOLDOWN_H={updates['circuit_breaker_cooldown_hours']:.1f}h")

    _save_runtime(runtime)
    logger.info(f"Daily target updated: {applied}")
    return ok({"applied": applied, "current": _get_current_values()})


@router.put("/lot", summary="Update lot settings")
async def update_lot(body: LotSettings, bot: BotService = Depends(get_bot)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Tidak ada perubahan")

    import config as cfg
    applied = []
    runtime = _load_runtime()

    cfg_map = {
        "fixed_lot"        : ("REAL_LOT",         None),
        "real_auto_lot"    : ("REAL_AUTO_LOT",     None),
        "real_auto_lot_min": ("REAL_AUTO_LOT_MIN", None),
        "real_auto_lot_max": ("REAL_AUTO_LOT_MAX", None),
        "real_risk_pct"    : ("REAL_RISK_PCT",     None),
        "max_lot_safe"     : ("MAX_LOT_SAFE",      None),
    }
    for k, v in updates.items():
        if k in cfg_map:
            cfg_key = cfg_map[k][0]
            setattr(cfg, cfg_key, v)
            runtime[cfg_key.lower()] = v
            applied.append(f"{cfg_key}={v}")

    _save_runtime(runtime)
    logger.info(f"Lot updated: {applied}")
    return ok({"applied": applied, "current": _get_current_values()})


@router.put("/filters", summary="Update filter sinyal (score, threshold, session, dll)")
async def update_filters(body: FilterSettings, bot: BotService = Depends(get_bot)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Tidak ada perubahan")

    import config as cfg
    applied = []
    runtime = _load_runtime()

    cfg_map = {
        "min_signal_score"    : "MIN_SIGNAL_SCORE",
        "max_spread_pips"     : "MAX_SPREAD_PIPS",
        "session_filter"      : "SESSION_FILTER",
        "trend_filter_enabled": "TREND_FILTER_ENABLED",
        "max_trades_per_hour" : "MAX_TRADES_PER_HOUR",
        "trade_cooldown_min"  : "TRADE_COOLDOWN_MIN",
    }
    for k, v in updates.items():
        if k in cfg_map:
            setattr(cfg, cfg_map[k], v)
            runtime[cfg_map[k].lower()] = v
            applied.append(f"{cfg_map[k]}={v}")

    # ML threshold update — langsung ke predictor instance
    if "ml_prob_threshold" in updates:
        try:
            from app.engine.bot import _scalping_pred
            if _scalping_pred:
                _scalping_pred.prob_threshold = updates["ml_prob_threshold"]
                applied.append(f"ML prob_threshold={updates['ml_prob_threshold']}")
            cfg.ML_PROB_THRESHOLD = updates["ml_prob_threshold"]
            runtime["ml_prob_threshold"] = updates["ml_prob_threshold"]
        except Exception as e:
            applied.append(f"ML threshold: gagal ({e})")

    _save_runtime(runtime)
    logger.info(f"Filters updated: {applied}")
    return ok({"applied": applied, "current": _get_current_values()})


@router.delete("/reset", summary="Reset semua runtime overrides ke default config")
async def reset_settings():
    if RUNTIME_PATH.exists():
        RUNTIME_PATH.unlink()
    return ok({"message": "Runtime overrides direset — nilai kembali ke config.py default"})
