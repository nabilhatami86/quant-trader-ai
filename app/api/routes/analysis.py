"""
/analysis — Signal analysis log (benar/salah per sinyal + outcome)
"""
import logging
from fastapi import APIRouter, Depends, Query
from app.api.deps import get_bot
from app.services.bot_service import BotService
from app.utils.response import ok

logger = logging.getLogger("trader_ai.routes.analysis")
router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.get("", summary="Log sinyal + analisis candle + outcome")
async def get_analysis_log(
    limit: int = Query(50, ge=1, le=500),
    bot: BotService = Depends(get_bot),
):
    records = bot.get_signal_analysis_log(limit=limit)
    return ok({
        "count"  : len(records),
        "records": records,
    })


@router.get("/summary", summary="Ringkasan akurasi sinyal (tp_score vs outcome)")
async def get_analysis_summary(bot: BotService = Depends(get_bot)):
    records = bot.get_signal_analysis_log(limit=500)
    closed  = [r for r in records if r.get("outcome") in ("WIN", "LOSS")]

    if not closed:
        return ok({"message": "Belum ada data closed dengan outcome"})

    # Accuracy per tp_score
    score_stats: dict = {}
    for r in closed:
        sc = r.get("tp_score")
        if sc is None:
            continue
        sc = int(sc)
        if sc not in score_stats:
            score_stats[sc] = {"win": 0, "loss": 0}
        if r["outcome"] == "WIN":
            score_stats[sc]["win"] += 1
        else:
            score_stats[sc]["loss"] += 1

    score_summary = []
    for sc in sorted(score_stats):
        d = score_stats[sc]
        total = d["win"] + d["loss"]
        score_summary.append({
            "tp_score" : sc,
            "win"      : d["win"],
            "loss"     : d["loss"],
            "total"    : total,
            "win_rate" : round(d["win"] / total * 100, 1) if total else 0,
        })

    # Accuracy per alignment
    align_stats: dict = {}
    for r in closed:
        al = "conflict" if "KONFLIK" in r.get("alignment", "") else \
             "aligned"  if "✓" in r.get("alignment", "") else "mixed"
        if al not in align_stats:
            align_stats[al] = {"win": 0, "loss": 0}
        if r["outcome"] == "WIN":
            align_stats[al]["win"] += 1
        else:
            align_stats[al]["loss"] += 1

    align_summary = []
    for al, d in align_stats.items():
        total = d["win"] + d["loss"]
        align_summary.append({
            "alignment": al,
            "win"      : d["win"],
            "loss"     : d["loss"],
            "total"    : total,
            "win_rate" : round(d["win"] / total * 100, 1) if total else 0,
        })

    # Overall
    total_c = len(closed)
    wins_c  = sum(1 for r in closed if r["outcome"] == "WIN")
    return ok({
        "total_analyzed"  : total_c,
        "overall_win_rate": round(wins_c / total_c * 100, 1),
        "by_tp_score"     : score_summary,
        "by_alignment"    : align_summary,
    })
