"""
ml/adaptive.py — Self-learning system untuk trading bot.

Bot belajar dari 3 sumber:
  1. Indicator Hit Rate  : catat indikator mana yang benar/salah tiap trade
  2. Dynamic Threshold   : sesuaikan MIN_SIGNAL_SCORE otomatis berdasarkan win rate
  3. ML Retraining       : retrain ML dengan label nyata (WIN/LOSS dari MT5)

Kelas utama:
  AdaptiveLearner  : dipanggil tiap kali posisi ditutup

Metode penting:
  record_trade_outcome(ticket, result, pnl, direction, source, score, indicators)
      Catat hasil trade, update hit rate indikator, update source performance,
      jalankan dynamic threshold adjustment, simpan state.

  get_weight_multipliers() -> dict
      Return {indicator: multiplier} berdasarkan hit rate historis.
      Accuracy > 65% → boost 1.2x | Accuracy < 40% → reduce 0.8x
      Minimal 20 sample sebelum adjustment aktif.

  min_score (property) -> float
      MIN_SIGNAL_SCORE yang sudah di-adjust dinamis.

  should_retrain_ml() -> bool
      True jika sudah >= retrain_every (15) trade baru sejak retrain terakhir.

  get_real_labels() -> DataFrame
      Label nyata dari trade yang sudah tutup — dipakai untuk mix ke training ML.

  print_report()
      Tampilkan laporan: win rate, performa per sumber sinyal, indikator terbaik,
      riwayat score adjustments.

Dynamic Threshold Logic:
  Win rate < 35% → score +0.5 (lebih selektif)
  Win rate < 45% → score +0.25
  Win rate > 75% → score -0.5  (bisa lebih agresif)
  Win rate > 65% → score -0.25
  45-65%         → tidak adjust
  Range: [base-1.0, base+3.0]

State disimpan di: ml/adaptive_state.json
Singleton: gunakan get_learner() untuk instance bersama.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

STATE_PATH = os.path.join(os.path.dirname(__file__), "adaptive_state.json")

# ── Default state ─────────────────────────────────────────────────────────────
_DEFAULT_STATE = {
    "version":          1,
    "last_updated":     "",
    "total_trades":     0,
    "recent_window":    20,      # evaluasi 20 trade terakhir

    # Indicator performance: {"rsi": {"correct": 5, "wrong": 3}, ...}
    "indicator_hits":   {},

    # Signal source performance: {"#1-Rule+ML": {"win": 3, "loss": 1}, ...}
    "source_hits":      {},

    # Dynamic threshold state — base diambil dari config MIN_SIGNAL_SCORE
    "base_min_score":   3.0,
    "current_min_score": 3.0,
    "score_adjustments": [],     # log riwayat adjustment

    # Recent trade window (untuk evaluasi)
    "recent_trades":    [],      # max 50

    # Retrain trigger
    "trades_since_retrain": 0,
    "retrain_every":        15,  # retrain ML setiap 15 trade baru
    "last_retrain":         "",
}


# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveLearner:
    """
    Dipanggil setiap kali posisi ditutup untuk belajar dari hasilnya.
    """

    def __init__(self):
        self.state = self._load()

    # ── State I/O ─────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r") as f:
                    s = json.load(f)
                # Merge keys yang mungkin belum ada di state lama
                for k, v in _DEFAULT_STATE.items():
                    s.setdefault(k, v)
                return s
            except Exception:
                pass
        return dict(_DEFAULT_STATE)

    def _save(self):
        self.state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(STATE_PATH, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    # ── Core: record outcome ──────────────────────────────────────────────────

    def record_trade_outcome(
        self,
        ticket:        int,
        result:        str,          # "WIN" / "LOSS" / "MANUAL"
        pnl:           float,
        direction:     str,          # "BUY" / "SELL"
        source:        str,          # "#1-Rule+ML", "#3-Rule-Only", dll
        signal_score:  float = 0.0,
        indicator_snapshot: dict = None,  # {"rsi_bullish": True, "macd_bullish": True, ...}
    ):
        """
        Dipanggil setiap kali trade ditutup.
        Pelajari: indikator mana yang searah dengan hasil trade.
        """
        if result not in ("WIN", "LOSS", "MANUAL"):
            return

        is_win = result == "WIN"

        # ── Guard: skip duplikat ticket (sync bisa panggil 2x) ───────────────
        existing_tickets = {t["ticket"] for t in self.state["recent_trades"]}
        if ticket in existing_tickets:
            return   # sudah tercatat, skip

        self.state["total_trades"] += 1
        self.state["trades_since_retrain"] += 1

        # ── 1. Catat ke recent_trades ────────────────────────────────────────
        entry = {
            "ticket":    ticket,
            "result":    result,
            "pnl":       pnl,
            "direction": direction,
            "source":    source,
            "score":     signal_score,
            "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "indicators": indicator_snapshot or {},
        }
        self.state["recent_trades"].append(entry)
        # Jaga max 50
        if len(self.state["recent_trades"]) > 50:
            self.state["recent_trades"] = self.state["recent_trades"][-50:]

        # ── 2. Update indicator hit rate ─────────────────────────────────────
        if indicator_snapshot:
            for ind_key, ind_bullish in indicator_snapshot.items():
                # ind_bullish = True berarti indikator searah BUY
                # Cocok dengan direction → correct jika trade WIN
                ind_agrees = (
                    (direction == "BUY"  and ind_bullish is True) or
                    (direction == "SELL" and ind_bullish is False)
                )
                if ind_key not in self.state["indicator_hits"]:
                    self.state["indicator_hits"][ind_key] = {"correct": 0, "wrong": 0, "neutral": 0}

                hits = self.state["indicator_hits"][ind_key]
                if ind_bullish is None:
                    hits["neutral"] = hits.get("neutral", 0) + 1
                elif ind_agrees and is_win:
                    hits["correct"] += 1
                elif not ind_agrees and not is_win:
                    hits["correct"] += 1   # indikator berlawanan → benar prediksi juga
                else:
                    hits["wrong"] += 1

        # ── 3. Update source hit rate ────────────────────────────────────────
        src_key = source.split("(")[0].strip()  # "#1-Rule+ML" dari "#1-Rule+ML(72%)"
        if src_key not in self.state["source_hits"]:
            self.state["source_hits"][src_key] = {"win": 0, "loss": 0}
        sh = self.state["source_hits"][src_key]
        if is_win:
            sh["win"] += 1
        else:
            sh["loss"] += 1

        # ── 4. Dynamic threshold adjustment ─────────────────────────────────
        self._adjust_threshold()

        self._save()

    # ── Dynamic Threshold ─────────────────────────────────────────────────────

    def _adjust_threshold(self):
        """
        Sesuaikan MIN_SIGNAL_SCORE berdasarkan win rate 10 trade terakhir.
        Logika:
          Win rate < 35% → naikkan score +0.5 (lebih selektif)
          Win rate < 45% → naikkan score +0.25
          Win rate > 65% → turunkan score -0.25 (bisa lebih agresif)
          Win rate > 75% → turunkan score -0.5
        """
        recents = self.state["recent_trades"][-10:]
        if len(recents) < 5:
            return  # Butuh min 5 data

        wins     = sum(1 for t in recents if t["result"] == "WIN")
        total    = len(recents)
        win_rate = wins / total

        base    = self.state["base_min_score"]
        current = self.state["current_min_score"]

        if win_rate < 0.35:
            delta = +0.5
            reason = f"WR={win_rate:.0%} < 35% — lebih selektif"
        elif win_rate < 0.45:
            delta = +0.25
            reason = f"WR={win_rate:.0%} < 45% — sedikit perketat"
        elif win_rate > 0.75:
            delta = -0.5
            reason = f"WR={win_rate:.0%} > 75% — bisa lebih agresif"
        elif win_rate > 0.65:
            delta = -0.25
            reason = f"WR={win_rate:.0%} > 65% — sedikit longgarkan"
        else:
            return  # 45-65% — tidak perlu adjust

        # Cap: tidak boleh lebih dari base+1.5 atau kurang dari base-0.5
        # Mencegah threshold naik terlalu tinggi dan memblok semua sinyal
        new_score = round(max(base - 0.5, min(base + 1.5, current + delta)), 2)
        if new_score == current:
            return

        self.state["current_min_score"] = new_score
        log_entry = {
            "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "old":       current,
            "new":       new_score,
            "win_rate":  round(win_rate, 3),
            "reason":    reason,
        }
        self.state["score_adjustments"].append(log_entry)
        # Jaga max 20 log
        if len(self.state["score_adjustments"]) > 20:
            self.state["score_adjustments"] = self.state["score_adjustments"][-20:]

        print(f"  [Adaptive] Score threshold: {current} -> {new_score}  ({reason})")

    # ── Weight recommendations ────────────────────────────────────────────────

    def get_weight_multipliers(self) -> dict:
        """
        Return dict {indicator: multiplier} berdasarkan hit rate.
        Indikator dengan accuracy > 65% → boost 1.2x
        Indikator dengan accuracy < 40% → reduce 0.8x
        Min 20 sample sebelum adjust.
        """
        multipliers = {}
        for ind, hits in self.state["indicator_hits"].items():
            total = hits.get("correct", 0) + hits.get("wrong", 0)
            if total < 20:
                continue
            acc = hits["correct"] / total
            if acc > 0.65:
                multipliers[ind] = 1.2
            elif acc > 0.55:
                multipliers[ind] = 1.1
            elif acc < 0.40:
                multipliers[ind] = 0.8
            elif acc < 0.50:
                multipliers[ind] = 0.9
        return multipliers

    # ── Performance Mode (Upgrade) ────────────────────────────────────────────

    def get_performance_mode(self) -> str:
        """
        Return mode berdasarkan win rate 10 trade terakhir:
          CONSERVATIVE  (<40%)  : kurangi frekuensi, tunggu setup terbaik
          NORMAL        (40-65%): default behavior
          AGGRESSIVE    (>65%)  : boleh lebih aktif, setup lebih banyak diterima
        """
        recents = self.state["recent_trades"][-10:]
        if len(recents) < 5:
            return "NORMAL"
        wins = sum(1 for t in recents if t["result"] == "WIN")
        wr   = wins / len(recents)
        if wr < 0.40:
            return "CONSERVATIVE"
        if wr > 0.65:
            return "AGGRESSIVE"
        return "NORMAL"

    def get_cooldown_mult(self) -> float:
        """
        Multiplier untuk TRADE_COOLDOWN_MIN berdasarkan mode.
        CONSERVATIVE: 2.0x — cooldown lebih panjang (kurangi frekuensi)
        AGGRESSIVE:   0.7x — cooldown lebih pendek (lebih aktif)
        """
        return {"CONSERVATIVE": 2.0, "NORMAL": 1.0, "AGGRESSIVE": 0.7}.get(
            self.get_performance_mode(), 1.0)

    def get_max_trades_mult(self) -> float:
        """
        Multiplier untuk MAX_TRADES_PER_HOUR.
        CONSERVATIVE: 0.5x (max 1 trade/jam saat jelek)
        AGGRESSIVE:   1.5x (max 3 trade/jam saat bagus)
        """
        return {"CONSERVATIVE": 0.5, "NORMAL": 1.0, "AGGRESSIVE": 1.5}.get(
            self.get_performance_mode(), 1.0)

    # ── Dynamic score threshold ───────────────────────────────────────────────

    @property
    def min_score(self) -> float:
        """Threshold score saat ini (sudah di-adjust)."""
        return self.state["current_min_score"]

    # ── ML Retrain trigger ────────────────────────────────────────────────────

    def should_retrain_ml(self) -> bool:
        """True jika sudah cukup trade baru untuk retrain ML."""
        return self.state["trades_since_retrain"] >= self.state["retrain_every"]

    def mark_retrained(self):
        self.state["trades_since_retrain"] = 0
        self.state["last_retrain"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def get_real_labels(self) -> pd.DataFrame:
        """
        Return DataFrame label nyata dari trade yang sudah ditutup.
        Kolom: entry_time, direction_label (1=BUY, 0=SELL)
        Dipakai untuk mix ke training data ML.
        """
        rows = []
        for t in self.state["recent_trades"]:
            if t["result"] not in ("WIN", "LOSS"):
                continue
            label = 1 if t["result"] == "WIN" and t["direction"] == "BUY" else \
                    0 if t["result"] == "WIN" and t["direction"] == "SELL" else \
                    1 if t["result"] == "LOSS" and t["direction"] == "SELL" else 0
            rows.append({"time": t["time"], "label": label, "pnl": t["pnl"]})
        return pd.DataFrame(rows)

    # ── Print report ──────────────────────────────────────────────────────────

    def print_report(self):
        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        DIM    = "\033[2m"

        recents = self.state["recent_trades"]
        total   = self.state["total_trades"]
        if not recents:
            print(f"  [Adaptive] Belum ada data trade untuk dipelajari.")
            return

        wins     = sum(1 for t in recents[-20:] if t["result"] == "WIN")
        total_r  = min(len(recents), 20)
        win_rate = wins / total_r if total_r else 0

        wr_color = GREEN if win_rate >= 0.55 else (YELLOW if win_rate >= 0.40 else RED)

        print(f"\n  {'='*52}")
        print(f"  {BOLD}ADAPTIVE LEARNING REPORT{RESET}")
        print(f"  {'='*52}")
        print(f"  Total trade dipelajari : {total}")
        print(f"  Win Rate (20 terakhir) : {wr_color}{BOLD}{win_rate:.0%}{RESET} ({wins}/{total_r})")
        print(f"  Min Score saat ini     : {BOLD}{self.min_score}{RESET} "
              f"{DIM}(base: {self.state['base_min_score']}){RESET}")

        # Source performance
        if self.state["source_hits"]:
            print(f"\n  {BOLD}Performa per Sumber Sinyal:{RESET}")
            for src, sh in sorted(self.state["source_hits"].items()):
                w = sh["win"]
                l = sh["loss"]
                t = w + l
                wr = w / t if t else 0
                c = GREEN if wr >= 0.55 else (YELLOW if wr >= 0.40 else RED)
                print(f"    {src:<22} {c}{w}W/{l}L  ({wr:.0%}){RESET}")

        # Indicator performance (top/bottom 5)
        if self.state["indicator_hits"]:
            ranked = []
            for ind, hits in self.state["indicator_hits"].items():
                t = hits.get("correct", 0) + hits.get("wrong", 0)
                if t >= 10:
                    ranked.append((ind, hits["correct"] / t, t))
            ranked.sort(key=lambda x: -x[1])

            if ranked:
                print(f"\n  {BOLD}Indikator Terbaik (min 10 sample):{RESET}")
                for ind, acc, cnt in ranked[:5]:
                    c = GREEN if acc >= 0.60 else (YELLOW if acc >= 0.50 else RED)
                    mult = self.get_weight_multipliers().get(ind, 1.0)
                    mult_s = f"x{mult}" if mult != 1.0 else ""
                    print(f"    {ind:<18} acc={c}{acc:.0%}{RESET}  "
                          f"({cnt} sample)  {DIM}{mult_s}{RESET}")

        # Last adjustments
        adjs = self.state["score_adjustments"][-3:]
        if adjs:
            print(f"\n  {BOLD}Score Adjustments terakhir:{RESET}")
            for a in adjs:
                d = a["new"] - a["old"]
                dc = GREEN if d < 0 else RED
                print(f"    {a['time']}  {a['old']} -> {dc}{a['new']}{RESET}  {DIM}{a['reason']}{RESET}")

        print(f"  {'='*52}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_learner: AdaptiveLearner | None = None

def get_learner() -> AdaptiveLearner:
    global _learner
    if _learner is None:
        _learner = AdaptiveLearner()
    return _learner
