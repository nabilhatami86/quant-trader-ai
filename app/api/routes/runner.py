"""
/runner — Manage subprocess main.py (start, stop, status, logs)

Endpoint ini memungkinkan API untuk menjalankan main.py sebagai subprocess
terpisah, persis seperti: python -X utf8 main.py --symbol XAUUSD --tf 5m --live --mt5 --real
"""
import logging
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.utils.response import ok, err

logger = logging.getLogger("trader_ai.routes.runner")
router = APIRouter(prefix="/runner", tags=["Runner"])

# ── State subprocess (singleton, satu proses pada satu waktu) ─────────────────

_lock      = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_log_lines: deque = deque(maxlen=200)   # ring buffer 200 baris log terakhir
_start_time: Optional[float] = None
_last_args: dict = {}

# Path ke direktori trader-ai (root project, tempat main.py berada)
_ROOT = Path(__file__).resolve().parents[3]   # trader-ai/


def _stream_logs(proc: subprocess.Popen):
    """Thread: baca stdout+stderr subprocess dan simpan ke _log_lines."""
    try:
        for line in proc.stdout:                 # type: ignore[union-attr]
            stripped = line.rstrip("\n")
            _log_lines.append(stripped)
    except Exception:
        pass
    finally:
        _log_lines.append("[runner] Process selesai / mati.")


# ── Schema ────────────────────────────────────────────────────────────────────

class RunnerStartBody(BaseModel):
    symbol: str   = "XAUUSD"
    tf: str       = "5m"
    live: bool    = True
    mt5: bool     = True
    real: bool    = False
    micro: bool   = False
    no_news: bool = False
    no_ml: bool   = False
    trail: float  = 0.0
    orders: int   = 15
    lot: float    = 0.01
    risk: float   = 0.0
    dca: float    = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_running() -> bool:
    global _proc
    if _proc is None:
        return False
    if _proc.poll() is not None:      # proses sudah selesai
        _proc = None
        return False
    return True


def _build_cmd(body: RunnerStartBody) -> list[str]:
    cmd = [sys.executable, "-X", "utf8", "main.py",
           "--symbol", body.symbol,
           "--tf",     body.tf,
           "--orders", str(body.orders),
           "--lot",    str(body.lot),
    ]
    if body.live:     cmd.append("--live")
    if body.mt5:      cmd.append("--mt5")
    if body.real:     cmd.append("--real")
    if body.micro:    cmd.append("--micro")
    if body.no_news:  cmd.append("--no-news")
    if body.no_ml:    cmd.append("--no-ml")
    if body.trail > 0:    cmd += ["--trail",  str(body.trail)]
    if body.risk  > 0:    cmd += ["--risk",   str(body.risk)]
    if body.dca   > 0:    cmd += ["--dca",    str(body.dca)]
    return cmd


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", summary="Status proses main.py")
async def get_status():
    running = _is_running()
    uptime  = round(time.time() - _start_time, 1) if _start_time and running else 0
    return ok({
        "running"   : running,
        "pid"       : _proc.pid if running and _proc else None,
        "uptime_sec": uptime,
        "args"      : _last_args,
    })


@router.post("/start", summary="Jalankan main.py sebagai subprocess")
async def start_runner(body: RunnerStartBody):
    global _proc, _start_time, _last_args, _log_lines

    with _lock:
        if _is_running():
            return err("main.py sudah berjalan — stop dulu sebelum start ulang")

        cmd = _build_cmd(body)
        _log_lines.clear()
        _last_args = body.model_dump()

        logger.info(f"Menjalankan: {' '.join(cmd)}")
        try:
            _proc = subprocess.Popen(
                cmd,
                cwd=str(_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            logger.error(f"Gagal start main.py: {exc}")
            return err(f"Gagal start: {exc}")

        _start_time = time.time()
        _log_lines.append(f"[runner] Dimulai: {' '.join(cmd)}")

        # Mulai thread pembaca log
        t = threading.Thread(target=_stream_logs, args=(_proc,), daemon=True)
        t.start()

    return ok({
        "pid"    : _proc.pid,
        "cmd"    : " ".join(cmd),
        "message": "main.py berhasil dijalankan",
    })


@router.post("/stop", summary="Hentikan main.py yang sedang berjalan")
async def stop_runner():
    global _proc, _start_time

    with _lock:
        if not _is_running():
            return err("main.py tidak sedang berjalan")

        pid = _proc.pid
        logger.info(f"Menghentikan main.py (PID {pid})")
        try:
            _proc.terminate()
            try:
                _proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                _proc.kill()
        except Exception as exc:
            logger.warning(f"Gagal terminate: {exc}")

        _proc       = None
        _start_time = None
        _log_lines.append("[runner] Proses dihentikan.")

    return ok({"pid": pid, "message": "main.py berhasil dihentikan"})


@router.get("/logs", summary="Ambil 200 baris log terakhir dari main.py")
async def get_logs(limit: int = 100):
    lines = list(_log_lines)
    return ok({
        "count": len(lines),
        "logs" : lines[-limit:],
    })
