"""
WebSocket /ws/live — stream data real-time ke FE
Kirim update setiap 5 detik: harga, sinyal, posisi, P&L
"""
import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger("trader_ai.routes.ws")

# ── Connection manager ────────────────────────────────────────────────────────
class _Manager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        logger.info(f"WS connected — total: {len(self.connections)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
        logger.info(f"WS disconnected — total: {len(self.connections)}")

    async def broadcast(self, data: dict):
        payload = json.dumps(data, ensure_ascii=False, default=str)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = _Manager()


@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Stream real-time ke FE. Payload setiap 5 detik:
    {
      "type": "update",
      "timestamp": "...",
      "price": 3285.5,
      "signal": { direction, prob_buy, prob_sell, confidence },
      "positions": [ ... ],
      "account": { balance, equity, profit },
      "daily": { win, loss, net_pnl },
      "circuit": { circuit_breaker_active, consec_loss_buy, ... },
      "scalp": { direction, prob_buy, confidence, notes, warnings }
    }
    """
    from app.services.bot_service import bot_service

    await manager.connect(ws)
    try:
        while True:
            try:
                # Kumpulkan semua data
                last = bot_service.get_last_signal()
                scalp = bot_service.get_scalp_signal()
                positions = bot_service.get_positions()
                account = bot_service.get_account()
                daily = bot_service.get_daily_stats()
                circuit = bot_service.get_circuit_status()

                sig = last.get("signal", {})
                payload = {
                    "type"      : "update",
                    "timestamp" : datetime.now().isoformat(),
                    "price"     : last.get("close") or scalp.get("close"),
                    "signal"    : {
                        "direction" : sig.get("direction", "WAIT"),
                        "score"     : sig.get("score", 0),
                        "sl"        : sig.get("sl"),
                        "tp"        : sig.get("tp"),
                    },
                    "scalp"     : {
                        "direction"      : scalp.get("direction", "WAIT"),
                        "prob_buy"       : scalp.get("prob_buy", 0),
                        "prob_sell"      : scalp.get("prob_sell", 0),
                        "confidence"     : scalp.get("confidence", ""),
                        "sl"             : scalp.get("sl"),
                        "tp"             : scalp.get("tp"),
                        "rr"             : scalp.get("rr"),
                        "notes"          : scalp.get("signal_notes", []),
                        "warnings"       : scalp.get("signal_warnings", []),
                        "swing_based"    : scalp.get("swing_based", False),
                    },
                    "positions" : positions,
                    "account"   : {
                        "balance": account.get("balance", 0),
                        "equity" : account.get("equity", 0),
                        "profit" : account.get("profit", 0),
                    },
                    "daily"     : daily,
                    "circuit"   : circuit,
                    "bot_cycle" : bot_service.cycle,
                    "paused"    : getattr(bot_service, '_paused', False),
                }

                await ws.send_text(json.dumps(payload, ensure_ascii=False, default=str))

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"WS send error: {e}")
                try:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
                except Exception:
                    break

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
