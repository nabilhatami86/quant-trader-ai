from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime


class ForceTradeRequest(BaseModel):
    direction: str   # BUY atau SELL
    symbol: str = "XAUUSD"
    lot: float = 0.01

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        v = v.upper()
        if v not in ("BUY", "SELL"):
            raise ValueError("direction harus BUY atau SELL")
        return v


class TradeOut(BaseModel):
    id: int
    symbol: str
    timeframe: str
    direction: str
    entry: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: float
    ticket: Optional[int] = None
    close_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    result: str
    source: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TradeStatsOut(BaseModel):
    total: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
