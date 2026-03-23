from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class CandleOut(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    class Config:
        from_attributes = True
