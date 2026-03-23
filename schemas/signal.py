from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SignalOut(BaseModel):
    symbol: str
    timeframe: str
    timestamp: Optional[str] = None
    direction: str
    score: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr_ratio: Optional[float] = None
    close: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    ml_direction: Optional[str] = None
    ml_confidence: Optional[float] = None
    news_risk: Optional[str] = None
    exec_direction: str
    exec_source: Optional[str] = None
    final_advice: Optional[str] = None

    class Config:
        from_attributes = True


class SignalHistoryOut(BaseModel):
    id: int
    symbol: str
    timeframe: str
    timestamp: Optional[datetime] = None
    direction: str
    score: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    exec_direction: str
    news_risk: Optional[str] = None
    ml_direction: Optional[str] = None
    ml_confidence: Optional[float] = None

    class Config:
        from_attributes = True
