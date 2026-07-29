from enum import Enum

from pydantic import BaseModel, Field


class MarketBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MarketAnalysis(BaseModel):
    bias: MarketBias
    confidence_score: float = Field(..., ge=0, le=1)
    trading_advice: str
    rationale: str | None = None
