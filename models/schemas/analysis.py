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
    # Confluence across the 4H/1H/15M/5M matrix (strategies/mtf_bias.py),
    # 0 = no timeframes agree, 1 = full alignment. None when no
    # multi-timeframe matrix was given to JarvisBrain.analyze_market().
    mtf_alpha_score: float | None = Field(default=None, ge=0, le=1)
