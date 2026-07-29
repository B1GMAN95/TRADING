from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    strategy_name: str = Field(..., examples=["sma_crossover"])
    symbol: str = Field(..., examples=["AAPL"])
    start_date: str = Field(..., examples=["2023-01-01"])
    end_date: str = Field(..., examples=["2023-12-31"])
    cash: float = Field(default=100_000.0, gt=0)
    commission: float = Field(default=0.001, ge=0)
    parameters: dict = Field(default_factory=dict)


class EquityPoint(BaseModel):
    date: str
    equity: float
    drawdown_pct: float


class BacktestResult(BaseModel):
    strategy_name: str
    symbol: str
    starting_cash: float
    ending_value: float
    total_return_pct: float
    trades: int
    win_rate_pct: float | None = None
    profit_factor: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown_pct: float | None = None
    equity_curve: list[EquityPoint] = Field(default_factory=list)
