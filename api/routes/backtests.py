from fastapi import APIRouter, HTTPException

from backtesting.data_loader import load_csv
from backtesting.engine import run_backtest
from models.schemas.strategy import BacktestRequest, BacktestResult

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=BacktestResult)
def create_backtest(request: BacktestRequest) -> BacktestResult:
    try:
        price_data = load_csv(f"data/{request.symbol}.csv")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No price data for {request.symbol}") from exc

    try:
        return run_backtest(request, price_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
