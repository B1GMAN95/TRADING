import numpy as np
import pandas as pd

from backtesting.engine import run_backtest
from models.schemas.strategy import BacktestRequest


def _make_price_data(periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(seed=42)
    prices = 100 + np.cumsum(rng.normal(0, 1, size=periods))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1_000,
        },
        index=dates,
    )


def test_run_backtest_sma_crossover() -> None:
    request = BacktestRequest(
        strategy_name="sma_crossover",
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-05-01",
        cash=10_000,
    )
    result = run_backtest(request, _make_price_data())

    assert result.strategy_name == "sma_crossover"
    assert result.starting_cash == 10_000
    assert isinstance(result.ending_value, float)
