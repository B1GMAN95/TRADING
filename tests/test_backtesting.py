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
    assert len(result.equity_curve) == 120
    assert result.equity_curve[0].equity > 0
    assert result.equity_curve[-1].date == "2023-04-30"


def test_run_backtest_reports_win_rate_and_profit_factor_for_a_win() -> None:
    from tests.test_icc_strategy import _LONG_ROWS, _STRATEGY_PARAMS

    dates = pd.date_range("2023-01-01", periods=len(_LONG_ROWS), freq="D")
    price_data = pd.DataFrame(_LONG_ROWS, index=dates)

    request = BacktestRequest(
        strategy_name="icc_gold",
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-01-01",
        cash=10_000,
        parameters=_STRATEGY_PARAMS,
    )
    result = run_backtest(request, price_data)

    assert result.trades == 1
    assert result.win_rate_pct == 100.0
    assert result.profit_factor is None  # undefined with zero losing trades
    assert result.equity_curve[-1].equity > result.starting_cash
