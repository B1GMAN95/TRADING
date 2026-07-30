import backtrader as bt
import pandas as pd

from backtesting.data_loader import load_csv
from backtesting.engine import run_backtest
from models.schemas.strategy import BacktestRequest
from strategies.registry import get_strategy
from strategies.smc_strategy import SMCStrategy
from tests.test_icc_strategy import _mirror

_STRATEGY_PARAMS = dict(
    swing_lookback=5,
    fvg_window=10,
    retest_timeout=10,
    min_fvg_atr_ratio=0.1,
    atr_period=4,
)

# A hand-crafted long setup: flat consolidation seeds swing_high/low around
# 100 (bar 6 becomes the Order Block), a strong impulse bar breaks the swing
# high (Market Structure Shift), the next bar's low gaps clear of the OB
# bar's high (Fair Value Gap), then price pulls back to retest the OB limit
# order before rallying through the take profit.
_LONG_ROWS = [
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.0, high=100.15, low=99.85, close=99.95, volume=1000),
    dict(open=99.95, high=100.1, low=99.9, close=100.05, volume=1000),
    dict(open=100.05, high=100.2, low=99.8, close=99.9, volume=1000),
    dict(open=99.9, high=100.1, low=99.85, close=100.0, volume=1000),
    dict(open=100.0, high=100.15, low=99.9, close=100.05, volume=1000),  # Order Block
    dict(open=100.05, high=110.5, low=100.0, close=110.0, volume=1000),  # MSS
    dict(open=110.0, high=115.5, low=109.0, close=115.0, volume=1000),  # FVG
    dict(open=115.0, high=115.2, low=108.0, close=109.0, volume=1000),
    dict(open=109.0, high=109.5, low=103.0, close=104.0, volume=1000),
    dict(open=104.0, high=104.5, low=100.05, close=100.15, volume=1000),  # OB retest fill
    dict(open=100.15, high=105.0, low=100.0, close=104.5, volume=1000),
    dict(open=104.5, high=112.0, low=104.0, close=111.0, volume=1000),  # take profit hit
]


def _run(rows: list[dict], strategy_cls=SMCStrategy, **extra_params) -> tuple[list, object]:
    dates = pd.date_range("2023-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, index=dates)

    params = {**_STRATEGY_PARAMS, **extra_params}
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, **params)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(10_000)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    analysis = results[0].analyzers.trades.get_analysis()
    return results, analysis


def test_smc_strategy_is_registered() -> None:
    assert get_strategy("smc_gold") is SMCStrategy


def test_smc_strategy_mss_fvg_ob_retest_enters_and_wins() -> None:
    _, analysis = _run(_LONG_ROWS)

    assert analysis["total"]["closed"] == 1
    assert analysis["long"]["total"] == 1
    assert analysis["pnl"]["net"]["total"] > 0


def test_smc_strategy_short_side_also_enters_and_wins() -> None:
    _, analysis = _run(_mirror(_LONG_ROWS))

    assert analysis["total"]["closed"] == 1
    assert analysis["short"]["total"] == 1
    assert analysis["pnl"]["net"]["total"] > 0


def test_smc_strategy_blocks_new_entries_once_daily_drawdown_locked() -> None:
    class PreLockedSMC(SMCStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.risk_manager.is_trading_allowed = lambda: False

    _, analysis = _run(_LONG_ROWS, strategy_cls=PreLockedSMC)

    assert analysis.get("total", {}).get("total", 0) == 0


def test_smc_strategy_cancels_unfilled_ob_retest_after_timeout() -> None:
    # Price never comes back down to retest the Order Block within the
    # timeout window, so the pending limit entry should be cancelled and the
    # strategy should go back to searching rather than leaving a stale order.
    rows = _LONG_ROWS[:8] + [
        dict(open=115.0, high=140.0, low=114.0, close=139.0, volume=1000)
        for _ in range(12)
    ]
    results, analysis = _run(rows, retest_timeout=5)

    assert analysis.get("total", {}).get("total", 0) == 0
    assert results[0].order is None


def test_smc_strategy_runs_via_engine_with_default_params_on_bundled_data() -> None:
    """Smoke test with the real defaults against the bundled intraday sample data."""
    price_data = load_csv("data/XAUUSD.csv")
    request = BacktestRequest(
        strategy_name="smc_gold",
        symbol="XAUUSD",
        start_date=str(price_data.index.min().date()),
        end_date=str(price_data.index.max().date()),
        cash=100_000,
    )
    result = run_backtest(request, price_data)

    assert result.strategy_name == "smc_gold"
    assert isinstance(result.ending_value, float)
    # This is the whole point of this strategy: fire far more often than
    # icc_gold's full trend/pullback cycle on the same intraday data.
    assert result.trades >= 30
