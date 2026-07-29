import backtrader as bt
import numpy as np
import pandas as pd

from backtesting.engine import run_backtest
from models.schemas.strategy import BacktestRequest
from strategies.icc_strategy import ICCStrategy
from strategies.registry import get_strategy

_STRATEGY_PARAMS = dict(
    ema_fast1=3,
    ema_fast2=4,
    ema_trend=5,
    rsi_period=4,
    volume_period=4,
    volume_multiplier=1.3,
    zone_tolerance=0.02,
    max_bars_indication=50,
    max_bars_correction=50,
    atr_period=4,
)

# A hand-crafted long setup: flat warmup, an EMA200-breaking impulse bar with a
# volume spike (Indication), a pullback that dips RSI into oversold territory
# near the fib/EMA zone (Correction), then a break of the prior bar's high
# (Continuation) that should trigger a bracket entry. The final bars rally
# past the 1:2 take profit so the trade closes within the test window.
_LONG_ROWS = [
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.0, high=100.1, low=99.7, close=99.9, volume=1000),
    dict(open=100, high=103.2, low=99.9, close=103, volume=1600),  # Indication
    dict(open=103, high=106.3, low=102.8, close=106, volume=1000),
    dict(open=106, high=109.4, low=105.5, close=109, volume=1000),
    dict(open=109, high=109.5, low=105.8, close=106.5, volume=1000),
    dict(open=106.5, high=107, low=104.0, close=104.5, volume=1000),
    dict(open=104.5, high=105, low=101.8, close=102.0, volume=1000),
    dict(open=102.0, high=102.5, low=100.0, close=100.5, volume=1000),  # Correction
    dict(open=100.3, high=104.5, low=100.2, close=104.2, volume=1200),  # Continuation
    dict(open=104.2, high=107.5, low=104.0, close=107.0, volume=1000),
    dict(open=107.0, high=110.5, low=106.8, close=110.0, volume=1000),
    dict(open=110.0, high=114.0, low=109.5, close=113.5, volume=1000),  # take profit hit
]


def _mirror(rows: list[dict]) -> list[dict]:
    """Reflect a price series around 100 to derive an equivalent short setup."""
    return [
        dict(
            open=200 - r["open"],
            close=200 - r["close"],
            high=200 - r["low"],
            low=200 - r["high"],
            volume=r["volume"],
        )
        for r in rows
    ]


def _run(rows: list[dict]) -> dict:
    dates = pd.date_range("2023-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, index=dates)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(ICCStrategy, **_STRATEGY_PARAMS)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(10_000)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    return results[0].analyzers.trades.get_analysis()


def test_icc_strategy_is_registered() -> None:
    assert get_strategy("icc_gold") is ICCStrategy


def test_long_indication_correction_continuation_enters_and_wins() -> None:
    analysis = _run(_LONG_ROWS)

    assert analysis["total"]["closed"] == 1
    assert analysis["long"]["total"] == 1
    assert analysis["pnl"]["net"]["total"] > 0


def test_short_indication_correction_continuation_enters_and_wins() -> None:
    analysis = _run(_mirror(_LONG_ROWS))

    assert analysis["total"]["closed"] == 1
    assert analysis["short"]["total"] == 1
    assert analysis["pnl"]["net"]["total"] > 0


def test_icc_strategy_runs_via_engine_with_default_params() -> None:
    """Smoke test with the real EMA200/RSI14 defaults over a longer series."""
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    rng = np.random.default_rng(7)
    prices = 1900 + np.cumsum(rng.normal(0, 5, size=400))
    price_data = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(1, 5, 400),
            "low": prices - rng.uniform(1, 5, 400),
            "close": prices,
            "volume": rng.integers(800, 5000, size=400),
        },
        index=dates,
    )

    request = BacktestRequest(
        strategy_name="icc_gold",
        symbol="XAUUSD",
        start_date="2022-01-01",
        end_date="2023-02-04",
        cash=10_000,
    )
    result = run_backtest(request, price_data)

    assert result.strategy_name == "icc_gold"
    assert isinstance(result.ending_value, float)
