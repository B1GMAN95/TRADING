import backtrader as bt
import numpy as np
import pandas as pd

from backtesting.data_loader import _is_intraday, add_price_feeds
from strategies.icc_strategy import ICCStrategy
from strategies.mtf_bias import mtf_bias_allows
from strategies.smc_strategy import SMCStrategy
from tests.test_icc_strategy import _LONG_ROWS, _mirror
from tests.test_icc_strategy import _STRATEGY_PARAMS as _ICC_PARAMS
from tests.test_smc_strategy import (
    _LONG_ROWS as _SMC_LONG_ROWS,
)
from tests.test_smc_strategy import (
    _STRATEGY_PARAMS as _SMC_PARAMS,
)

# -- mtf_bias_allows (pure function) --------------------------------------


def test_mtf_bias_allows_long_only_when_both_timeframes_bullish() -> None:
    assert mtf_bias_allows("long", "bullish", "bullish") is True
    assert mtf_bias_allows("long", "bullish", "bearish") is False
    assert mtf_bias_allows("long", "bearish", "bullish") is False
    assert mtf_bias_allows("long", "neutral", "bullish") is False


def test_mtf_bias_allows_short_only_when_both_timeframes_bearish() -> None:
    assert mtf_bias_allows("short", "bearish", "bearish") is True
    assert mtf_bias_allows("short", "bearish", "bullish") is False
    assert mtf_bias_allows("short", "neutral", "bearish") is False


# -- add_price_feeds / _is_intraday -----------------------------------------


def test_is_intraday_true_for_15_minute_bars() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="15min")
    df = pd.DataFrame({"close": range(10)}, index=dates)
    assert _is_intraday(df) is True


def test_is_intraday_false_for_daily_bars() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"close": range(10)}, index=dates)
    assert _is_intraday(df) is False


def test_add_price_feeds_skips_mtf_feeds_for_daily_data() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100}, index=dates
    )
    cerebro = bt.Cerebro()
    add_price_feeds(cerebro, ICCStrategy, df)
    assert len(cerebro.datas) == 1


def test_add_price_feeds_adds_mtf_feeds_for_intraday_data() -> None:
    dates = pd.date_range("2024-01-01", periods=1000, freq="15min")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100}, index=dates
    )
    cerebro = bt.Cerebro()
    add_price_feeds(cerebro, ICCStrategy, df)
    assert len(cerebro.datas) == 3


# -- Integration: MTF bias actually gates entries ---------------------------


def _sustained_uptrend(n_bars: int, seed: int = 3, start_price: float = 1900.0) -> list[dict]:
    """A long, low-noise uptrend: enough 15-min bars for a genuine 1H EMA200
    warmup (needs >=200 1H bars = 800 15-min bars) that stays consistently
    bullish on both the 1H trend and the 4H structure throughout.
    """
    rng = np.random.default_rng(seed)
    price = start_price
    bars = []
    for _ in range(n_bars):
        ret = 0.0007 + rng.normal(0, 0.0003)
        new_close = price * (1 + ret)
        open_ = price
        high = max(open_, new_close) * 1.0005
        low = min(open_, new_close) * 0.9995
        bars.append(dict(open=open_, high=high, low=low, close=new_close, volume=1000))
        price = new_close
    return bars


def _rescale(rows: list[dict], scale: float) -> list[dict]:
    return [{k: (v * scale if k != "volume" else v) for k, v in r.items()} for r in rows]


def _run_mtf(rows: list[dict], strategy_cls, params: dict) -> dict:
    dates = pd.date_range("2023-01-01", periods=len(rows), freq="15min")
    df = pd.DataFrame(rows, index=dates)

    cerebro = bt.Cerebro()
    add_price_feeds(cerebro, strategy_cls, df)
    cerebro.addstrategy(strategy_cls, **params)
    cerebro.broker.setcash(100_000)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    return results[0].analyzers.trades.get_analysis().get("total", {})


def test_icc_strategy_allows_entry_aligned_with_bullish_mtf_context() -> None:
    warmup = _sustained_uptrend(900)
    end_price = warmup[-1]["close"]
    long_setup = _rescale(_LONG_ROWS, end_price / 100.0)

    total = _run_mtf(warmup + long_setup, ICCStrategy, _ICC_PARAMS)
    assert total.get("total", 0) == 1


def test_icc_strategy_blocks_entry_misaligned_with_bullish_mtf_context() -> None:
    warmup = _sustained_uptrend(900)
    end_price = warmup[-1]["close"]
    # The mirrored (short) setup would fire on its own, but the surrounding
    # context here is a sustained uptrend - bullish on both 1H and 4H.
    short_setup = _rescale(_mirror(_LONG_ROWS), end_price / 100.0)

    total = _run_mtf(warmup + short_setup, ICCStrategy, _ICC_PARAMS)
    assert total.get("total", 0) == 0


def test_smc_strategy_allows_entry_aligned_with_bullish_mtf_context() -> None:
    warmup = _sustained_uptrend(900)
    end_price = warmup[-1]["close"]
    long_setup = _rescale(_SMC_LONG_ROWS, end_price / 100.0)

    total = _run_mtf(warmup + long_setup, SMCStrategy, _SMC_PARAMS)
    assert total.get("total", 0) == 1


def test_smc_strategy_blocks_entry_misaligned_with_bullish_mtf_context() -> None:
    warmup = _sustained_uptrend(900)
    end_price = warmup[-1]["close"]
    short_setup = _rescale(_mirror(_SMC_LONG_ROWS), end_price / 100.0)

    total = _run_mtf(warmup + short_setup, SMCStrategy, _SMC_PARAMS)
    assert total.get("total", 0) == 0


def test_icc_strategy_without_mtf_feeds_is_unaffected_by_bias_gate() -> None:
    """Backward compatibility: a single-feed setup (no 1H/4H context) trades
    exactly as it did before the MTF gate existed.
    """
    dates = pd.date_range("2023-01-01", periods=len(_LONG_ROWS), freq="D")
    df = pd.DataFrame(_LONG_ROWS, index=dates)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(ICCStrategy, **_ICC_PARAMS)
    cerebro.broker.setcash(10_000)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()

    assert results[0].mtf_bias is None
    assert results[0].analyzers.trades.get_analysis()["total"]["closed"] == 1
