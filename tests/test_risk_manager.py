from datetime import date

import backtrader as bt
import pandas as pd
import pytest

from api.risk_manager import (
    RiskManager,
    RiskManagerError,
    breakeven_stop_price,
    calculate_position_size,
)
from strategies.icc_strategy import ICCStrategy
from tests.test_icc_strategy import _LONG_ROWS, _STRATEGY_PARAMS, _mirror

# -- calculate_position_size ---------------------------------------------


def test_calculate_position_size_matches_expected_formula() -> None:
    size = calculate_position_size(
        account_balance=10_000, atr=5.0, risk_pct=0.01, atr_multiplier=1.5
    )
    # risk_amount = 10_000 * 0.01 = 100; stop_distance = 5.0 * 1.5 = 7.5
    assert size == pytest.approx(100 / 7.5)


def test_calculate_position_size_scales_with_risk_pct() -> None:
    small = calculate_position_size(account_balance=10_000, atr=5.0, risk_pct=0.01)
    large = calculate_position_size(account_balance=10_000, atr=5.0, risk_pct=0.02)
    assert large == pytest.approx(small * 2)


def test_calculate_position_size_caps_at_what_the_account_can_afford() -> None:
    # A tiny ATR (e.g. on a lower timeframe) implies a huge position; without
    # a price-based cap this would be unaffordable and get rejected as margin.
    uncapped = calculate_position_size(account_balance=10_000, atr=0.05, risk_pct=0.01)
    capped = calculate_position_size(account_balance=10_000, atr=0.05, risk_pct=0.01, price=2000)

    assert uncapped * 2000 > 10_000  # confirms the scenario is actually unaffordable
    assert capped == pytest.approx((10_000 * 0.95) / 2000)


def test_calculate_position_size_price_cap_is_a_noop_when_already_affordable() -> None:
    # size = 100 / 7.5 ≈ 13.33 units; at price=1 that's ~$13.33 notional,
    # nowhere near the $10,000 balance, so the cap shouldn't change anything.
    uncapped = calculate_position_size(account_balance=10_000, atr=5.0, risk_pct=0.01)
    capped = calculate_position_size(account_balance=10_000, atr=5.0, risk_pct=0.01, price=1)

    assert capped == pytest.approx(uncapped)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(account_balance=0, atr=5.0),
        dict(account_balance=-100, atr=5.0),
        dict(account_balance=10_000, atr=0),
        dict(account_balance=10_000, atr=-1),
        dict(account_balance=10_000, atr=5.0, risk_pct=0),
        dict(account_balance=10_000, atr=5.0, risk_pct=1.5),
        dict(account_balance=10_000, atr=5.0, price=0),
        dict(account_balance=10_000, atr=5.0, price=-10),
    ],
)
def test_calculate_position_size_rejects_invalid_input(kwargs: dict) -> None:
    with pytest.raises(RiskManagerError):
        calculate_position_size(**kwargs)


# -- breakeven_stop_price --------------------------------------------------


def test_breakeven_stop_price_triggers_at_1_to_1_for_long() -> None:
    kwargs = dict(direction="long", entry_price=100, initial_stop_price=95)
    assert breakeven_stop_price(current_price=105, **kwargs) == 100
    assert breakeven_stop_price(current_price=110, **kwargs) == 100


def test_breakeven_stop_price_not_yet_reached_for_long() -> None:
    kwargs = dict(direction="long", entry_price=100, initial_stop_price=95)
    assert breakeven_stop_price(current_price=104.9, **kwargs) is None


def test_breakeven_stop_price_triggers_at_1_to_1_for_short() -> None:
    kwargs = dict(direction="short", entry_price=100, initial_stop_price=105)
    assert breakeven_stop_price(current_price=95, **kwargs) == 100


def test_breakeven_stop_price_not_yet_reached_for_short() -> None:
    kwargs = dict(direction="short", entry_price=100, initial_stop_price=105)
    assert breakeven_stop_price(current_price=95.1, **kwargs) is None


def test_breakeven_stop_price_handles_zero_risk() -> None:
    kwargs = dict(direction="long", entry_price=100, initial_stop_price=100)
    assert breakeven_stop_price(current_price=110, **kwargs) is None


# -- RiskManager (daily max drawdown lockout) ------------------------------


def test_risk_manager_starts_unlocked() -> None:
    rm = RiskManager(daily_max_drawdown_pct=3.0)
    rm.update(10_000, date(2024, 1, 1))
    assert rm.is_trading_allowed() is True


def test_risk_manager_locks_once_daily_drawdown_breached() -> None:
    rm = RiskManager(daily_max_drawdown_pct=3.0)
    rm.update(10_000, date(2024, 1, 1))  # sets the day's starting balance
    rm.update(9_800, date(2024, 1, 1))  # -2%, still allowed
    assert rm.is_trading_allowed() is True
    rm.update(9_650, date(2024, 1, 1))  # -3.5%, breach
    assert rm.is_trading_allowed() is False


def test_risk_manager_stays_locked_for_rest_of_day_even_if_equity_recovers() -> None:
    rm = RiskManager(daily_max_drawdown_pct=3.0)
    rm.update(10_000, date(2024, 1, 1))
    rm.update(9_600, date(2024, 1, 1))  # breach -> locked
    assert rm.is_trading_allowed() is False
    rm.update(10_500, date(2024, 1, 1))  # recovers, but still the same day
    assert rm.is_trading_allowed() is False


def test_risk_manager_unlocks_on_the_next_trading_day() -> None:
    rm = RiskManager(daily_max_drawdown_pct=3.0)
    rm.update(10_000, date(2024, 1, 1))
    rm.update(9_600, date(2024, 1, 1))  # breach -> locked
    assert rm.is_trading_allowed() is False

    rm.update(9_600, date(2024, 1, 2))  # new day: resets, even at the same equity
    assert rm.is_trading_allowed() is True
    assert rm.daily_start_balance == 9_600


# -- Integration: the risk engine actually gates ICCStrategy ---------------


def _run(rows: list[dict], strategy_cls=ICCStrategy, **extra_params) -> tuple[list, object]:
    dates = pd.date_range("2023-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, index=dates)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, **_STRATEGY_PARAMS, **extra_params)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(10_000)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    analysis = results[0].analyzers.trades.get_analysis()
    return results, analysis


def test_icc_strategy_moves_stop_to_breakeven_before_closing() -> None:
    results, analysis = _run(_LONG_ROWS)

    # breakeven_triggered itself resets once the position closes and next()
    # clears position tracking again, so use the lifetime counter instead.
    assert results[0].breakeven_count == 1
    assert analysis["total"]["closed"] == 1
    assert analysis["pnl"]["net"]["total"] > 0


def test_icc_strategy_sizes_positions_via_atr_and_risk_pct() -> None:
    # Wrap _enter to capture the ATR/balance it saw and the size it ultimately
    # submitted, without duplicating the strategy's own sizing logic.
    original_enter = ICCStrategy._enter
    captured = {}

    class Capturing(ICCStrategy):
        def _enter(self, direction: str) -> None:
            captured["atr"] = self.atr[0]
            captured["balance"] = self.broker.getvalue()
            original_enter(self, direction)
            # the entry order is still pending at this point (fills next bar),
            # so read the size it was submitted with rather than self.position.
            captured["size"] = abs(self.order.size)

    _run(_LONG_ROWS, strategy_cls=Capturing)

    expected_size = calculate_position_size(
        account_balance=captured["balance"],
        atr=captured["atr"],
        risk_pct=_STRATEGY_PARAMS.get("risk_pct", 0.01),
        atr_multiplier=_STRATEGY_PARAMS.get("atr_multiplier", 1.5),
    )
    assert captured["size"] == pytest.approx(expected_size)
    assert captured["size"] != 1  # would be 1 if the default Backtrader sizer were used


def test_icc_strategy_blocks_new_entries_once_daily_drawdown_locked() -> None:
    class PreLockedICC(ICCStrategy):
        def __init__(self) -> None:
            super().__init__()
            # Force the gate closed regardless of RiskManager's own day-rollover
            # bookkeeping (covered separately above) - this isolates exactly
            # one thing: does ICCStrategy actually respect the gate.
            self.risk_manager.is_trading_allowed = lambda: False

    _, analysis = _run(_LONG_ROWS, strategy_cls=PreLockedICC)

    assert analysis.get("total", {}).get("total", 0) == 0


def test_icc_strategy_short_side_also_moves_to_breakeven() -> None:
    results, analysis = _run(_mirror(_LONG_ROWS))

    assert results[0].breakeven_count == 1
    assert analysis["total"]["closed"] == 1
