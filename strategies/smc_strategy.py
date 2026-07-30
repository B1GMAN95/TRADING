import backtrader as bt

from api.risk_manager import RiskManager, breakeven_stop_price, calculate_position_size
from strategies.base import BaseStrategy


class SMCStrategy(BaseStrategy):
    """Simplified Smart Money Concepts (SMC/ICT-style) strategy for intraday gold.

    1. Market Structure Shift (MSS): price closes beyond the most recent
       swing high/low, signaling a potential change of character.
    2. Fair Value Gap (FVG): within a short window after the MSS, a 3-candle
       imbalance forms in the direction of the shift - a gap between the
       high two bars back and the current low (bullish), or the low two
       bars back and the current high (bearish).
    3. Order Block (OB) retest: a limit order is placed at the body of the
       last opposite-colored candle before the FVG's impulse candle, with a
       stop beyond that candle's liquidity extreme. If price never retests
       the OB within `retest_timeout` bars, the order is cancelled.

    Because MSS and FVGs are common, low-bar-count patterns compared to
    icc_gold's full trend/pullback cycle, this strategy fires considerably
    more often - by design, for higher-frequency intraday trading.

    Every order goes through the global risk engine (api/risk_manager.py):
    ATR/1%-risk position sizing, a daily max-drawdown lockout, and a
    breakeven stop once a trade reaches 1:1 reward-to-risk.
    """

    params = (
        ("swing_lookback", 20),
        ("fvg_window", 15),
        # Minimum FVG gap size, as a multiple of ATR. Filters out noise-sized
        # "gaps" that don't represent a genuine displacement/imbalance.
        ("min_fvg_atr_ratio", 0.65),
        ("retest_timeout", 20),
        ("sl_buffer", 0.001),
        ("risk_reward_ratio", 2.0),
        ("atr_period", 14),
        ("risk_pct", 0.01),
        ("atr_multiplier", 1.5),
        ("daily_max_drawdown_pct", 3.0),
    )

    STATE_SEARCH_MSS = "search_mss"
    STATE_WAIT_FVG = "wait_fvg"

    def __init__(self) -> None:
        self.swing_high = bt.ind.Highest(self.data.high, period=self.p.swing_lookback)
        self.swing_low = bt.ind.Lowest(self.data.low, period=self.p.swing_lookback)
        self.atr = bt.ind.ATR(period=self.p.atr_period)

        self.risk_manager = RiskManager(daily_max_drawdown_pct=self.p.daily_max_drawdown_pct)
        self.breakeven_count = 0  # lifetime counter; not reset between trades

        self.order: bt.Order | None = None
        self._order_submitted_bar: int | None = None
        self._reset_setup()
        self._clear_position_tracking()

    def _reset_setup(self) -> None:
        self.state = self.STATE_SEARCH_MSS
        self.direction: str | None = None
        self.mss_bar: int | None = None
        self.structure_reference: float | None = None

    def _clear_position_tracking(self) -> None:
        self.position_direction: str | None = None
        self.entry_price: float | None = None
        self.initial_stop_price: float | None = None
        self.take_profit_price: float | None = None
        self.stop_order: bt.Order | None = None
        self.limit_order: bt.Order | None = None
        self.breakeven_triggered = False

    def notify_order(self, order: bt.Order) -> None:
        super().notify_order(order)
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            if order == self.order:
                self.order = None
                self._order_submitted_bar = None

    def next(self) -> None:
        self.risk_manager.update(self.broker.getvalue(), self.data.datetime.date(0))

        if self.position:
            self._manage_open_position()
            return

        self._clear_position_tracking()

        if self.order:
            if self._entry_expired():
                self.cancel(self.order)
            return

        if not self.risk_manager.is_trading_allowed():
            return

        if self.state == self.STATE_SEARCH_MSS:
            self._check_mss()
        elif self.state == self.STATE_WAIT_FVG:
            if self._fvg_window_expired() or self._structure_broken():
                self._reset_setup()
            else:
                self._check_fvg()

    def _entry_expired(self) -> bool:
        return (
            self._order_submitted_bar is not None
            and len(self) - self._order_submitted_bar > self.p.retest_timeout
        )

    # -- Risk management (breakeven, same pattern as ICCStrategy) --------

    def _manage_open_position(self) -> None:
        if self.breakeven_triggered or self.entry_price is None or self.stop_order is None:
            return

        new_stop = breakeven_stop_price(
            self.position_direction, self.entry_price, self.initial_stop_price, self.data.close[0]
        )
        if new_stop is None:
            return

        self.cancel(self.stop_order)
        size = abs(self.position.size)
        close_position = self.sell if self.position_direction == "long" else self.buy

        new_stop_order = close_position(exectype=bt.Order.Stop, price=new_stop, size=size)
        new_limit_order = close_position(
            exectype=bt.Order.Limit, price=self.take_profit_price, size=size, oco=new_stop_order
        )
        self.stop_order = new_stop_order
        self.limit_order = new_limit_order
        self.breakeven_triggered = True
        self.breakeven_count += 1
        self.log(f"Breakeven: stop moved to entry at {new_stop:.2f}")

    # -- 1. Market Structure Shift -----------------------------------------

    def _check_mss(self) -> None:
        prior_swing_high = self.swing_high[-1]
        prior_swing_low = self.swing_low[-1]

        if self.data.close[0] > prior_swing_high:
            self._start_mss("long", prior_swing_low)
        elif self.data.close[0] < prior_swing_low:
            self._start_mss("short", prior_swing_high)

    def _start_mss(self, direction: str, structure_reference: float) -> None:
        self.direction = direction
        self.structure_reference = structure_reference
        self.mss_bar = len(self)
        self.state = self.STATE_WAIT_FVG
        self.log(f"MSS ({direction}) at {self.data.close[0]:.2f}")

    def _fvg_window_expired(self) -> bool:
        return len(self) - self.mss_bar > self.p.fvg_window

    def _structure_broken(self) -> bool:
        if self.direction == "long":
            return self.data.close[0] < self.structure_reference
        return self.data.close[0] > self.structure_reference

    # -- 2 & 3. Fair Value Gap + Order Block retest ------------------------

    def _check_fvg(self) -> None:
        if len(self) < 3:
            return

        min_gap = self.atr[0] * self.p.min_fvg_atr_ratio
        if self.direction == "long":
            gap = self.data.low[0] - self.data.high[-2]
            if gap > min_gap:
                self._enter_at_order_block("long")
        else:
            gap = self.data.low[-2] - self.data.high[0]
            if gap > min_gap:
                self._enter_at_order_block("short")

    def _enter_at_order_block(self, direction: str) -> None:
        """Place a limit order at the last opposite-colored candle (the Order
        Block) before the FVG's impulse candle, i.e. two bars back.
        """
        ob_open = self.data.open[-2]
        ob_close = self.data.close[-2]
        ob_high = self.data.high[-2]
        ob_low = self.data.low[-2]

        if direction == "long":
            entry_price = max(ob_open, ob_close)
            stop_price = ob_low * (1 - self.p.sl_buffer)
            risk = entry_price - stop_price
        else:
            entry_price = min(ob_open, ob_close)
            stop_price = ob_high * (1 + self.p.sl_buffer)
            risk = stop_price - entry_price

        if risk <= 0:
            self._reset_setup()
            return

        take_profit = (
            entry_price + risk * self.p.risk_reward_ratio
            if direction == "long"
            else entry_price - risk * self.p.risk_reward_ratio
        )

        size = calculate_position_size(
            account_balance=self.broker.getvalue(),
            atr=self.atr[0],
            risk_pct=self.p.risk_pct,
            atr_multiplier=self.p.atr_multiplier,
            price=entry_price,
        )

        bracket = self.buy_bracket if direction == "long" else self.sell_bracket
        orders = bracket(
            price=entry_price,
            exectype=bt.Order.Limit,
            size=size,
            stopprice=stop_price,
            limitprice=take_profit,
        )
        self.order = orders[0]
        self.stop_order = orders[1]
        self.limit_order = orders[2]
        self._order_submitted_bar = len(self)
        self.position_direction = direction
        self.entry_price = entry_price
        self.initial_stop_price = stop_price
        self.take_profit_price = take_profit
        self.breakeven_triggered = False
        self.log(
            f"{direction.upper()} OB retest limit={entry_price:.2f} "
            f"SL={stop_price:.2f} TP={take_profit:.2f} size={size:.4f}"
        )
        self._reset_setup()
