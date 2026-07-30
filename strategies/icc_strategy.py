import backtrader as bt

from api.risk_manager import RiskManager, breakeven_stop_price, calculate_position_size
from strategies.base import BaseStrategy
from strategies.mtf_bias import MTFBias


class ICCStrategy(BaseStrategy):
    """Indication-Correction-Continuation strategy, designed for gold (XAU/USD)
    on intraday (5m/15m) timeframes by default - see the params below for the
    EMA/RSI tuning that assumes. Note: "5m" here is currently the same 15M
    feed as everything else - a genuinely finer feed would need price_data
    itself re-sourced at 5-minute granularity (see backtesting/data_loader.py).

    1. Indication (Impuls): price breaks strongly through the trend EMA (up or
       down) on above-average volume, signaling an impulsive move.
    2. Correction (Rekyl): price pulls back to a faster EMA or a Fibonacci
       retracement of the impulse leg, while RSI becomes momentarily
       overbought/oversold.
    3. Continuation (Fortsettelse): an engulfing candle, or a break of the
       prior bar's high/low in the trend direction, confirms the pullback
       is over.

    On continuation the strategy submits a bracket order: the stop loss is
    placed just beyond the correction's extreme, and the take profit is set
    at a minimum 1:2 reward-to-risk ratio. Every order is routed through the
    global risk engine (api/risk_manager.py): position size is derived from
    ATR and a fixed 1% account risk, new entries are blocked once a day's
    losses breach the daily max-drawdown limit, and the stop is moved to
    breakeven once a trade reaches 1:1 reward-to-risk.

    When given 1H/4H feeds (strategies/mtf_bias.py, typically added via
    cerebro.resampledata()), a setup is only started - and re-checked before
    the final entry - if its direction matches both the 1H trend and 4H
    market structure. Without those extra feeds, this gate is a no-op, so
    single-feed callers/tests are unaffected.
    """

    requires_mtf = True

    params = (
        # Shortened from 20/50/200 for intraday (5m/15m) timeframes, where a
        # 200-bar trend EMA is too slow to react within a single session.
        ("ema_fast1", 10),
        ("ema_fast2", 25),
        ("ema_trend", 50),
        ("rsi_period", 14),
        # Loosened from 30/70 so intraday pullbacks (which are shallower than
        # daily-bar corrections) actually qualify as a Correction.
        ("rsi_oversold", 45),
        ("rsi_overbought", 55),
        ("volume_period", 20),
        ("volume_multiplier", 1.5),
        ("fib_level", 0.5),
        ("zone_tolerance", 0.0015),
        ("sl_buffer", 0.001),
        ("risk_reward_ratio", 2.0),
        ("min_risk_reward_ratio", 2.0),
        ("max_bars_indication", 50),
        ("max_bars_correction", 30),
        ("atr_period", 14),
        ("risk_pct", 0.01),
        ("atr_multiplier", 1.5),
        ("daily_max_drawdown_pct", 3.0),
    )

    STATE_SEARCH_INDICATION = "search_indication"
    STATE_WAIT_CORRECTION = "wait_correction"
    STATE_WAIT_CONTINUATION = "wait_continuation"

    def __init__(self) -> None:
        self.ema_fast1 = bt.ind.EMA(period=self.p.ema_fast1)
        self.ema_fast2 = bt.ind.EMA(period=self.p.ema_fast2)
        self.ema_trend = bt.ind.EMA(period=self.p.ema_trend)
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)
        self.volume_ma = bt.ind.SMA(self.data.volume, period=self.p.volume_period)
        self.trend_cross = bt.ind.CrossOver(self.data.close, self.ema_trend)
        self.atr = bt.ind.ATR(period=self.p.atr_period)

        self.mtf_bias = MTFBias(self) if len(self.datas) >= 3 else None

        self.risk_manager = RiskManager(daily_max_drawdown_pct=self.p.daily_max_drawdown_pct)
        self.breakeven_count = 0  # lifetime counter; not reset between trades

        self.order: bt.Order | None = None
        self._reset_setup()
        self._clear_position_tracking()

    def _reset_setup(self) -> None:
        self.state = self.STATE_SEARCH_INDICATION
        self.direction: str | None = None
        self.impulse_start_price: float | None = None
        self.impulse_extreme: float | None = None
        self.structure_reference: float | None = None
        self.indication_bar: int | None = None
        self.correction_extreme: float | None = None
        self.correction_bar: int | None = None

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

    def _mtf_allows(self, direction: str) -> bool:
        return self.mtf_bias is None or self.mtf_bias.allows(direction)

    def next(self) -> None:
        self.risk_manager.update(self.broker.getvalue(), self.data.datetime.date(0))
        if self.mtf_bias is not None:
            self.mtf_bias.update()

        if self.position:
            self._manage_open_position()
            return

        self._clear_position_tracking()

        if self.order:
            return

        if not self.risk_manager.is_trading_allowed():
            return

        if self.state == self.STATE_SEARCH_INDICATION:
            self._check_indication()
        elif self.state == self.STATE_WAIT_CORRECTION:
            self._update_impulse_extreme()
            if self._indication_expired() or self._structure_broken():
                self._reset_setup()
            else:
                self._check_correction()
        elif self.state == self.STATE_WAIT_CONTINUATION:
            self._update_correction_extreme()
            if self._correction_expired() or self._structure_broken():
                self._reset_setup()
            else:
                self._check_continuation()

    # -- Risk management ------------------------------------------------

    def _manage_open_position(self) -> None:
        """Move the stop to breakeven once the trade reaches 1:1 reward-to-risk."""
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

    # -- 1. Indication -------------------------------------------------

    def _check_indication(self) -> None:
        volume_confirmed = self.data.volume[0] > self.volume_ma[0] * self.p.volume_multiplier
        if not volume_confirmed:
            return

        if self.trend_cross[0] > 0 and self._mtf_allows("long"):
            self._start_indication("long")
        elif self.trend_cross[0] < 0 and self._mtf_allows("short"):
            self._start_indication("short")

    def _start_indication(self, direction: str) -> None:
        self.direction = direction
        self.impulse_start_price = self.data.close[0]
        if direction == "long":
            self.impulse_extreme = self.data.high[0]
            self.structure_reference = self.data.low[0]
        else:
            self.impulse_extreme = self.data.low[0]
            self.structure_reference = self.data.high[0]
        self.indication_bar = len(self)
        self.state = self.STATE_WAIT_CORRECTION
        self.log(f"Indication ({direction}) at {self.data.close[0]:.2f}")

    def _indication_expired(self) -> bool:
        return len(self) - self.indication_bar > self.p.max_bars_indication

    def _update_impulse_extreme(self) -> None:
        if self.direction == "long":
            self.impulse_extreme = max(self.impulse_extreme, self.data.high[0])
        else:
            self.impulse_extreme = min(self.impulse_extreme, self.data.low[0])

    # -- 2. Correction ---------------------------------------------------

    def _fib_level(self) -> float:
        span = abs(self.impulse_extreme - self.impulse_start_price)
        if self.direction == "long":
            return self.impulse_extreme - span * self.p.fib_level
        return self.impulse_extreme + span * self.p.fib_level

    def _near(self, price: float, level: float) -> bool:
        return level != 0 and abs(price - level) / level <= self.p.zone_tolerance

    def _check_correction(self) -> None:
        price = self.data.close[0]
        in_zone = (
            self._near(price, self.ema_fast1[0])
            or self._near(price, self.ema_fast2[0])
            or self._near(price, self._fib_level())
        )
        if not in_zone:
            return

        if self.direction == "long" and self.rsi[0] < self.p.rsi_oversold:
            self._start_correction()
        elif self.direction == "short" and self.rsi[0] > self.p.rsi_overbought:
            self._start_correction()

    def _start_correction(self) -> None:
        self.correction_extreme = (
            self.data.low[0] if self.direction == "long" else self.data.high[0]
        )
        self.correction_bar = len(self)
        self.state = self.STATE_WAIT_CONTINUATION
        self.log(f"Correction confirmed at {self.data.close[0]:.2f}")

    def _correction_expired(self) -> bool:
        return len(self) - self.correction_bar > self.p.max_bars_correction

    def _update_correction_extreme(self) -> None:
        if self.direction == "long":
            self.correction_extreme = min(self.correction_extreme, self.data.low[0])
        else:
            self.correction_extreme = max(self.correction_extreme, self.data.high[0])

    # -- 3. Continuation -------------------------------------------------

    def _structure_broken(self) -> bool:
        if self.direction == "long":
            return self.data.close[0] < self.structure_reference
        return self.data.close[0] > self.structure_reference

    def _bullish_engulfing(self) -> bool:
        return (
            self.data.close[-1] < self.data.open[-1]
            and self.data.close[0] > self.data.open[0]
            and self.data.close[0] >= self.data.open[-1]
            and self.data.open[0] <= self.data.close[-1]
        )

    def _bearish_engulfing(self) -> bool:
        return (
            self.data.close[-1] > self.data.open[-1]
            and self.data.close[0] < self.data.open[0]
            and self.data.close[0] <= self.data.open[-1]
            and self.data.open[0] >= self.data.close[-1]
        )

    def _check_continuation(self) -> None:
        if self.direction == "long":
            triggered = self._bullish_engulfing() or self.data.close[0] > self.data.high[-1]
            if triggered:
                self._enter("long")
        else:
            triggered = self._bearish_engulfing() or self.data.close[0] < self.data.low[-1]
            if triggered:
                self._enter("short")

    def _enter(self, direction: str) -> None:
        if not self._mtf_allows(direction):
            # The 1H/4H bias flipped against us while waiting for Correction
            # and Continuation - the setup is no longer valid.
            self._reset_setup()
            return

        entry_price = self.data.close[0]

        if direction == "long":
            stop_price = self.correction_extreme * (1 - self.p.sl_buffer)
            risk = entry_price - stop_price
        else:
            stop_price = self.correction_extreme * (1 + self.p.sl_buffer)
            risk = stop_price - entry_price

        if risk <= 0:
            self._reset_setup()
            return

        reward_ratio = max(self.p.risk_reward_ratio, self.p.min_risk_reward_ratio)
        take_profit = (
            entry_price + risk * reward_ratio
            if direction == "long"
            else entry_price - risk * reward_ratio
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
            size=size, exectype=bt.Order.Market, stopprice=stop_price, limitprice=take_profit
        )
        self.order = orders[0]
        self.stop_order = orders[1]
        self.limit_order = orders[2]
        self.position_direction = direction
        self.entry_price = entry_price
        self.initial_stop_price = stop_price
        self.take_profit_price = take_profit
        self.breakeven_triggered = False
        self.log(
            f"{direction.upper()} continuation entry={entry_price:.2f} "
            f"SL={stop_price:.2f} TP={take_profit:.2f} size={size:.4f}"
        )
        self._reset_setup()
