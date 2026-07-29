import backtrader as bt

from strategies.base import BaseStrategy


class ICCStrategy(BaseStrategy):
    """Indication-Correction-Continuation strategy, designed for gold (XAU/USD).

    1. Indication (Impuls): price breaks strongly through EMA200 (up or down)
       on above-average volume, signaling an impulsive move.
    2. Correction (Rekyl): price pulls back to EMA20, EMA50, or a Fibonacci
       retracement of the impulse leg, while RSI becomes momentarily
       overbought/oversold.
    3. Continuation (Fortsettelse): an engulfing candle, or a break of the
       prior bar's high/low in the trend direction, confirms the pullback
       is over.

    On continuation the strategy submits a bracket order: the stop loss is
    placed just beyond the correction's extreme, and the take profit is set
    at a minimum 1:2 reward-to-risk ratio.
    """

    params = (
        ("ema_fast1", 20),
        ("ema_fast2", 50),
        ("ema_trend", 200),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("volume_period", 20),
        ("volume_multiplier", 1.5),
        ("fib_level", 0.5),
        ("zone_tolerance", 0.0015),
        ("sl_buffer", 0.001),
        ("risk_reward_ratio", 2.0),
        ("min_risk_reward_ratio", 2.0),
        ("max_bars_indication", 50),
        ("max_bars_correction", 30),
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

        self.order: bt.Order | None = None
        self._reset_setup()

    def _reset_setup(self) -> None:
        self.state = self.STATE_SEARCH_INDICATION
        self.direction: str | None = None
        self.impulse_start_price: float | None = None
        self.impulse_extreme: float | None = None
        self.structure_reference: float | None = None
        self.indication_bar: int | None = None
        self.correction_extreme: float | None = None
        self.correction_bar: int | None = None

    def notify_order(self, order: bt.Order) -> None:
        super().notify_order(order)
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            if order == self.order:
                self.order = None

    def next(self) -> None:
        if self.order or self.position:
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

    # -- 1. Indication -------------------------------------------------

    def _check_indication(self) -> None:
        volume_confirmed = self.data.volume[0] > self.volume_ma[0] * self.p.volume_multiplier
        if not volume_confirmed:
            return

        if self.trend_cross[0] > 0:
            self._start_indication("long")
        elif self.trend_cross[0] < 0:
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

        bracket = self.buy_bracket if direction == "long" else self.sell_bracket
        orders = bracket(exectype=bt.Order.Market, stopprice=stop_price, limitprice=take_profit)
        self.order = orders[0]
        self.log(
            f"{direction.upper()} continuation entry={entry_price:.2f} "
            f"SL={stop_price:.2f} TP={take_profit:.2f}"
        )
        self._reset_setup()
