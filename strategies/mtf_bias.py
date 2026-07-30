"""Multi-timeframe long-term bias gate, shared by icc_gold and smc_gold.

A trade on the execution timeframe (15M, and in principle 5M once a native
5-minute feed is available - see the note in strategies/icc_strategy.py) is
only allowed when its direction matches both:
  - the 1H trend, defined as price vs. a 200-period EMA on the 1H feed, and
  - the 4H market structure, defined the same way MSS is detected in
    strategies/smc_strategy.py: has price most recently broken above the
    prior swing high (bullish) or below the prior swing low (bearish).
"""

import backtrader as bt


def mtf_bias_allows(direction: str, trend_1h: str, structure_4h: str) -> bool:
    """True if `direction` ("long"/"short") aligns with both the 1H trend and
    the 4H market structure.
    """
    wanted = "bullish" if direction == "long" else "bearish"
    return trend_1h == wanted and structure_4h == wanted


class MTFBias:
    """Tracks the 1H trend and 4H market structure for a strategy that has
    been given secondary data feeds for those timeframes (typically via
    `cerebro.resampledata()` from the execution-timeframe feed).

    Expects `strategy.datas[1]` to be the 1H feed and `strategy.datas[2]`
    the 4H feed.
    """

    def __init__(
        self,
        strategy: bt.Strategy,
        ema_period_1h: int = 200,
        swing_lookback_4h: int = 20,
    ) -> None:
        self.data_1h = strategy.datas[1]
        self.data_4h = strategy.datas[2]

        self.ema_1h = bt.ind.EMA(self.data_1h.close, period=ema_period_1h)
        self.swing_high_4h = bt.ind.Highest(self.data_4h.high, period=swing_lookback_4h)
        self.swing_low_4h = bt.ind.Lowest(self.data_4h.low, period=swing_lookback_4h)

        self._structure_4h = "neutral"

    def update(self) -> None:
        """Refresh the 4H structure bias. Call once per bar, from next()."""
        if len(self.data_4h) < 2:
            return

        prior_high = self.swing_high_4h[-1]
        prior_low = self.swing_low_4h[-1]
        if self.data_4h.close[0] > prior_high:
            self._structure_4h = "bullish"
        elif self.data_4h.close[0] < prior_low:
            self._structure_4h = "bearish"
        # else: no fresh break this 4H bar, keep the last known structure

    @property
    def trend_1h(self) -> str:
        if len(self.data_1h) == 0:
            return "neutral"
        if self.data_1h.close[0] > self.ema_1h[0]:
            return "bullish"
        if self.data_1h.close[0] < self.ema_1h[0]:
            return "bearish"
        return "neutral"

    @property
    def structure_4h(self) -> str:
        return self._structure_4h

    def allows(self, direction: str) -> bool:
        return mtf_bias_allows(direction, self.trend_1h, self.structure_4h)
