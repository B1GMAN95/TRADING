import backtrader as bt

from strategies.base import BaseStrategy


class SmaCrossoverStrategy(BaseStrategy):
    """Goes long when the fast SMA crosses above the slow SMA, and closes on cross-under."""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self) -> None:
        sma_fast = bt.ind.SMA(period=self.p.fast_period)
        sma_slow = bt.ind.SMA(period=self.p.slow_period)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)

    def next(self) -> None:
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()
