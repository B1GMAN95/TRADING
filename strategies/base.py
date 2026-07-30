import backtrader as bt


class BaseStrategy(bt.Strategy):
    """Common logging and order-notification behavior for all strategies."""

    #: Strategies that need 1H/4H bias feeds (see strategies/mtf_bias.py) set
    #: this to True so backtesting/engine.py and api/trading_engine.py know
    #: to add the resampled secondary data feeds.
    requires_mtf: bool = False

    def log(self, txt: str, dt=None) -> None:
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{side} EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}"
            )
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log(f"Order failed: {order.getstatusname()}")

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.log(f"Trade closed, pnl={trade.pnl:.2f}, net_pnl={trade.pnlcomm:.2f}")
