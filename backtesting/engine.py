import backtrader as bt
import pandas as pd

from models.schemas.strategy import BacktestRequest, BacktestResult
from strategies.registry import get_strategy


def run_backtest(request: BacktestRequest, price_data: pd.DataFrame) -> BacktestResult:
    """Run a backtest over OHLCV data indexed by datetime.

    price_data must contain columns: open, high, low, close, volume.
    """
    cerebro = bt.Cerebro()
    strategy_cls = get_strategy(request.strategy_name)
    cerebro.addstrategy(strategy_cls, **request.parameters)

    data_feed = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(request.cash)
    cerebro.broker.setcommission(commission=request.commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    starting_cash = cerebro.broker.getvalue()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()

    strat = results[0]
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    drawdown = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown")

    return BacktestResult(
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        starting_cash=starting_cash,
        ending_value=ending_value,
        total_return_pct=(ending_value - starting_cash) / starting_cash * 100,
        trades=total_trades,
        sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown,
    )
