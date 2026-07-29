import backtrader as bt
import pandas as pd

from models.schemas.strategy import BacktestRequest, BacktestResult, EquityPoint
from strategies.registry import get_strategy


def _build_equity_curve(time_returns: dict, starting_cash: float) -> list[EquityPoint]:
    equity = starting_cash
    peak = starting_cash
    curve = []
    for date, period_return in time_returns.items():
        equity *= 1 + period_return
        peak = max(peak, equity)
        drawdown_pct = (equity - peak) / peak * 100 if peak > 0 else 0.0
        curve.append(
            EquityPoint(
                date=date.strftime("%Y-%m-%d"),
                equity=round(equity, 2),
                drawdown_pct=round(drawdown_pct, 2),
            )
        )
    return curve


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
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    starting_cash = cerebro.broker.getvalue()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()

    strat = results[0]
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    drawdown = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown")
    equity_curve = _build_equity_curve(strat.analyzers.time_return.get_analysis(), starting_cash)

    won = trade_analysis.get("won", {}).get("total", 0)
    win_rate_pct = won / total_trades * 100 if total_trades else None

    gross_won = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0.0)
    gross_lost = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0))
    profit_factor = gross_won / gross_lost if gross_lost > 0 else None

    return BacktestResult(
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        starting_cash=starting_cash,
        ending_value=ending_value,
        total_return_pct=(ending_value - starting_cash) / starting_cash * 100,
        trades=total_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown,
        equity_curve=equity_curve,
    )
