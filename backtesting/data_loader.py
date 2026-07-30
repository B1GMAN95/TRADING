import backtrader as bt
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into a DataFrame indexed by datetime.

    Expects a 'date' column plus open/high/low/close/volume columns.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _is_intraday(price_data: pd.DataFrame) -> bool:
    """True if price_data's bars are frequent enough to resample up to 1H/4H.

    Resampling only aggregates finer bars into coarser ones - daily (or
    slower) bars can't be turned into 1H/4H bars, since that would require
    inventing detail that isn't there.
    """
    if len(price_data.index) < 2:
        return False
    median_gap = price_data.index.to_series().diff().median()
    return median_gap < pd.Timedelta(hours=1)


def add_price_feeds(cerebro: bt.Cerebro, strategy_cls: type, price_data: pd.DataFrame) -> None:
    """Add the execution-timeframe feed to `cerebro`, plus resampled 1H/4H
    bias feeds (strategies/mtf_bias.py) when `strategy_cls.requires_mtf` and
    `price_data` is actually fine-grained enough to support it.

    `price_data` is assumed to already be at the execution timeframe (15M);
    1H and 4H are derived from it via Backtrader's own resampling, which
    only aggregates finer bars into coarser ones - it cannot invent a truly
    finer 5M feed from 15M data. A genuine 5M feed would need price_data
    itself to be re-sourced at 5-minute granularity.

    If price_data is too coarse (e.g. daily bars) for 1H/4H resampling, the
    MTF feeds are skipped entirely rather than added in a broken state -
    strategies with requires_mtf treat missing MTF feeds as "no bias filter"
    (see strategies/mtf_bias.py), so this is a graceful, not a silent-break,
    fallback.
    """
    data_feed = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data_feed)

    if getattr(strategy_cls, "requires_mtf", False) and _is_intraday(price_data):
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=60)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=240)
