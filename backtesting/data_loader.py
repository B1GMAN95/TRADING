import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into a DataFrame indexed by datetime.

    Expects a 'date' column plus open/high/low/close/volume columns.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]
