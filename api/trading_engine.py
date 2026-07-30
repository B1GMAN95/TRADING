from dataclasses import dataclass
from typing import Any

import backtrader as bt
import httpx
import pandas as pd

from ai.jarvis_brain import JarvisBrain, JarvisBrainError
from config import get_settings
from models.schemas.analysis import MarketAnalysis, MarketBias
from strategies.icc_strategy import ICCStrategy


class TradingEngineError(RuntimeError):
    """Raised when the trading engine can't reach a decision."""


@dataclass
class TradeDecision:
    """The outcome of gating a candidate trade behind both signal sources."""

    approved: bool
    technical_signal: MarketBias
    ai_analysis: MarketAnalysis | None
    reason: str


class _SignalCapturingICC(ICCStrategy):
    """ICCStrategy subclass that records a continuation trigger on the latest bar.

    A live technical signal only counts if the Continuation trigger fires on
    the most recent candle, not at some point earlier in the supplied history.
    """

    def __init__(self) -> None:
        super().__init__()
        self.technical_signal: str | None = None

    def _enter(self, direction: str) -> None:
        if len(self) == self.data.buflen():
            self.technical_signal = direction
        super()._enter(direction)


def get_technical_signal(price_data: pd.DataFrame, **strategy_params: Any) -> MarketBias:
    """Run the ICC strategy over recent OHLCV data and report the latest signal.

    Returns BULLISH/BEARISH if a Continuation trigger fires on the most recent
    bar of `price_data`, otherwise NEUTRAL.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(_SignalCapturingICC, **strategy_params)
    cerebro.adddata(bt.feeds.PandasData(dataname=price_data))
    cerebro.broker.setcash(1_000_000.0)
    results = cerebro.run()
    signal = results[0].technical_signal

    if signal == "long":
        return MarketBias.BULLISH
    if signal == "short":
        return MarketBias.BEARISH
    return MarketBias.NEUTRAL


_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _tier_bias(close: pd.Series, ema_period: int = 200) -> str:
    if len(close) < 2:
        return "neutral"
    ema = close.ewm(span=min(ema_period, len(close)), adjust=False).mean()
    if close.iloc[-1] > ema.iloc[-1]:
        return "bullish"
    if close.iloc[-1] < ema.iloc[-1]:
        return "bearish"
    return "neutral"


def get_mtf_snapshot(price_data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Lightweight (pandas-only) technical snapshot per timeframe tier, for
    JarvisBrain's Multi-Timeframe Alpha Score and the dashboard's per-
    timeframe status circles.

    `price_data` is assumed to already be at the finest (15M) granularity,
    the same convention as backtesting/data_loader.add_price_feeds(): 4H/1H
    are derived by resampling it, and "5M" is aliased to the native 15M data
    (a genuine 5M feed would need price_data itself re-sourced at 5-minute
    granularity - see that module's docstring).
    """
    tier_frames = {
        "4h": price_data.resample("4h").agg(_OHLCV_AGG).dropna(),
        "1h": price_data.resample("1h").agg(_OHLCV_AGG).dropna(),
        "15m": price_data,
        "5m": price_data,
    }

    snapshot: dict[str, dict[str, Any]] = {}
    for label, tier_df in tier_frames.items():
        indicators = get_latest_technical_values(tier_df) if len(tier_df) >= 200 else {}
        snapshot[label] = {**indicators, "bias": _tier_bias(tier_df["close"])}
    return snapshot


def compute_mtf_alpha_score(timeframe_biases: dict[str, str]) -> float:
    """Fraction of timeframes agreeing with the dominant bias (0.0-1.0).

    A deterministic complement to JarvisBrain's own qualitative score - LLMs
    aren't reliable at precise arithmetic, so this is computed here and
    passed into the prompt as a reference rather than left for the model to
    compute unassisted.
    """
    biases = list(timeframe_biases.values())
    if not biases:
        return 0.0
    dominant_count = max(biases.count(bias) for bias in set(biases))
    return round(dominant_count / len(biases), 2)


def get_latest_technical_values(
    price_data: pd.DataFrame,
    rsi_period: int = 14,
    ema_periods: tuple[int, ...] = (20, 50, 200),
) -> dict[str, float]:
    """Compute the latest RSI and EMA values from OHLCV market data."""
    close = price_data["close"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    indicators = {"rsi": round(float(rsi.iloc[-1]), 2)}
    for period in ema_periods:
        ema = close.ewm(span=period, adjust=False).mean()
        indicators[f"ema_{period}"] = round(float(ema.iloc[-1]), 2)
    return indicators


def fetch_headlines(query: str, limit: int = 5, client: httpx.Client | None = None) -> list[str]:
    """Fetch the latest news headlines matching `query` via a news API."""
    settings = get_settings()
    owns_client = client is None
    http_client = client or httpx.Client(
        base_url=settings.news_api_base_url.rstrip("/"), timeout=15.0
    )
    try:
        response = http_client.get(
            "/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": limit,
                "apiKey": settings.news_api_key,
            },
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [article["title"] for article in articles if article.get("title")]
    except httpx.HTTPError as exc:
        raise TradingEngineError(f"Failed to fetch headlines for '{query}': {exc}") from exc
    finally:
        if owns_client:
            http_client.close()


def fetch_gold_headlines(limit: int = 5, client: httpx.Client | None = None) -> list[str]:
    """Fetch the latest news headlines about gold (XAU/USD) via a news API."""
    return fetch_headlines("gold OR XAU/USD OR XAUUSD", limit=limit, client=client)


def evaluate_trade(
    price_data: pd.DataFrame,
    brain: JarvisBrain | None = None,
    news_client: httpx.Client | None = None,
    strategy_params: dict[str, Any] | None = None,
    headline_limit: int = 5,
) -> TradeDecision:
    """Approve a trade only when the ICC strategy and JarvisBrain agree on direction.

    1. Reads the latest RSI/EMA values from `price_data`.
    2. Fetches fresh gold (XAU/USD) news headlines.
    3. Asks JarvisBrain.analyze_market() for an AI bias.
    4. Approves the trade only if icc_strategy.py's technical signal and the
       AI bias point the same direction (e.g. both bullish).
    """
    technical_signal = get_technical_signal(price_data, **(strategy_params or {}))

    if technical_signal == MarketBias.NEUTRAL:
        return TradeDecision(
            approved=False,
            technical_signal=technical_signal,
            ai_analysis=None,
            reason="No ICC continuation signal on the latest bar.",
        )

    indicators = get_latest_technical_values(price_data)
    headlines = fetch_gold_headlines(limit=headline_limit, client=news_client)

    owns_brain = brain is None
    active_brain = brain or JarvisBrain()
    try:
        ai_analysis = active_brain.analyze_market(indicators, headlines)
    except JarvisBrainError as exc:
        raise TradingEngineError(f"JarvisBrain analysis failed: {exc}") from exc
    finally:
        if owns_brain:
            active_brain.close()

    agrees = ai_analysis.bias == technical_signal
    reason = (
        f"Technical ({technical_signal.value}) and AI ({ai_analysis.bias.value}) agree."
        if agrees
        else f"Technical ({technical_signal.value}) and AI ({ai_analysis.bias.value}) disagree."
    )
    return TradeDecision(
        approved=agrees,
        technical_signal=technical_signal,
        ai_analysis=ai_analysis,
        reason=reason,
    )
