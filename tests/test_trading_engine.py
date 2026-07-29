import json

import httpx
import numpy as np
import pandas as pd

from ai.jarvis_brain import JarvisBrain
from api.trading_engine import (
    evaluate_trade,
    fetch_gold_headlines,
    get_latest_technical_values,
    get_technical_signal,
)
from models.schemas.analysis import MarketBias

_STRATEGY_PARAMS = dict(
    ema_fast1=3,
    ema_fast2=4,
    ema_trend=5,
    rsi_period=4,
    volume_period=4,
    volume_multiplier=1.3,
    zone_tolerance=0.02,
    max_bars_indication=50,
    max_bars_correction=50,
)

# Same hand-crafted setup as tests/test_icc_strategy.py, trimmed to end exactly
# on the Continuation bar so the signal is "live" (fired on the latest candle).
_BULLISH_ROWS = [
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.0, high=100.1, low=99.7, close=99.9, volume=1000),
    dict(open=100, high=103.2, low=99.9, close=103, volume=1600),  # Indication
    dict(open=103, high=106.3, low=102.8, close=106, volume=1000),
    dict(open=106, high=109.4, low=105.5, close=109, volume=1000),
    dict(open=109, high=109.5, low=105.8, close=106.5, volume=1000),
    dict(open=106.5, high=107, low=104.0, close=104.5, volume=1000),
    dict(open=104.5, high=105, low=101.8, close=102.0, volume=1000),
    dict(open=102.0, high=102.5, low=100.0, close=100.5, volume=1000),  # Correction
    dict(open=100.3, high=104.5, low=100.2, close=104.2, volume=1200),  # Continuation (latest bar)
]

_NEUTRAL_ROWS = [
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
    dict(open=100.0, high=100.2, low=99.8, close=100.0, volume=1000),
    dict(open=100.01, high=100.21, low=99.81, close=100.01, volume=1000),
    dict(open=99.99, high=100.19, low=99.79, close=99.99, volume=1000),
]


def _to_df(rows: list[dict]) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=dates)


def _brain_with_bias(bias: str) -> JarvisBrain:
    content = json.dumps(
        {
            "bias": bias,
            "confidence_score": 0.75,
            "trading_advice": "Follow the confirmed setup.",
            "rationale": "Technical and news context align.",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(base_url="https://yunwu.ai", transport=httpx.MockTransport(handler))
    return JarvisBrain(api_key="test-key", model="test-model", client=client)


def _news_client_with_titles(titles: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"articles": [{"title": t} for t in titles]})

    return httpx.Client(base_url="https://newsapi.org", transport=httpx.MockTransport(handler))


def test_get_technical_signal_bullish_on_latest_bar() -> None:
    signal = get_technical_signal(_to_df(_BULLISH_ROWS), **_STRATEGY_PARAMS)
    assert signal == MarketBias.BULLISH


def test_get_technical_signal_neutral_without_trigger() -> None:
    signal = get_technical_signal(_to_df(_NEUTRAL_ROWS), **_STRATEGY_PARAMS)
    assert signal == MarketBias.NEUTRAL


def test_get_latest_technical_values() -> None:
    dates = pd.date_range("2023-01-01", periods=250, freq="D")
    rng = np.random.default_rng(3)
    prices = 1900 + np.cumsum(rng.normal(0, 5, size=250))
    df = pd.DataFrame({"close": prices}, index=dates)

    values = get_latest_technical_values(df)

    assert 0 <= values["rsi"] <= 100
    assert set(values) == {"rsi", "ema_20", "ema_50", "ema_200"}


def test_fetch_gold_headlines_parses_titles() -> None:
    client = _news_client_with_titles(["Gold hits record high", "Fed signals rate pause"])
    headlines = fetch_gold_headlines(client=client)
    assert headlines == ["Gold hits record high", "Fed signals rate pause"]


def test_evaluate_trade_approves_when_technical_and_ai_agree() -> None:
    decision = evaluate_trade(
        _to_df(_BULLISH_ROWS),
        brain=_brain_with_bias("bullish"),
        news_client=_news_client_with_titles(["Gold rallies on safe-haven demand"]),
        strategy_params=_STRATEGY_PARAMS,
    )

    assert decision.approved is True
    assert decision.technical_signal == MarketBias.BULLISH
    assert decision.ai_analysis.bias == MarketBias.BULLISH


def test_evaluate_trade_rejects_when_technical_and_ai_disagree() -> None:
    decision = evaluate_trade(
        _to_df(_BULLISH_ROWS),
        brain=_brain_with_bias("bearish"),
        news_client=_news_client_with_titles(["Gold slides on stronger dollar"]),
        strategy_params=_STRATEGY_PARAMS,
    )

    assert decision.approved is False
    assert decision.technical_signal == MarketBias.BULLISH
    assert decision.ai_analysis.bias == MarketBias.BEARISH


def test_evaluate_trade_skips_ai_when_no_technical_signal() -> None:
    decision = evaluate_trade(_to_df(_NEUTRAL_ROWS), strategy_params=_STRATEGY_PARAMS)

    assert decision.approved is False
    assert decision.technical_signal == MarketBias.NEUTRAL
    assert decision.ai_analysis is None
