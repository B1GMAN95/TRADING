import json

import httpx
import pytest

from ai.jarvis_brain import JarvisBrain, JarvisBrainError
from models.schemas.analysis import MarketBias


def _client_with_content(content: str, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(base_url="https://yunwu.ai", transport=httpx.MockTransport(handler))


def test_analyze_market_returns_structured_result() -> None:
    content = json.dumps(
        {
            "bias": "bullish",
            "confidence_score": 0.82,
            "trading_advice": "Consider a long position with a tight stop.",
            "rationale": "Momentum indicators align with positive sentiment.",
        }
    )
    brain = JarvisBrain(
        api_key="test-key", model="test-model", client=_client_with_content(content)
    )

    result = brain.analyze_market(
        indicators={"rsi": 65, "macd": 1.2},
        headlines=["Company X beats earnings expectations"],
    )

    assert result.bias == MarketBias.BULLISH
    assert result.confidence_score == 0.82
    assert "long position" in result.trading_advice


def test_analyze_market_raises_on_malformed_response() -> None:
    brain = JarvisBrain(
        api_key="test-key", model="test-model", client=_client_with_content("not json")
    )

    with pytest.raises(JarvisBrainError):
        brain.analyze_market(indicators={"rsi": 50}, headlines=[])


def test_analyze_market_raises_on_http_error() -> None:
    brain = JarvisBrain(
        api_key="test-key",
        model="test-model",
        client=_client_with_content("{}", status_code=500),
    )

    with pytest.raises(JarvisBrainError):
        brain.analyze_market(indicators={"rsi": 50}, headlines=[])
