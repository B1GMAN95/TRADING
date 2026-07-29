from fastapi.testclient import TestClient

import dashboard.app as dashboard_module
from ai.jarvis_brain import JarvisBrainError
from api.main import app
from models.schemas.analysis import MarketAnalysis, MarketBias

client = TestClient(app)


class _StubBrain:
    def __init__(self, bias: str = "bullish", raise_error: bool = False) -> None:
        self._bias = bias
        self._raise_error = raise_error

    def __enter__(self) -> "_StubBrain":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def analyze_market(self, indicators: dict, headlines: list[str]) -> MarketAnalysis:
        if self._raise_error:
            raise JarvisBrainError("Yunwu API unreachable")
        return MarketAnalysis(
            bias=MarketBias(self._bias),
            confidence_score=0.8,
            trading_advice="Consider a long position.",
            rationale="Momentum and news both point the same way.",
        )


def test_index_page_lists_strategies_including_icc_gold() -> None:
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "icc_gold" in response.text
    assert "sma_crossover" in response.text


def test_backtest_runs_icc_gold_against_bundled_sample_data() -> None:
    response = client.post(
        "/backtests",
        json={
            "strategy_name": "icc_gold",
            "symbol": "XAUUSD",
            "start_date": "2024-01-02",
            "end_date": "2025-02-24",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["strategy_name"] == "icc_gold"
    assert isinstance(body["ending_value"], float)


def test_jarvis_status_reports_agreement(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.BULLISH
    )
    monkeypatch.setattr(
        dashboard_module, "fetch_gold_headlines", lambda: ["Gold rallies on demand"]
    )
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(bias="bullish"))

    response = client.get("/dashboard/status/jarvis")

    assert response.status_code == 200
    body = response.json()
    assert body["technical_signal"] == "bullish"
    assert body["ai_bias"] == "bullish"
    assert body["agrees"] is True


def test_jarvis_status_reports_disagreement(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.BULLISH
    )
    monkeypatch.setattr(dashboard_module, "fetch_gold_headlines", lambda: [])
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(bias="bearish"))

    response = client.get("/dashboard/status/jarvis")

    assert response.status_code == 200
    body = response.json()
    assert body["agrees"] is False


def test_jarvis_status_degrades_gracefully_when_brain_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.NEUTRAL
    )
    monkeypatch.setattr(dashboard_module, "fetch_gold_headlines", lambda: [])
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(raise_error=True))

    response = client.get("/dashboard/status/jarvis")

    assert response.status_code == 200
    body = response.json()
    assert body["technical_signal"] == "neutral"
    assert "ai_error" in body
