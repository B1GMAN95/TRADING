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

    def analyze_market(self, indicators: dict, headlines: list[str], **_kwargs) -> MarketAnalysis:
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


def test_index_page_lists_all_supported_assets() -> None:
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "XAUUSD" in response.text
    assert "NASDAQ" in response.text
    assert "SP500" in response.text


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


def test_backtest_runs_dynamically_against_nasdaq_sample_data() -> None:
    response = client.post(
        "/backtests",
        json={
            "strategy_name": "icc_gold",
            "symbol": "NASDAQ",
            "start_date": "2024-01-01",
            "end_date": "2024-12-30",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "NASDAQ"
    assert isinstance(body["ending_value"], float)


def test_backtest_runs_dynamically_against_sp500_sample_data() -> None:
    response = client.post(
        "/backtests",
        json={
            "strategy_name": "smc_gold",
            "symbol": "SP500",
            "start_date": "2024-01-01",
            "end_date": "2024-12-30",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "SP500"
    assert isinstance(body["ending_value"], float)


def test_backtest_returns_404_for_unknown_asset() -> None:
    response = client.post(
        "/backtests",
        json={
            "strategy_name": "icc_gold",
            "symbol": "DOESNOTEXIST",
            "start_date": "2024-01-01",
            "end_date": "2024-12-30",
        },
    )
    assert response.status_code == 404


def test_jarvis_status_reports_agreement(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.BULLISH
    )
    monkeypatch.setattr(
        dashboard_module, "fetch_headlines", lambda query, **kwargs: ["Gold rallies on demand"]
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
    monkeypatch.setattr(dashboard_module, "fetch_headlines", lambda query, **kwargs: [])
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(bias="bearish"))

    response = client.get("/dashboard/status/jarvis")

    assert response.status_code == 200
    body = response.json()
    assert body["agrees"] is False


def test_jarvis_status_degrades_gracefully_when_brain_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.NEUTRAL
    )
    monkeypatch.setattr(dashboard_module, "fetch_headlines", lambda query, **kwargs: [])
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(raise_error=True))

    response = client.get("/dashboard/status/jarvis")

    assert response.status_code == 200
    body = response.json()
    assert body["technical_signal"] == "neutral"
    assert "ai_error" in body


def test_jarvis_status_defaults_to_gold_symbol(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.NEUTRAL
    )
    monkeypatch.setattr(dashboard_module, "fetch_headlines", lambda query, **kwargs: [])
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(bias="bullish"))

    response = client.get("/dashboard/status/jarvis")

    assert response.status_code == 200
    assert response.json()["symbol"] == "XAUUSD"


def test_jarvis_status_runs_against_selected_asset(monkeypatch) -> None:
    seen_queries = []

    def _fake_fetch_headlines(query: str, **kwargs) -> list[str]:
        seen_queries.append(query)
        return []

    monkeypatch.setattr(
        dashboard_module, "get_technical_signal", lambda price_data: MarketBias.BULLISH
    )
    monkeypatch.setattr(dashboard_module, "fetch_headlines", _fake_fetch_headlines)
    monkeypatch.setattr(dashboard_module, "JarvisBrain", lambda: _StubBrain(bias="bullish"))

    response = client.get("/dashboard/status/jarvis", params={"symbol": "NASDAQ"})

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "NASDAQ"
    assert body["technical_signal"] == "bullish"
    assert "timeframes" in body
    assert set(body["timeframes"]) == {"4h", "1h", "15m", "5m"}
    assert seen_queries == [dashboard_module.ASSET_REGISTRY["NASDAQ"]["news_query"]]


def test_jarvis_status_reports_error_for_unknown_asset() -> None:
    response = client.get("/dashboard/status/jarvis", params={"symbol": "DOESNOTEXIST"})

    assert response.status_code == 200
    assert "error" in response.json()
