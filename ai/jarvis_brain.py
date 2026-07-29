import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from models.schemas.analysis import MarketAnalysis

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a disciplined market analyst. Given technical indicators and news "
    "headlines, respond ONLY with a JSON object with keys: "
    "'bias' (one of 'bullish', 'bearish', 'neutral'), "
    "'confidence_score' (a float between 0 and 1), "
    "'trading_advice' (a short, actionable recommendation), and "
    "'rationale' (a brief explanation of the reasoning)."
)


class JarvisBrainError(RuntimeError):
    """Raised when the Yunwu API call fails or its response can't be parsed."""


class JarvisBrain:
    """Sends market context to the Yunwu LLM API and returns a structured trading analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("YUNWU_API_KEY", "")
        self._model = model or os.environ.get("YUNWU_MODEL", "claude-3-5-sonnet")
        resolved_base_url = base_url or os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai")
        self._client = client or httpx.Client(
            base_url=resolved_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JarvisBrain":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def analyze_market(
        self,
        indicators: dict[str, Any],
        headlines: list[str],
    ) -> MarketAnalysis:
        """Ask the LLM for a market bias, confidence score, and trading advice.

        `indicators` maps indicator names to their current values (e.g. {"rsi": 65}).
        `headlines` is a list of recent news headlines relevant to the market.
        """
        prompt = self._build_prompt(indicators, headlines)

        try:
            response = self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise JarvisBrainError(f"Yunwu API request failed: {exc}") from exc

        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
            data = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise JarvisBrainError(f"Unexpected Yunwu response: {payload}") from exc

        return MarketAnalysis.model_validate(data)

    @staticmethod
    def _build_prompt(indicators: dict[str, Any], headlines: list[str]) -> str:
        indicators_text = "\n".join(f"- {key}: {value}" for key, value in indicators.items())
        headlines_text = "\n".join(f"- {headline}" for headline in headlines) or "None"
        return f"Technical indicators:\n{indicators_text}\n\nNews headlines:\n{headlines_text}"
