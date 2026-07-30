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
    "'trading_advice' (a short, actionable recommendation), "
    "'rationale' (a brief explanation of the reasoning), and, only when a "
    "multi-timeframe matrix is supplied, 'mtf_alpha_score' (a float between "
    "0 and 1 measuring how strongly the 4H/1H/15M/5M timeframes confirm "
    "each other)."
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
        timeframes: dict[str, dict[str, Any]] | None = None,
        reference_alpha_score: float | None = None,
    ) -> MarketAnalysis:
        """Ask the LLM for a market bias, confidence score, and trading advice.

        `indicators` maps indicator names to their current values (e.g. {"rsi": 65}).
        `headlines` is a list of recent news headlines relevant to the market.

        `timeframes`, when given, is a matrix of per-timeframe technical data
        (see api.trading_engine.get_mtf_snapshot) - keyed "4h"/"1h"/"15m"/"5m"
        - that the model is asked to weigh together into a single
        'mtf_alpha_score'. `reference_alpha_score` is a deterministic
        confluence score computed from that same matrix
        (api.trading_engine.compute_mtf_alpha_score); it's passed to the model
        as a sanity-check reference (LLMs aren't reliable at precise
        arithmetic) and used to fill in `mtf_alpha_score` if the model's
        response omits it.
        """
        prompt = self._build_prompt(indicators, headlines, timeframes, reference_alpha_score)

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

        analysis = MarketAnalysis.model_validate(data)
        if analysis.mtf_alpha_score is None and reference_alpha_score is not None:
            analysis = analysis.model_copy(update={"mtf_alpha_score": reference_alpha_score})
        return analysis

    @staticmethod
    def _build_prompt(
        indicators: dict[str, Any],
        headlines: list[str],
        timeframes: dict[str, dict[str, Any]] | None = None,
        reference_alpha_score: float | None = None,
    ) -> str:
        indicators_text = "\n".join(f"- {key}: {value}" for key, value in indicators.items())
        headlines_text = "\n".join(f"- {headline}" for headline in headlines) or "None"
        sections = [
            f"Technical indicators:\n{indicators_text}",
            f"News headlines:\n{headlines_text}",
        ]

        if timeframes:
            tf_lines = []
            for label in ("4h", "1h", "15m", "5m"):
                tier = timeframes.get(label)
                if not tier:
                    continue
                details = ", ".join(f"{key}={value}" for key, value in tier.items())
                tf_lines.append(f"- {label.upper()}: {details}")
            sections.append(
                "Multi-timeframe matrix (4H/1H = long-term bias, 15M/5M = "
                "execution timeframes):\n"
                + "\n".join(tf_lines)
                + "\n\nWeigh the confluence across all four timeframes and "
                "include an 'mtf_alpha_score' key in your JSON response."
            )
            if reference_alpha_score is not None:
                sections.append(
                    "Reference confluence score computed from the matrix "
                    f"above: {reference_alpha_score:.2f}. Use it as a sanity "
                    "check for your own 'mtf_alpha_score'."
                )

        return "\n\n".join(sections)
