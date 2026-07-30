from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai.jarvis_brain import JarvisBrain, JarvisBrainError
from api.trading_engine import (
    TradingEngineError,
    compute_mtf_alpha_score,
    fetch_headlines,
    get_latest_technical_values,
    get_mtf_snapshot,
    get_technical_signal,
)
from backtesting.data_loader import load_csv
from strategies.registry import STRATEGY_REGISTRY

BASE_DIR = Path(__file__).resolve().parent
GOLD_SYMBOL = "XAUUSD"

# Multi-asset support: symbol -> display label + the news search query used
# to fetch headlines for that asset's Jarvis analysis. Every entry must have
# a matching data/<symbol>.csv sample file.
ASSET_REGISTRY: dict[str, dict[str, str]] = {
    "XAUUSD": {"label": "Gold (XAU/USD)", "news_query": "gold OR XAU/USD OR XAUUSD"},
    "NASDAQ": {"label": "Nasdaq 100", "news_query": "Nasdaq 100 OR NDX OR Nasdaq composite"},
    "SP500": {"label": "S&P 500", "news_query": "S&P 500 OR SPX OR S&P500"},
}

# Live status only needs enough recent history for EMA50/RSI14/ATR14 to be
# fully warmed up - replaying the entire (now intraday, tens of thousands of
# bars) sample file on every request would make the endpoint sluggish.
LIVE_STATUS_LOOKBACK_BARS = 3000

dashboard_app = FastAPI(title="TradingBot Dashboard")
dashboard_app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@dashboard_app.get("/")
def index(request: Request):
    try:
        price_data = load_csv(f"data/{GOLD_SYMBOL}.csv")
        data_start_date = price_data.index.min().strftime("%Y-%m-%d")
        data_end_date = price_data.index.max().strftime("%Y-%m-%d")
    except FileNotFoundError:
        data_start_date = ""
        data_end_date = ""

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "strategies": list(STRATEGY_REGISTRY),
            "assets": [
                {"symbol": symbol, "label": info["label"]}
                for symbol, info in ASSET_REGISTRY.items()
            ],
            "symbol": GOLD_SYMBOL,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
        },
    )


@dashboard_app.get("/status/jarvis")
def jarvis_status(symbol: str = GOLD_SYMBOL) -> dict:
    """Live status of what icc_gold and JarvisBrain think of `symbol` right now."""
    asset = ASSET_REGISTRY.get(symbol, {"label": symbol, "news_query": symbol})

    try:
        price_data = load_csv(f"data/{symbol}.csv").tail(LIVE_STATUS_LOOKBACK_BARS)
    except FileNotFoundError:
        return {"error": f"No sample price data found for {symbol}."}

    technical_signal = get_technical_signal(price_data)
    indicators = get_latest_technical_values(price_data)

    mtf_snapshot = get_mtf_snapshot(price_data)
    timeframes = {label: tier["bias"] for label, tier in mtf_snapshot.items()}
    mtf_alpha_score = compute_mtf_alpha_score(timeframes)

    try:
        headlines = fetch_headlines(asset["news_query"])
    except TradingEngineError:
        headlines = []

    try:
        with JarvisBrain() as brain:
            ai_analysis = brain.analyze_market(
                indicators,
                headlines,
                timeframes=mtf_snapshot,
                reference_alpha_score=mtf_alpha_score,
            )
    except JarvisBrainError as exc:
        return {
            "symbol": symbol,
            "technical_signal": technical_signal.value,
            "indicators": indicators,
            "timeframes": timeframes,
            "mtf_alpha_score": mtf_alpha_score,
            "ai_error": f"JarvisBrain is unavailable: {exc}",
        }

    agrees = technical_signal.value != "neutral" and ai_analysis.bias == technical_signal

    return {
        "symbol": symbol,
        "technical_signal": technical_signal.value,
        "indicators": indicators,
        "timeframes": timeframes,
        "mtf_alpha_score": ai_analysis.mtf_alpha_score
        if ai_analysis.mtf_alpha_score is not None
        else mtf_alpha_score,
        "ai_bias": ai_analysis.bias.value,
        "ai_confidence_score": ai_analysis.confidence_score,
        "ai_trading_advice": ai_analysis.trading_advice,
        "ai_rationale": ai_analysis.rationale,
        "agrees": agrees,
    }
