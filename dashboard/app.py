from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai.jarvis_brain import JarvisBrain, JarvisBrainError
from api.trading_engine import (
    TradingEngineError,
    fetch_gold_headlines,
    get_latest_technical_values,
    get_technical_signal,
)
from backtesting.data_loader import load_csv
from strategies.registry import STRATEGY_REGISTRY

BASE_DIR = Path(__file__).resolve().parent
GOLD_SYMBOL = "XAUUSD"
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
            "symbol": GOLD_SYMBOL,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
        },
    )


@dashboard_app.get("/status/jarvis")
def jarvis_status() -> dict:
    """Live status of what the ICC strategy and JarvisBrain think of gold right now."""
    try:
        price_data = load_csv(f"data/{GOLD_SYMBOL}.csv").tail(LIVE_STATUS_LOOKBACK_BARS)
    except FileNotFoundError:
        return {"error": f"No sample price data found for {GOLD_SYMBOL}."}

    technical_signal = get_technical_signal(price_data)
    indicators = get_latest_technical_values(price_data)

    try:
        headlines = fetch_gold_headlines()
    except TradingEngineError:
        headlines = []

    try:
        with JarvisBrain() as brain:
            ai_analysis = brain.analyze_market(indicators, headlines)
    except JarvisBrainError as exc:
        return {
            "technical_signal": technical_signal.value,
            "indicators": indicators,
            "ai_error": f"JarvisBrain is unavailable: {exc}",
        }

    agrees = technical_signal.value != "neutral" and ai_analysis.bias == technical_signal

    return {
        "technical_signal": technical_signal.value,
        "indicators": indicators,
        "ai_bias": ai_analysis.bias.value,
        "ai_confidence_score": ai_analysis.confidence_score,
        "ai_trading_advice": ai_analysis.trading_advice,
        "ai_rationale": ai_analysis.rationale,
        "agrees": agrees,
    }
