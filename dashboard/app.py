from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent

dashboard_app = FastAPI(title="TradingBot Dashboard")
dashboard_app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@dashboard_app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})
