from fastapi import FastAPI

from api.routes import backtests, health, orders
from config import get_settings
from dashboard.app import dashboard_app

settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.include_router(health.router)
app.include_router(orders.router)
app.include_router(backtests.router)
app.mount("/dashboard", dashboard_app)


@app.get("/")
def root() -> dict:
    return {"name": settings.app_name, "environment": settings.environment}
