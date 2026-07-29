# TradingBot

Produksjonsklar Python-prosjektstruktur for en tradingbot: REST API, strategier,
backtesting og et dashboard.

## Prosjektstruktur

```
.
├── api/                # FastAPI app, endepunkter (ordre, backtests, health)
│   ├── main.py
│   └── routes/
├── strategies/         # Backtrader-strategier og et register over dem
│   ├── base.py
│   ├── registry.py
│   └── examples/
├── backtesting/         # Backtest-motor og datainnlasting
│   ├── engine.py
│   ├── data_loader.py
│   └── results/
├── models/              # Pydantic-skjemaer (ordre, strategi/backtest)
│   └── schemas/
├── dashboard/            # Enkel FastAPI-drevet dashboard-app
│   ├── app.py
│   ├── static/
│   └── templates/
├── tests/
├── config.py            # Sentrale innstillinger via pydantic-settings
├── requirements.txt
└── requirements-dev.txt
```

## Kom i gang

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

uvicorn api.main:app --reload
```

API-en kjører på http://localhost:8000, dashboard på http://localhost:8000/dashboard.

## Test

```bash
pytest
```

## Legg til en ny strategi

1. Opprett en fil under `strategies/examples/` som arver fra `strategies.base.BaseStrategy`.
2. Registrer den i `strategies/registry.py` sitt `STRATEGY_REGISTRY`.
3. Kjør en backtest via `POST /backtests` med `strategy_name` satt til nøkkelen din.
