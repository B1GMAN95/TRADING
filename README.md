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

## Publisere dashboardet (Render)

Repoet har en `render.yaml` som gjør oppsett på [Render](https://render.com) enkelt:

1. Opprett en gratis konto på render.com og koble til GitHub-kontoen din.
2. Velg **New +** → **Blueprint**, og pek på dette repoet (`B1GMAN95/TRADING`).
   Render leser `render.yaml` automatisk og setter opp bygg-/startkommandoer.
3. (Valgfritt) Sett `YUNWU_API_KEY` og `NEWS_API_KEY` under Environment i Render
   for at Jarvis-status-panelet skal vise ekte AI-analyse og nyheter. Uten
   disse viser dashboardet fortsatt teknisk signal, bare med en tydelig
   "utilgjengelig"-melding for AI-delen.
4. Etter deploy får du en offentlig URL i stil med `https://tradingbot-xxxx.onrender.com`.
   Dashboardet ligger på `/dashboard/`.
