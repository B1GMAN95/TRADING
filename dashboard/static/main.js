const GOLD_SYMBOL = "XAUUSD";
let equityChart = null;

fetch("/health")
    .then((res) => res.json())
    .then((data) => {
        document.getElementById("status").textContent = data.status;
    })
    .catch(() => {
        document.getElementById("status").textContent = "unreachable";
    });

function setKpiCard(cardId, valueId, text, tone) {
    const card = document.getElementById(cardId);
    const value = document.getElementById(valueId);
    value.textContent = text;
    card.classList.remove("kpi-positive", "kpi-negative");
    if (tone === "positive") {
        card.classList.add("kpi-positive");
    } else if (tone === "negative") {
        card.classList.add("kpi-negative");
    }
}

function toneFor(value) {
    if (value > 0) return "positive";
    if (value < 0) return "negative";
    return null;
}

function renderKpis(result) {
    setKpiCard(
        "kpi-return",
        "kpi-return-value",
        `${result.total_return_pct.toFixed(2)}%`,
        toneFor(result.total_return_pct)
    );
    setKpiCard("kpi-trades", "kpi-trades-value", `${result.trades}`, null);
    setKpiCard(
        "kpi-winrate",
        "kpi-winrate-value",
        result.win_rate_pct != null ? `${result.win_rate_pct.toFixed(1)}%` : "n/a",
        null
    );
    setKpiCard(
        "kpi-drawdown",
        "kpi-drawdown-value",
        result.max_drawdown_pct != null ? `${result.max_drawdown_pct.toFixed(2)}%` : "n/a",
        result.max_drawdown_pct ? "negative" : null
    );
    setKpiCard(
        "kpi-sharpe",
        "kpi-sharpe-value",
        result.sharpe_ratio != null ? result.sharpe_ratio.toFixed(2) : "n/a",
        null
    );
    setKpiCard(
        "kpi-endvalue",
        "kpi-endvalue-value",
        `$${result.ending_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
        toneFor(result.ending_value - result.starting_cash)
    );
    document.getElementById("kpi-grid").hidden = false;
}

function renderEquityChart(equityCurve) {
    const container = document.getElementById("chart-container");
    const canvas = document.getElementById("equity-chart");

    if (!equityCurve || equityCurve.length === 0) {
        container.hidden = true;
        return;
    }
    container.hidden = false;

    const labels = equityCurve.map((p) => p.date);
    const values = equityCurve.map((p) => p.equity);

    if (equityChart) {
        equityChart.destroy();
    }
    equityChart = new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Equity",
                    data: values,
                    borderColor: "#1a7a35",
                    backgroundColor: "rgba(26, 122, 53, 0.1)",
                    fill: true,
                    pointRadius: 0,
                    tension: 0.1,
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                x: { ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: "Account value" } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

document.getElementById("backtest-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const messageEl = document.getElementById("backtest-message");
    document.getElementById("kpi-grid").hidden = true;
    document.getElementById("chart-container").hidden = true;
    messageEl.textContent = "Running backtest...";

    const strategyName = document.getElementById("strategy").value;
    const startDate = document.getElementById("start_date").value;
    const endDate = document.getElementById("end_date").value;

    fetch("/backtests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            strategy_name: strategyName,
            symbol: GOLD_SYMBOL,
            start_date: startDate,
            end_date: endDate,
        }),
    })
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || "Backtest failed");
            }
            return data;
        })
        .then((result) => {
            messageEl.textContent = "";
            renderKpis(result);
            renderEquityChart(result.equity_curve);
        })
        .catch((err) => {
            messageEl.textContent = `Error: ${err.message}`;
        });
});

function applyJarvisTone(bias) {
    const panel = document.getElementById("jarvis-status");
    panel.classList.remove("jarvis-bullish", "jarvis-bearish", "jarvis-neutral");
    if (bias === "bullish") {
        panel.classList.add("jarvis-bullish");
    } else if (bias === "bearish") {
        panel.classList.add("jarvis-bearish");
    } else {
        panel.classList.add("jarvis-neutral");
    }
}

function loadJarvisStatus() {
    const statusEl = document.getElementById("jarvis-status");
    statusEl.textContent = "Loading...";
    applyJarvisTone(null);

    fetch("/dashboard/status/jarvis")
        .then((res) => res.json())
        .then((data) => {
            if (data.error) {
                statusEl.textContent = data.error;
                applyJarvisTone(null);
                return;
            }

            const lines = [`Technical signal (icc_gold): ${data.technical_signal}`];

            if (data.ai_error) {
                lines.push(data.ai_error);
                applyJarvisTone(null);
            } else {
                lines.push(`Jarvis bias: ${data.ai_bias} (confidence ${data.ai_confidence_score})`);
                lines.push(`Jarvis advice: ${data.ai_trading_advice}`);
                if (data.ai_rationale) {
                    lines.push(`Rationale: ${data.ai_rationale}`);
                }
                lines.push(data.agrees ? "Technical and AI agree ✓" : "Technical and AI do not agree");
                applyJarvisTone(data.ai_bias);
            }

            statusEl.textContent = lines.join("\n");
        })
        .catch(() => {
            statusEl.textContent = "Unable to reach Jarvis status endpoint.";
            applyJarvisTone(null);
        });
}

document.getElementById("refresh-jarvis").addEventListener("click", loadJarvisStatus);
loadJarvisStatus();
