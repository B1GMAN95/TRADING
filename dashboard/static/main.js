const GOLD_SYMBOL = "XAUUSD";

fetch("/health")
    .then((res) => res.json())
    .then((data) => {
        document.getElementById("status").textContent = data.status;
    })
    .catch(() => {
        document.getElementById("status").textContent = "unreachable";
    });

document.getElementById("backtest-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const resultEl = document.getElementById("backtest-result");
    resultEl.textContent = "Running backtest...";

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
        .then((data) => {
            resultEl.textContent = JSON.stringify(data, null, 2);
        })
        .catch((err) => {
            resultEl.textContent = `Error: ${err.message}`;
        });
});

function loadJarvisStatus() {
    const statusEl = document.getElementById("jarvis-status");
    statusEl.textContent = "Loading...";

    fetch("/dashboard/status/jarvis")
        .then((res) => res.json())
        .then((data) => {
            if (data.error) {
                statusEl.textContent = data.error;
                return;
            }

            const lines = [`Technical signal (icc_gold): ${data.technical_signal}`];

            if (data.ai_error) {
                lines.push(data.ai_error);
            } else {
                lines.push(`Jarvis bias: ${data.ai_bias} (confidence ${data.ai_confidence_score})`);
                lines.push(`Jarvis advice: ${data.ai_trading_advice}`);
                if (data.ai_rationale) {
                    lines.push(`Rationale: ${data.ai_rationale}`);
                }
                lines.push(data.agrees ? "Technical and AI agree ✓" : "Technical and AI do not agree");
            }

            statusEl.textContent = lines.join("\n");
        })
        .catch(() => {
            statusEl.textContent = "Unable to reach Jarvis status endpoint.";
        });
}

document.getElementById("refresh-jarvis").addEventListener("click", loadJarvisStatus);
loadJarvisStatus();
