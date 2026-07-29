from strategies.examples.sma_crossover import SmaCrossoverStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SmaCrossoverStrategy,
}


def get_strategy(name: str) -> type:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc
