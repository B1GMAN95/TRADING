from strategies.examples.sma_crossover import SmaCrossoverStrategy
from strategies.icc_strategy import ICCStrategy
from strategies.smc_strategy import SMCStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SmaCrossoverStrategy,
    "icc_gold": ICCStrategy,
    "smc_gold": SMCStrategy,
}


def get_strategy(name: str) -> type:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc
