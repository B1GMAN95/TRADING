"""Global risk gate: ATR-based position sizing, a daily max-drawdown lockout,
and breakeven-stop logic shared by every strategy.
"""

from datetime import date


class RiskManagerError(RuntimeError):
    """Raised when a risk calculation receives invalid input."""


def calculate_position_size(
    account_balance: float,
    atr: float,
    risk_pct: float = 0.01,
    atr_multiplier: float = 1.5,
    price: float | None = None,
) -> float:
    """Size a position so a stop placed `atr_multiplier` * ATR away risks `risk_pct`
    of the account balance.

    When `price` is given, the result is also capped so the position's notional
    value (size * price) never exceeds 95% of the account balance - a small
    ATR (e.g. on a lower timeframe) would otherwise imply a position the
    account can't actually afford, which a real broker would reject as a
    margin failure. The 5% headroom leaves room for commission and any
    slippage between the signal price and the actual fill.

    Returns the number of units/contracts to trade.
    """
    if account_balance <= 0:
        raise RiskManagerError("account_balance must be positive")
    if atr <= 0:
        raise RiskManagerError("atr must be positive")
    if not 0 < risk_pct <= 1:
        raise RiskManagerError("risk_pct must be between 0 and 1")
    if price is not None and price <= 0:
        raise RiskManagerError("price must be positive")

    risk_amount = account_balance * risk_pct
    stop_distance = atr * atr_multiplier
    size = risk_amount / stop_distance

    if price is not None:
        max_affordable_size = (account_balance * 0.95) / price
        size = min(size, max_affordable_size)

    return size


def breakeven_stop_price(
    direction: str,
    entry_price: float,
    initial_stop_price: float,
    current_price: float,
) -> float | None:
    """Return the entry price as the new stop once a trade reaches 1:1 reward-to-risk.

    Returns None while breakeven hasn't been reached yet, meaning the caller
    should leave the existing stop in place.
    """
    initial_risk = abs(entry_price - initial_stop_price)
    if initial_risk <= 0:
        return None

    if direction == "long" and current_price >= entry_price + initial_risk:
        return entry_price
    if direction == "short" and current_price <= entry_price - initial_risk:
        return entry_price
    return None


class RiskManager:
    """Prop-firm style daily max-drawdown lockout.

    Tracks each trading day's starting equity and, once losses on that day
    reach `daily_max_drawdown_pct`, blocks new trades until the next day.
    """

    def __init__(self, daily_max_drawdown_pct: float = 3.0) -> None:
        self.daily_max_drawdown_pct = daily_max_drawdown_pct
        self._current_date: date | None = None
        self._daily_start_balance: float | None = None
        self._locked = False

    def update(self, equity: float, today: date) -> None:
        """Advance the tracker with the account's current equity and today's date."""
        if today != self._current_date:
            self._current_date = today
            self._daily_start_balance = equity
            self._locked = False

        if self._daily_start_balance and self._daily_start_balance > 0:
            drawdown_pct = (
                (self._daily_start_balance - equity) / self._daily_start_balance * 100
            )
            if drawdown_pct >= self.daily_max_drawdown_pct:
                self._locked = True

    def is_trading_allowed(self) -> bool:
        return not self._locked

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def daily_start_balance(self) -> float | None:
        return self._daily_start_balance
