from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderCreate(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(..., gt=0)
    limit_price: float | None = Field(default=None, gt=0)


class Order(OrderCreate):
    id: str
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    filled_price: float | None = None
