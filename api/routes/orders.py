from uuid import uuid4

from fastapi import APIRouter, HTTPException

from models.schemas.order import Order, OrderCreate

router = APIRouter(prefix="/orders", tags=["orders"])

_orders: dict[str, Order] = {}


@router.post("", response_model=Order, status_code=201)
def create_order(order_in: OrderCreate) -> Order:
    order = Order(id=str(uuid4()), **order_in.model_dump())
    _orders[order.id] = order
    return order


@router.get("", response_model=list[Order])
def list_orders() -> list[Order]:
    return list(_orders.values())


@router.get("/{order_id}", response_model=Order)
def get_order(order_id: str) -> Order:
    order = _orders.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return order
