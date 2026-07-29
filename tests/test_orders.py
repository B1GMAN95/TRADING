from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_create_and_get_order() -> None:
    payload = {"symbol": "AAPL", "side": "buy", "quantity": 10}
    create_response = client.post("/orders", json=payload)
    assert create_response.status_code == 201
    order = create_response.json()
    assert order["symbol"] == "AAPL"
    assert order["status"] == "pending"

    get_response = client.get(f"/orders/{order['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == order["id"]


def test_get_unknown_order_returns_404() -> None:
    response = client.get("/orders/does-not-exist")
    assert response.status_code == 404
