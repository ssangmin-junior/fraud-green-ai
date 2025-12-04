from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "fraud-detection"}

def test_predict_normal():
    # 정상 거래 (금액 적음)
    payload = {
        "TransactionAmt": 50.0,
        "ProductCD": "W"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.5

def test_predict_fraud():
    # 사기 의심 거래 (금액 큼 - Mock 로직 기준)
    payload = {
        "TransactionAmt": 1500.0,
        "ProductCD": "H"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.5
