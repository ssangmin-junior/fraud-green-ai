from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
import torch

# 1. FastAPI 앱 초기화
app = FastAPI(
    title="Green Fraud Detection API",
    description="친환경 금융 사기 탐지 모델 서빙 API",
    version="1.0.0"
)

# 2. 요청 데이터 스키마 정의 (Pydantic)
class TransactionData(BaseModel):
    TransactionAmt: float
    ProductCD: str
    DeviceType: Optional[str] = None
    # 필요한 다른 피처들도 여기에 추가

class PredictionResponse(BaseModel):
    transaction_id: Optional[int] = None
    is_fraud: bool
    fraud_probability: float

# 전역 변수로 모델 저장
model = None

@app.on_event("startup")
def load_model():
    """서버 시작 시 MLflow Model Registry에서 Production 모델을 로드합니다."""
    global model
    import mlflow.pytorch
    import os
    
    model_name = "fraud-detection-prod"
    stage = "Production"
    
    print(f">>> Loading model '{model_name}' (Stage: {stage}) from MLflow...")
    
    try:
        # MLflow Tracking URI 설정 (환경변수 또는 기본값)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # 모델 로드
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        
        print(f"✅ Model loaded successfully from {model_uri}")
        
    except Exception as e:
        print(f"⚠️ Failed to load model from MLflow: {e}")
        print(">>> Running in Mock Mode (Rule-based Fallback)")
        model = None

@app.get("/health")
def health_check():
    """서버 상태 확인용 엔드포인트"""
    return {"status": "healthy", "service": "fraud-detection"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionData):
    """
    단일 거래에 대한 사기 여부를 예측합니다.
    """
    try:
        # 1. 데이터 전처리 (실제 로직 필요)
        # 예: loader.py의 전처리 로직을 가져와서 사용해야 함
        # features = preprocess(data)
        
        # 2. 모델 추론 (Mock)
        # 실제 모델이 연결되면: output = model(features)
        
        # 임시: 거래 금액이 1000불 이상이면 사기로 예측 (테스트용)
        mock_prob = 0.95 if data.TransactionAmt > 1000 else 0.05
        is_fraud = mock_prob > 0.5
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=mock_prob
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
