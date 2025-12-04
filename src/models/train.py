import os
import sys
import time
import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from codecarbon import EmissionsTracker
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 전처리 모듈 import (Docker 경로 기준)
from data.loader import load_and_preprocess

# 1. 환경 변수 및 MLflow 설정
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("Fraud_Detection_Benchmark")

# 2. 모델 아키텍처 정의 (Factory Pattern)
class BaselineModel(nn.Module):
    """기존에 사용하던 준수한 성능의 기본 모델 (MLP)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class HeavyTransformer(nn.Module):
    """Deep Transformer: 연산량이 많아 에너지를 많이 쓰지만 복잡한 패턴 학습 가능"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # 입력 차원을 트랜스포머 차원으로 투영
        self.embedding = nn.Linear(input_dim, d_model)
        # 트랜스포머 인코더 레이어 (batch_first=True 필수)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # (Batch, Features) -> (Batch, 1, Features) 시퀀스 차원 추가
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        # (Batch, 1, Features) -> (Batch, Features) -> Output
        x = x.squeeze(1)
        return self.fc(x)

class LightModel(nn.Module):
    """Distillation-ready Light Model: 매우 얕고 좁은 네트워크 (고속 추론용)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), # 128 -> 32로 파라미터 대폭 축소
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_model(model_type, input_dim):
    """입력받은 타입에 따라 모델 객체를 반환하는 팩토리 함수"""
    if model_type == "heavy":
        return HeavyTransformer(input_dim)
    elif model_type == "light":
        return LightModel(input_dim)
    else:
        return BaselineModel(input_dim)

# 3. 데이터셋 클래스
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(1) # (N,) -> (N, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model_type, epochs):
    print(f">>> [Experiment Start] Model Type: {model_type}, Epochs: {epochs}")
    
    # 데이터 로드
    X, y = load_and_preprocess(data_dir="/app/data/raw", nrows=None)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=64, shuffle=True)
    input_dim = X_train.shape[1]
    print(f">>> Input Dimension: {input_dim}")

    # CodeCarbon 설정 (소비전력 측정)
    # 수동 스냅샷 저장을 위한 리스트
    epoch_emissions = []
    start_time = time.time()
    
    # 짧은 학습 시간에도 데이터를 잡기 위해 측정 주기를 0.1초로 설정
    tracker = EmissionsTracker(output_dir="./", output_file="emissions.csv", measure_power_secs=0.1)
    tracker.start()

    with mlflow.start_run(run_name=f"{model_type}_run"):
        # 파라미터 로깅
        params = {"model_type": model_type, "epochs": epochs, "batch_size": 64, "input_dim": input_dim}
        mlflow.log_params(params)
        
        # [New] Run ID 저장 (Airflow 등 오케스트레이션 연동용)
        run_id = mlflow.active_run().info.run_id
        with open(f"/app/data/run_id_{model_type}.txt", "w") as f:
            f.write(run_id)
        print(f">>> Run ID saved to /app/data/run_id_{model_type}.txt: {run_id}")
        
        # 모델 및 학습 설정
        model = get_model(model_type, input_dim)
        
        # 비용 민감형 가중치 계산
        pos_weight = torch.tensor([len(y_train) / y_train.sum()]) 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f">>> Class Weight (pos_weight): {pos_weight.item():.2f}")
        print(">>> [Step 2] Training Start... (Capturing Time-Series)")
        
        # 학습 루프
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Epoch 기록
            avg_loss = running_loss / len(train_loader)
            print(f"[{model_type}] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # --- [수동 스냅샷 저장 로직] ---
            # 1. 현재까지의 측정값을 파일에 씀
            tracker.flush()
            # 2. 파일 쓰기 완료 대기 (그래프 시간축 확보용)
            time.sleep(0.5) 
            
            # 3. 파일을 읽어서 현재 값 백업
            try:
                if os.path.exists("emissions.csv"):
                    temp_df = pd.read_csv("emissions.csv")
                    if not temp_df.empty:
                        # 가장 마지막(최신) 배출량 가져오기
                        current_emission = temp_df['emissions'].iloc[-1]
                        
                        # 리스트에 [시간, 배출량] 추가
                        epoch_emissions.append({
                            "timestamp": time.time(),
                            "elapsed_seconds": time.time() - start_time,
                            "emissions": current_emission
                        })
                        print(f"    >> Snapshot saved: {current_emission*1e6:.4f} mg")
            except Exception as e:
                print(f"    >> Snapshot failed: {e}")

        # 최종 평가
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val)).sigmoid().numpy()
            roc_auc = roc_auc_score(y_val, val_preds)
            mlflow.log_metric("val_roc_auc", roc_auc)
            print(f">>> Final ROC-AUC: {roc_auc:.4f}")

        # 아티팩트 저장
        torch.save(model.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")
        
        # [중요] 시계열 데이터 저장 및 업로드 (그래프 그리기용)
        if epoch_emissions:
            series_df = pd.DataFrame(epoch_emissions)
            series_df.to_csv("emissions_series.csv", index=False)
            mlflow.log_artifact("emissions_series.csv")
            print(">>> Saved & Uploaded: emissions_series.csv")

        final_emissions = tracker.stop()
        
        # 전체 배출량 메트릭 기록 (비교 차트용 핵심 지표)
        mlflow.log_metric("total_emissions_kg", final_emissions)
        # emissions.csv 파일이 있다면 업로드
        if os.path.exists("emissions.csv"):
            mlflow.log_artifact("emissions.csv")
            
import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from codecarbon import EmissionsTracker
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 전처리 모듈 import (Docker 경로 기준)
from data.loader import load_and_preprocess

# 1. 환경 변수 및 MLflow 설정
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("Fraud_Detection_Benchmark")

# 2. 모델 아키텍처 정의 (Factory Pattern)
class BaselineModel(nn.Module):
    """기존에 사용하던 준수한 성능의 기본 모델 (MLP)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class HeavyTransformer(nn.Module):
    """Deep Transformer: 연산량이 많아 에너지를 많이 쓰지만 복잡한 패턴 학습 가능"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # 입력 차원을 트랜스포머 차원으로 투영
        self.embedding = nn.Linear(input_dim, d_model)
        # 트랜스포머 인코더 레이어 (batch_first=True 필수)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # (Batch, Features) -> (Batch, 1, Features) 시퀀스 차원 추가
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        # (Batch, 1, Features) -> (Batch, Features) -> Output
        x = x.squeeze(1)
        return self.fc(x)

class LightModel(nn.Module):
    """Distillation-ready Light Model: 매우 얕고 좁은 네트워크 (고속 추론용)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), # 128 -> 32로 파라미터 대폭 축소
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_model(model_type, input_dim):
    """입력받은 타입에 따라 모델 객체를 반환하는 팩토리 함수"""
    if model_type == "heavy":
        return HeavyTransformer(input_dim)
    elif model_type == "light":
        return LightModel(input_dim)
    else:
        return BaselineModel(input_dim)

# 3. 데이터셋 클래스
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X # 이미 Tensor임
        self.y = y.unsqueeze(1) # (N,) -> (N, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model_type, epochs, nrows=None, data_path=None, resume_from=None):
    print(f">>> [Experiment Start] Model Type: {model_type}, Epochs: {epochs}")
    
    # 1. 데이터 로드 (Chunk 파일 또는 Raw CSV)
    if data_path and data_path.endswith('.pt'):
        print(f">>> Loading preprocessed chunk from {data_path}...")
        data = torch.load(data_path)
        X, y = data['X'], data['y']
    else:
        # 기존 로직 (Raw CSV 로드)
        X_raw, y_raw = load_and_preprocess(data_dir="/app/data/raw", nrows=nrows)
        X = torch.tensor(X_raw)
        y = torch.tensor(y_raw)

    # Train/Val 분리 (Chunk 단위 학습 시에는 Val이 Chunk 내에서 분리됨 - 단순화)
    # 실제로는 별도의 Validation Set을 고정해두는 것이 좋음
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=64, shuffle=True)
    input_dim = X_train.shape[1]
    print(f">>> Input Dimension: {input_dim}")

    # CodeCarbon 설정
    tracker = EmissionsTracker(output_dir="./", output_file="emissions.csv", measure_power_secs=0.1)
    tracker.start()

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        # 파라미터 로깅
        params = {"model_type": model_type, "epochs": epochs, "batch_size": 64, "input_dim": input_dim}
        mlflow.log_params(params)
        
        # Run ID 저장
        run_id = run.info.run_id
        with open(f"/app/data/run_id_{model_type}.txt", "w") as f:
            f.write(run_id)
        print(f">>> Run ID saved to /app/data/run_id_{model_type}.txt: {run_id}")
        
        # 2. 모델 초기화 또는 불러오기 (Incremental Learning)
        model = get_model(model_type, input_dim)
        
        if resume_from and os.path.exists(resume_from):
            print(f">>> Resuming training from checkpoint: {resume_from}")
            model.load_state_dict(torch.load(resume_from))
        
        # 비용 민감형 가중치 계산
        pos_weight = torch.tensor([len(y_train) / (y_train.sum() + 1e-5)]) 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f">>> Class Weight (pos_weight): {pos_weight.item():.2f}")
        print(">>> [Step 2] Training Start...")
        
        # 학습 루프
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            print(f"[{model_type}] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # CodeCarbon Flush (생략 가능)
            tracker.flush()

        # 최종 평가
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).sigmoid().numpy()
            if len(np.unique(y_val)) > 1:
                roc_auc = roc_auc_score(y_val, val_preds)
                mlflow.log_metric("val_roc_auc", roc_auc)
                print(f">>> Final ROC-AUC: {roc_auc:.4f}")
            else:
                print(">>> Warning: Only one class present in validation set. Skipping AUC.")

        # 3. 모델 저장 (다음 단계를 위해)
        save_path = "model.pth"
        torch.save(model.state_dict(), save_path)
        mlflow.log_artifact(save_path)
        
        # 로컬에도 복사 (다음 Task가 읽을 수 있도록 /app/data에 저장)
        # 예: /app/data/model_baseline_latest.pth
        shared_model_path = f"/app/data/model_{model_type}_latest.pth"
        torch.save(model.state_dict(), shared_model_path)
        print(f">>> Model saved to {shared_model_path}")

        final_emissions = tracker.stop()
        mlflow.log_metric("total_emissions_kg", final_emissions)
        print(f">>> Total Emissions: {final_emissions} kg")
        print(">>> Experiment Complete!")

import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np # Added for np.unique
import os # Added for os.path.exists
from torch.utils.data import Dataset, DataLoader
from codecarbon import EmissionsTracker
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 전처리 모듈 import (Docker 경로 기준)
from data.loader import load_and_preprocess

# 1. 환경 변수 및 MLflow 설정
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("Fraud_Detection_Benchmark")

# 2. 모델 아키텍처 정의 (Factory Pattern)
class BaselineModel(nn.Module):
    """기존에 사용하던 준수한 성능의 기본 모델 (MLP)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class HeavyTransformer(nn.Module):
    """Deep Transformer: 연산량이 많아 에너지를 많이 쓰지만 복잡한 패턴 학습 가능"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # 입력 차원을 트랜스포머 차원으로 투영
        self.embedding = nn.Linear(input_dim, d_model)
        # 트랜스포머 인코더 레이어 (batch_first=True 필수)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # (Batch, Features) -> (Batch, 1, Features) 시퀀스 차원 추가
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        # (Batch, 1, Features) -> (Batch, Features) -> Output
        x = x.squeeze(1)
        return self.fc(x)

class LightModel(nn.Module):
    """Distillation-ready Light Model: 매우 얕고 좁은 네트워크 (고속 추론용)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), # 128 -> 32로 파라미터 대폭 축소
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_model(model_type, input_dim):
    """입력받은 타입에 따라 모델 객체를 반환하는 팩토리 함수"""
    if model_type == "heavy":
        return HeavyTransformer(input_dim)
    elif model_type == "light":
        return LightModel(input_dim)
    else:
        return BaselineModel(input_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "heavy", "light"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--nrows", type=int, default=None)
    # 추가된 인자
    parser.add_argument("--data_path", type=str, default=None, help="Path to preprocessed .pt file")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to previous model checkpoint")
    
    args = parser.parse_args()
    
    train(args.model_type, args.epochs, args.nrows, args.data_path, args.resume_from)