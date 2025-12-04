import time
import torch
import numpy as np
import pandas as pd
import argparse
import mlflow.pytorch
import matplotlib.pyplot as plt
import sys
import os

# [경로 설정] Python이 'src' 모듈을 찾을 수 있도록 현재 디렉토리의 상위 경로를 PYTHONPATH에 추가합니다.
# 이 설정은 Docker 환경에서 모듈을 로드하는 데 필수적입니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 이제 'src.models.train' 모듈에서 모델 정의를 가져올 수 있습니다.
from src.models.train import get_model 

def run_benchmark(run_id, model_type, input_dim=432, n_requests=1000):
    print(f">>> [Benchmark] Loading model from Run ID: {run_id} ({model_type})...")
    
    # 1. MLflow에서 모델 불러오기
    model = get_model(model_type, input_dim)
    
    try:
        # Docker 내부 경로의 아티팩트 다운로드 시뮬레이션 및 가중치 로드
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model.pth")
        model.load_state_dict(torch.load(local_path))
        model.eval()
    except Exception as e:
        print(f"❌ ERROR: 모델 가중치 로드 실패. Run ID '{run_id}'를 확인하거나, 파일이 MinIO에 업로드되었는지 확인하십시오. 오류: {e}")
        return
    
    # 2. 더미 데이터 생성 (Batch Size = 1, 실시간 요청 시뮬레이션)
    dummy_input = torch.randn(1, input_dim)
    
    latencies = []
    print(f">>> Starting {n_requests} inference requests for latency measurement...")
    
    # 3. 벤치마크 루프
    with torch.no_grad():
        # 워밍업 (초기 캐싱 효과 제거)
        for _ in range(100):
            model(dummy_input)
            
        # 실제 측정
        start_total = time.time()
        for _ in range(n_requests):
            t0 = time.time()
            model(dummy_input) # 추론 수행
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # 초 -> 밀리초(ms) 변환
            
    total_time = time.time() - start_total
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = n_requests / total_time
    
    print(f"--- [ {model_type.upper()} Benchmark Results ] ---")
    print(f"    [Result] Avg Latency: {avg_latency:.4f} ms")
    print(f"    [Result] P95 Latency: {p95_latency:.4f} ms")
    print(f"    [Result] Throughput : {throughput:.2f} req/s")
    
    # 4. 결과 저장 (이미지)
    plt.figure(figsize=(10, 5))
    plt.hist(latencies, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(avg_latency, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {avg_latency:.2f}ms')
    plt.title(f"Inference Latency Distribution ({model_type})")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"/app/data/latency_{model_type}.png")
    
    # 5. 결과를 MLflow에 메트릭으로 추가 기록 (기존 Run ID에 덮어씀)
    with mlflow.start_run(run_id=run_id):
        # mlflow.log_metric은 중복된 키로 기록될 경우 최신값으로 덮어씁니다.
        mlflow.log_metric("inference_latency_ms", avg_latency) 
        mlflow.log_metric("p95_latency_ms", p95_latency)
        mlflow.log_metric("throughput_req_s", throughput)
        mlflow.log_artifact(f"/app/data/latency_{model_type}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow Run ID to log results to.")
    parser.add_argument("--model_type", type=str, required=True, choices=["baseline", "heavy", "light"], help="Model architecture type.")
    args = parser.parse_args()
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow_server:5000")
    run_benchmark(args.run_id, args.model_type)