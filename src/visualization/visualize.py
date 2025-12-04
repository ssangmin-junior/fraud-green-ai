import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient

# 1. MLflow 설정 (Docker 내부 주소)
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(remote_server_uri)

def plot_results():
    print(">>> [Step 1] Connecting to MLflow Server...")
    client = MlflowClient()
    
    # 가장 최근 실험 결과 가져오기
    experiment = mlflow.get_experiment_by_name("IEEE_CIS_Real_Data_Run")
    if experiment is None:
        print("❌ 실험을 찾을 수 없습니다.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("❌ 실행된 런(Run)이 없습니다.")
        return
        
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f">>> Found Latest Run ID: {run_id}")

    # --- [그래프 1] 학습 곡선 (Loss & AUC) ---
    print(">>> [Step 2] Fetching Training Metrics...")
    
    # Loss 기록 가져오기
    loss_history = client.get_metric_history(run_id, "train_loss")
    losses = [m.value for m in loss_history]
    epochs = [m.step + 1 for m in loss_history]
    
    # AUC 값 가져오기 (마지막 값)
    auc_val = latest_run.data.metrics.get("val_roc_auc", 0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, marker='o', label='Training Loss', color='tab:blue')
    plt.title(f"Training Convergence (Final AUC: {auc_val:.4f})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("/app/data/training_curve.png") # 결과 저장
    print(">>> Saved: /app/data/training_curve.png")

    # --- [그래프 2] 탄소 배출량 (Energy Profile) ---
    print(">>> [Step 3] Downloading Emissions Data...")
    
    try:
        # [수정] CodeCarbon 원본이 아닌, 우리가 직접 만든 시계열 파일 다운로드
        local_path = client.download_artifacts(run_id, "emissions_series.csv", dst_path="/tmp")
        df_emissions = pd.read_csv(local_path)
        
        # [디버깅용] 데이터가 제대로 읽혔는지 확인
        print(f">>> Time-Series Data Loaded: {len(df_emissions)} rows")
        print(df_emissions.tail())

        # 데이터 포인트 부족 체크
        if len(df_emissions) < 2:
            print("⚠️ 데이터 포인트가 부족하여(2개 미만) 선 그래프를 그릴 수 없습니다.")
            return

        # [단위 변환] kg -> mg (1,000,000배) : 그래프가 잘 보이게 하기 위함
        df_emissions['emissions_mg'] = df_emissions['emissions'] * 1e6

        plt.figure(figsize=(10, 5))
        
        # 누적 배출량 그래프 (x축: elapsed_seconds 사용)
        sns.lineplot(data=df_emissions, x='elapsed_seconds', y='emissions_mg', color='tab:green', marker='o', linewidth=2)
        plt.fill_between(df_emissions['elapsed_seconds'], df_emissions['emissions_mg'], color='tab:green', alpha=0.3)
        
        plt.title("Cumulative Carbon Emissions (Time-Series Snapshot)")
        plt.xlabel("Training Time (seconds)")
        plt.ylabel("CO2 Emissions (mg)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig("/app/data/emissions_profile.png")
        print(">>> Saved: /app/data/emissions_profile.png")
        
    except Exception as e:
        print(f"⚠️ 탄소 데이터 시각화 실패: {e}")

if __name__ == "__main__":
    plot_results()