import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://mlflow_server:5000")

def compare_runs():
    print(">>> Fetching all runs...")
    df = mlflow.search_runs(experiment_names=["Fraud_Detection_Benchmark"])
    
    # 필요한 컬럼만 선택
    cols = ['tags.mlflow.runName', 'metrics.val_roc_auc', 'metrics.total_emissions_kg', 'params.model_type']
    # 실제 컬럼명이 다를 수 있으므로 확인 필요
    available_cols = [c for c in cols if c in df.columns]
    df_results = df[available_cols].dropna()
    
    print(df_results)
    
    if len(df_results) == 0:
        print("❌ 데이터가 없습니다.")
        return

    # 컬럼명 단순화
    df_results.rename(columns={
        'metrics.val_roc_auc': 'AUC',
        'metrics.total_emissions_kg': 'Emissions (kg)',
        'params.model_type': 'Model'
    }, inplace=True)

    # 차트 그리기
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_results, 
        x='Emissions (kg)', 
        y='AUC', 
        hue='Model', 
        style='Model', 
        s=200 # 점 크기
    )
    
    plt.title("Green AI Trade-off: Performance vs. Energy Cost")
    plt.xlabel("Total Carbon Emissions (kg) [Lower is Better]")
    plt.ylabel("ROC-AUC Score [Higher is Better]")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 최적점(오른쪽 아래) 강조
    plt.savefig("/app/data/tradeoff_chart.png")
    print(">>> Saved: /app/data/tradeoff_chart.png")

if __name__ == "__main__":
    compare_runs()