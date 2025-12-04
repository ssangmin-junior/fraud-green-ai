import mlflow
from mlflow.tracking import MlflowClient
import argparse

def register_best_model(experiment_name="fraud-detection-experiment", metric="accuracy"):
    """
    실험에서 가장 성능이 좋은 모델을 찾아 Model Registry에 등록하고 Production으로 승격합니다.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    # 1. 실험 ID 가져오기
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id
    print(f">>> Searching best model in experiment '{experiment_name}' (ID: {experiment_id})...")

    # 2. 가장 좋은 Run 찾기
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} DESC"], # 내림차순 정렬 (높을수록 좋음)
        max_results=1
    )
    
    if not runs:
        print("❌ No runs found.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_score = best_run.data.metrics.get(metric, 0.0)
    
    print(f"✅ Best Run ID: {best_run_id} ({metric}: {best_score:.4f})")
    
    # 3. 모델 등록 (Model Registry)
    model_name = "fraud-detection-prod"
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f">>> Registering model '{model_name}' from {model_uri}...")
    model_version = mlflow.register_model(model_uri, model_name)
    
    # 4. Production 스테이지로 승격
    print(f">>> Transitioning version {model_version.version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True # 기존 Production 모델은 Archived로 이동
    )
    
    print("🎉 Model successfully registered and promoted to Production!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="fraud-detection-experiment")
    parser.add_argument("--metric", type=str, default="accuracy")
    args = parser.parse_args()
    
    register_best_model(args.experiment_name, args.metric)
