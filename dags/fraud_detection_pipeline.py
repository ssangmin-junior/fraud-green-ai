from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

HOST_DATA_PATH = "/Users/macmini/.gemini/antigravity/scratch/fraud-green-ai/actions-runner/_work/fraud-green-ai/fraud-green-ai/data"
NETWORK_NAME = "docker_green_ml_net" 
DOCKER_IMAGE = "fraud-detection-train:latest"

default_args = {
    'owner': 'antigravity',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    'fraud_detection_pipeline',
    default_args=default_args,
    description='End-to-End Fraud Detection Pipeline',
    schedule_interval=None,
    start_date=days_ago(1),
    tags=['mlops', 'fraud-detection'],
    catchup=False,
) as dag:

    # 1. 전처리
    preprocessing = DockerOperator(
        task_id='preprocessing_split',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command="python -c 'from data.loader import preprocess_and_save_chunks; preprocess_and_save_chunks()'",
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK_NAME,
        mounts=[Mount(source=HOST_DATA_PATH, target="/app/data", type="bind")],
        environment={"PYTHONPATH": "/app/src"},
        mount_tmp_dir=False,
    )

    # 2. 학습 (간소화: 3가지 모델만 순차 실행)
    train_baseline = DockerOperator(
        task_id='train_baseline',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        # [수정] 전체 CSV 대신 전처리된 청크 파일(.pt) 로드하여 메모리 절약
        command="python src/models/train.py --model_type baseline --epochs 1 --data_path /app/data/processed/train_part_0.pt",
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK_NAME,
        mounts=[Mount(source=HOST_DATA_PATH, target="/app/data", type="bind")],
        environment={
            "PYTHONPATH": "/app/src",
            "MLFLOW_TRACKING_URI": "http://mlflow_server:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minio_admin",
            "AWS_SECRET_ACCESS_KEY": "minio_password"
        },
        mount_tmp_dir=False,
    )

    # 3. 모델 등록
    register_model = DockerOperator(
        task_id='register_best_model',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command="python src/serving/register_model.py --experiment_name Fraud_Detection_Benchmark --metric accuracy",
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK_NAME,
        mounts=[Mount(source=HOST_DATA_PATH, target="/app/data", type="bind")],
        environment={
            "PYTHONPATH": "/app/src",
            "MLFLOW_TRACKING_URI": "http://mlflow_server:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minio_admin",
            "AWS_SECRET_ACCESS_KEY": "minio_password"
        },
        mount_tmp_dir=False,
    )

    # 4. 리포트
    report = BashOperator(
        task_id='generate_report',
        bash_command='echo "Pipeline Completed!"',
    )

    # 연결
    preprocessing >> train_baseline >> register_model >> report
