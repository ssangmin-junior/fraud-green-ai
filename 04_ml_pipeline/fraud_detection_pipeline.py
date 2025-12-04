from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# =============================================================================
# [환경 설정]
# 로컬 환경에 맞게 아래 경로와 네트워크 설정을 변경해야 합니다.
# 1. HOST_DATA_PATH: 호스트(Windows)의 데이터 폴더 절대 경로
#    - DockerOperator는 호스트의 경로를 컨테이너에 마운트합니다.
# 2. NETWORK_NAME: MLflow, MinIO 등이 실행 중인 Docker Network 이름
#    - 'docker compose ps' 또는 'docker network ls'로 확인 가능
#    - 보통 '폴더명_네트워크명' 형식입니다 (예: docker_green_ml_net).
# =============================================================================
HOST_DATA_PATH = "d:/folder/ml/fraud-green-ai/data" 
NETWORK_NAME = "docker_green_ml_net" 
DOCKER_IMAGE = "fraud-detection-train:latest"

# DAG 기본 설정
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
    description='End-to-End Fraud Detection Pipeline (Preprocessing -> Train -> Benchmark)',
    schedule_interval=None, # 수동 실행 (Trigger)
    start_date=days_ago(1),
    tags=['mlops', 'fraud-detection'],
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # -------------------------------------------------------------------------
    # 1. Preprocessing Task
    # -------------------------------------------------------------------------
    # 실제로는 loader.py를 실행하거나 데이터 유효성을 검사합니다.
    # 여기서는 데이터 파일이 존재하는지 확인하는 명령어를 실행합니다.
    preprocessing = DockerOperator(
        task_id='preprocessing_check',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command="python -c 'import os; assert os.path.exists(\"/app/data/raw/train_transaction.csv\"), \"Data not found!\"'",
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK_NAME,
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
        ],
        # 헬스체크 실패 등으로 인한 타임아웃 방지
        mount_tmp_dir=False, 
    )

    # -------------------------------------------------------------------------
    # 2. Training & Benchmark Tasks (Parallel)
    # -------------------------------------------------------------------------
    # Baseline, Heavy, Light 모델을 병렬로 학습하고 각각 벤치마크를 수행합니다.
    
    models = ["baseline", "heavy", "light"]
    
    # 리포팅을 위해 벤치마크 태스크들을 모아둘 리스트
    benchmark_tasks = []

    with TaskGroup("model_training_group") as training_group:
        for model_type in models:
            # 2-1. Training Task
            train_task = DockerOperator(
                task_id=f'train_{model_type}',
                image=DOCKER_IMAGE,
                api_version='auto',
                auto_remove=True,
                # train.py 실행 (Run ID는 /app/data/run_id_{model_type}.txt에 저장됨)
                command=f"python src/models/train.py --model_type {model_type} --epochs 1",
                docker_url="unix://var/run/docker.sock",
                network_mode=NETWORK_NAME,
                mounts=[
                    Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
                ],
                environment={
                    "MLFLOW_TRACKING_URI": "http://mlflow_server:5000",
                    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
                    "AWS_ACCESS_KEY_ID": "minio_admin",
                    "AWS_SECRET_ACCESS_KEY": "minio_password"
                },
                mount_tmp_dir=False,
            )

            # 2-2. Benchmark Task
            # 이전 단계에서 저장한 Run ID를 읽어서 벤치마크 스크립트에 전달합니다.
            # sh -c를 사용하여 쉘 커맨드로 파일 내용을 읽습니다.
            benchmark_cmd = f"""
            sh -c '
            run_id=$(cat /app/data/run_id_{model_type}.txt) && 
            echo "Found Run ID: $run_id" && 
            python src/serving/benchmark.py --model_type {model_type} --run_id $run_id
            '
            """
            
            benchmark_task = DockerOperator(
                task_id=f'benchmark_{model_type}',
                image=DOCKER_IMAGE,
                api_version='auto',
                auto_remove=True,
                command=benchmark_cmd,
                docker_url="unix://var/run/docker.sock",
                network_mode=NETWORK_NAME,
                mounts=[
                    Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
                ],
                environment={
                    "MLFLOW_TRACKING_URI": "http://mlflow_server:5000",
                    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
                    "AWS_ACCESS_KEY_ID": "minio_admin",
                    "AWS_SECRET_ACCESS_KEY": "minio_password"
                },
                mount_tmp_dir=False,
            )

            # 의존성 설정: 전처리 -> 학습 -> 벤치마크
            preprocessing >> train_task >> benchmark_task
            benchmark_tasks.append(benchmark_task)

    # -------------------------------------------------------------------------
    # 3. Reporting Task
    # -------------------------------------------------------------------------
    # 모든 벤치마크가 끝나면 리포트를 생성(여기서는 단순 로그 출력)합니다.
    report = BashOperator(
        task_id='generate_report',
        bash_command='echo "All benchmarks completed. Generating report..."',
    )

    # 전체 DAG 흐름 연결
    start >> preprocessing
    
    # TaskGroup 내의 마지막 태스크들(benchmark_tasks)이 끝나면 report 실행
    for task in benchmark_tasks:
        task >> report
        
    report >> end
