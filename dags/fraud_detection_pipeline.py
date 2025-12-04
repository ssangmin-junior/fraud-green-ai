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

    # -------------------------------------------------------------------------
    # 1. Preprocessing Task (Split Data)
    # -------------------------------------------------------------------------
    # 데이터를 Chunk 단위로 쪼개서 /app/data/processed/train_part_X.pt 로 저장합니다.
    preprocessing = DockerOperator(
        task_id='preprocessing_split',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        # loader.py에 새로 만든 함수 실행
        command="python -c 'from data.loader import preprocess_and_save_chunks; preprocess_and_save_chunks()'",
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK_NAME,
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
        ],
        mount_tmp_dir=False, 
    )

    # -------------------------------------------------------------------------
    # 2. Incremental Training & Benchmark Tasks
    # -------------------------------------------------------------------------
    # 쪼개진 데이터(Chunk 0 -> Chunk 1 -> Chunk 2)를 순서대로 학습합니다.
    
    models = ["baseline", "heavy", "light"]
    n_chunks = 3 # loader.py에서 생성하는 청크 개수와 일치해야 함
    
    # 순차 실행을 위해 이전 태스크를 기억할 변수
    previous_task = preprocessing

    for model_type in models:
        
        # 모델별로 Chunk 0 -> Chunk 1 -> Chunk 2 순서로 학습
        model_previous_task = None
        
        for chunk_idx in range(n_chunks):
            
            # 첫 번째 청크는 맨땅에서 시작, 그 이후는 이전 모델을 로드(Resume)
            resume_arg = ""
            if chunk_idx > 0:
                resume_arg = f"--resume_from /app/data/model_{model_type}_latest.pth"
            
            train_task = DockerOperator(
                task_id=f'train_{model_type}_chunk_{chunk_idx}',
                image=DOCKER_IMAGE,
                api_version='auto',
                auto_remove=True,
                # 수정된 train.py 실행 (Chunk 파일 로드 + Resume)
                command=f"python src/models/train.py --model_type {model_type} --epochs 1 --data_path /app/data/processed/train_part_{chunk_idx}.pt {resume_arg}",
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
            
            if model_previous_task:
                model_previous_task >> train_task
            else:
                # 첫 청크는 이전 모델(또는 전처리) 완료 후 시작
                previous_task >> train_task
                
            model_previous_task = train_task

        # 마지막 청크 학습이 끝나면 벤치마크 수행
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

        # 학습(마지막 청크) >> 벤치마크
        model_previous_task >> benchmark_task
        
        # 다음 모델은 현재 모델의 벤치마크가 끝난 후 시작 (전체 직렬화)
        previous_task = benchmark_task

    # -------------------------------------------------------------------------
    # 3. Model Registration Task (Best Model -> Production)
    # -------------------------------------------------------------------------
    # 모든 모델의 벤치마크가 끝나면, 가장 성능 좋은 모델을 찾아 Registry에 등록합니다.
    register_cmd = "python src/serving/register_model.py --experiment_name fraud-detection-experiment --metric accuracy"
    
    register_task = DockerOperator(
        task_id='register_best_model',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=register_cmd,
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

    # -------------------------------------------------------------------------
    # 4. Reporting Task
    # -------------------------------------------------------------------------
    report = BashOperator(
        task_id='generate_report',
        bash_command='echo "Pipeline Completed. Best model registered to Production."',
    )

    # 전체 DAG 흐름 연결
    # 모든 벤치마크 완료 -> 모델 등록 -> 리포트
    previous_task >> register_task >> report


