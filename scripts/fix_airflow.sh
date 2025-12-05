#!/bin/bash

echo "🚀 Airflow 경로 자동 수정 스크립트 시작..."

# 1. 스크립트가 있는 폴더에서 상위 폴더(프로젝트 루트)로 이동
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "📍 프로젝트 위치: $PROJECT_ROOT"

# 2. 데이터 폴더 확인 및 생성
if [ ! -d "data/raw" ]; then
    echo "⚠️ data/raw 폴더가 없어서 생성합니다."
    mkdir -p data/raw
fi

# 3. DAG 파일 경로 수정 (HOST_DATA_PATH)
DAG_FILE="dags/fraud_detection_pipeline.py"
if [ -f "$DAG_FILE" ]; then
    echo "🛠️ DAG 파일의 경로를 수정합니다..."
    # sed 명령어로 HOST_DATA_PATH 라인 전체를 교체 (맥환 호환성 위해 -i '' 사용)
    # 구분자를 |로 사용하여 경로 슬래시(/) 충돌 방지
    sed -i '' "s|HOST_DATA_PATH = .*|HOST_DATA_PATH = \"$PROJECT_ROOT/data\"|g" "$DAG_FILE"
    
    echo "✅ 수정된 경로 확인:"
    grep "HOST_DATA_PATH =" "$DAG_FILE"
else
    echo "❌ DAG 파일을 찾을 수 없습니다! (dags 폴더 확인 필요)"
    exit 1
fi

# 4. Airflow 스케줄러 재시작
echo "🔄 Airflow 스케줄러 서비스를 재시작합니다..."
if [ -f "docker/docker-compose-airflow.yaml" ]; then
    docker-compose -f docker/docker-compose-airflow.yaml restart airflow-scheduler
else
    echo "❌ docker-compose-airflow.yaml 파일을 찾을 수 없습니다."
    exit 1
fi

echo "✨ 모든 작업 완료! Airflow 웹사이트에서 파이프라인을 Clear/Restart 하세요."
