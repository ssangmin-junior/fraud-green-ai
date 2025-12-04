# Workflow Orchestration with Airflow & DockerOperator

이 폴더는 Airflow를 사용하여 머신러닝 파이프라인(전처리 -> 학습 -> 벤치마크)을 자동화하는 DAG 코드를 포함하고 있습니다.

## 1. 사전 준비 (Prerequisites)

### 1.1 Docker 이미지 빌드
DAG에서 사용할 `fraud-detection-train` 이미지가 로컬에 빌드되어 있어야 합니다.
프로젝트 루트에서 다음 명령어를 실행하세요:

```bash
docker build -t fraud-detection-train:latest -f docker/Dockerfile.train .
```

### 1.2 Airflow 설정 (중요: Docker-in-Docker)
`DockerOperator`를 사용하려면 Airflow Worker 컨테이너가 호스트의 Docker 데몬에 접근할 수 있어야 합니다.
`docker/docker-compose-airflow.yaml` 파일은 이미 이 설정이 되어 있습니다:

```yaml
services:
  airflow-webserver:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock # 필수!
```

## 2. DAG 설정 및 배포

### 2.1 경로 및 네트워크 설정
`fraud_detection_pipeline.py` 파일을 열어 상단의 설정을 확인하세요:

```python
HOST_DATA_PATH = "d:/folder/ml/fraud-green-ai/data"  # 본인의 실제 데이터 경로로 수정
NETWORK_NAME = "docker_green_ml_net"                 # MLflow가 실행 중인 네트워크 이름
```
* `NETWORK_NAME` 확인법: 터미널에서 `docker network ls` 입력 후 `green_ml_net`이 포함된 이름을 찾으세요.

### 2.2 DAG 배포
Airflow 컨테이너가 `dags` 폴더를 마운트하고 있으므로, 이 폴더에 있는 파일은 자동으로 Airflow에 반영됩니다.

## 3. 파이프라인 구조 (Incremental Learning)
메모리 부족(OOM)을 방지하기 위해 데이터를 쪼개서 학습하는 방식을 사용합니다.

1.  **Preprocessing (Split)**: 전체 데이터를 Chunk 단위로 쪼개서 `train_part_X.pt`로 저장
2.  **Sequential Training**:
    *   **Chunk 0**: 처음부터 학습 -> 모델 저장
    *   **Chunk 1**: 저장된 모델을 불러와서(Resume) 이어서 학습 -> 저장
    *   ... (반복)
3.  **Benchmark**: 최종 모델로 추론 속도 측정
4.  **Report**: 완료 메시지 출력

## 4. 트러블슈팅 (Troubleshooting)

### 4.1 Localhost 연결 거부 / Docker 응답 없음
Airflow 웹서버(`localhost:8080`)나 MLflow(`localhost:5000`)에 접속이 안 되거나, `docker ps` 명령어가 멈출 경우 Docker 엔진(WSL)이 과부하로 멈춘 상태일 수 있습니다.

**해결 방법 (Docker 강제 재시작):**

1.  **Docker Desktop 종료**: 작업 표시줄 트레이 아이콘 우클릭 -> Quit Docker Desktop
2.  **WSL/프로세스 강제 종료** (PowerShell 관리자 모드):
    ```powershell
    wsl --shutdown
    taskkill /F /IM "Docker Desktop.exe"
    taskkill /F /IM com.docker.backend.exe
    ```
3.  **Docker Desktop 다시 실행**: 초록불이 들어올 때까지 대기
4.  **컨테이너 초기화 및 재실행**:
    ```powershell
    cd d:\folder\ml\fraud-green-ai
    docker rm -f $(docker ps -aq)
    docker-compose -f docker/docker-compose.yml up -d
    docker-compose -f docker/docker-compose-airflow.yaml up -d
    ```

### 4.2 메모리 부족 (OOM) 방지
대용량 데이터를 한 번에 학습하면 Docker가 멈출 수 있습니다.
현재 파이프라인은 이를 방지하기 위해 **Incremental Learning (Chunk 단위 학습)**이 적용되어 있습니다.
- `preprocessing_split`: 데이터를 작은 파일(`train_part_X.pt`)로 분할
- `train_chunk_X`: 분할된 데이터를 순서대로 학습
