import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler
)  # StandardScaler 추가


def load_and_preprocess(data_dir="/app/data/raw", nrows=None):
    print(f">>> Loading data from {data_dir} (nrows={nrows})...")

    # 1. 데이터 로드
    try:
        trans = pd.read_csv(f"{data_dir}/train_transaction.csv", nrows=nrows)
        identity = pd.read_csv(f"{data_dir}/train_identity.csv", nrows=nrows)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 데이터 파일이 {data_dir}에 없습니다.")

    # 2. 데이터 병합
    df = trans.merge(identity, on='TransactionID', how='left')

    # 3. 타겟 변수 분리
    y = df['isFraud']
    X = df.drop(['isFraud', 'TransactionID'], axis=1)

    # 4. 전처리 (스케일링 추가됨)
    print(">>> Preprocessing (Encoding & Scaling)...")

    for col in X.columns:
        # 문자열(Object) 처리
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 결측치 채우기 (-1)
    X = X.fillna(-1)

    # [핵심 수정] 데이터 스케일링 (평균 0, 분산 1로 변환)
    # 신경망 학습의 필수 요소입니다.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(">>> Preprocessing Complete. Statistics:")
    print(f"    Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")

    return X_scaled.astype(np.float32), y.values.astype(np.float32)


def preprocess_and_save_chunks(
    data_dir="/app/data/raw",
    output_dir="/app/data/processed",
    chunk_size=50000
):
    """
    메모리 절약을 위해 데이터를 Chunk 단위로 읽어서 전처리 후 저장합니다.
    """
    import os
    import torch

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f">>> [Preprocessing] Splitting data from {data_dir} "
          f"into chunks (Size: {chunk_size})...")

    # 1. Identity 데이터는 작으므로 미리 로드 (Merge용)
    try:
        identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    except FileNotFoundError:
        print("Warning: Identity file not found. "
              "Proceeding with Transaction only.")
        identity = None

    # 2. Transaction 데이터를 Chunk 단위로 로드
    chunks = pd.read_csv(
        f"{data_dir}/train_transaction.csv", chunksize=chunk_size
    )

    scaler = StandardScaler()
    is_scaler_fitted = False

    saved_files = []

    for i, chunk in enumerate(chunks):
        # 병합
        if identity is not None:
            df = chunk.merge(identity, on='TransactionID', how='left')
        else:
            df = chunk

        # 타겟 분리
        y = df['isFraud']
        X = df.drop(['isFraud', 'TransactionID'], axis=1)

        # 간단한 전처리 (문자열 인코딩 & 결측치)
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(-1)

        # 스케일링 (첫 번째 청크로 fit, 나머지는 transform)
        # 주의: 엄밀하게는 전체 데이터로 fit 해야 하지만, 실습용으로 첫 청크 기준 설정
        if not is_scaler_fitted:
            X_scaled = scaler.fit_transform(X)
            is_scaler_fitted = True
        else:
            X_scaled = scaler.transform(X)

        # 저장 (PyTorch Tensor)
        save_path = f"{output_dir}/train_part_{i}.pt"
        torch.save({
            'X': torch.tensor(X_scaled, dtype=torch.float32),
            'y': torch.tensor(y.values, dtype=torch.float32)
        }, save_path)

        saved_files.append(save_path)
        print(f"    >> Saved Chunk {i}: {save_path} (Rows: {len(df)})")

        # 실습을 위해 3개 청크까지만 생성하고 중단 (시간 절약)
        if i >= 2:
            break

    return saved_files
