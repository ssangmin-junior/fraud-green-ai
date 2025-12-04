import pandas as pd
import os
from datetime import datetime, timedelta

def convert_to_parquet():
    """
    CSV 데이터를 읽어서 Feast용 Parquet 파일로 변환합니다.
    Feast는 시계열 기반이므로 'event_timestamp' 컬럼이 필수입니다.
    """
    data_dir = "data"
    output_path = "data/processed/transaction_stats.parquet"
    
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    print(">>> Loading raw data...")
    # 데이터가 없으면 가짜 데이터 생성 (실습용)
    if not os.path.exists(f"{data_dir}/train_transaction.csv"):
        print("⚠️ Raw data not found. Generating dummy data for practice.")
        df = pd.DataFrame({
            "TransactionID": range(1, 101),
            "TransactionAmt": [10.0 + i for i in range(100)],
            "ProductCD": ["W"] * 50 + ["H"] * 50,
            "DeviceType": ["mobile"] * 50 + ["desktop"] * 50,
            "isFraud": [0] * 90 + [1] * 10
        })
    else:
        # 실제 데이터 로드 (일부만)
        trans = pd.read_csv(f"{data_dir}/train_transaction.csv", nrows=1000)
        identity = pd.read_csv(f"{data_dir}/train_identity.csv", nrows=1000)
        df = trans.merge(identity, on='TransactionID', how='left')
        
    # 필수 컬럼 선택 및 전처리
    cols = ["TransactionID", "TransactionAmt", "ProductCD", "DeviceType", "isFraud"]
    # 실제 데이터에 DeviceType이 없을 수 있으므로 처리
    if "DeviceType" not in df.columns:
        df["DeviceType"] = "unknown"
        
    df = df[cols].fillna("unknown")
    
    # [중요] event_timestamp 추가
    # 실습을 위해 현재 시간 기준으로 과거 데이터를 생성합니다.
    # 예: 최근 1000개의 거래가 지난 24시간 동안 발생했다고 가정
    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=i) for i in range(len(df))]
    df["event_timestamp"] = timestamps
    df["created_timestamp"] = now # 레코드 생성 시간

    print(f">>> Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("✅ Conversion Complete!")
    print(df.head())

if __name__ == "__main__":
    convert_to_parquet()
