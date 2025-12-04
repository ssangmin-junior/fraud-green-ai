from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String, Int64

# 1. Entity 정의 (데이터의 주체)
# 여기서는 TransactionID가 주체입니다.
transaction = Entity(name="transaction", join_keys=["TransactionID"])

# 2. Data Source 정의 (Offline Store)
# 학습용 데이터가 어디에 있는지 알려줍니다. (Parquet 파일 권장)
transaction_stats_source = FileSource(
    name="transaction_stats_source",
    path="d:/folder/ml/fraud-green-ai/data/processed/transaction_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Feature View 정의 (피처 그룹)
transaction_stats_fv = FeatureView(
    name="transaction_stats",
    entities=[transaction],
    ttl=timedelta(days=1), # 피처의 유효 기간
    schema=[
        Field(name="TransactionAmt", dtype=Float32),
        Field(name="ProductCD", dtype=String),
        Field(name="DeviceType", dtype=String),
        Field(name="isFraud", dtype=Int64),
    ],
    online=True, # Online Store에도 저장 (실시간 서빙용)
    source=transaction_stats_source,
    tags={"team": "fraud_detection"},
)
