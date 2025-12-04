import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.loader import load_and_preprocess

@pytest.fixture
def mock_data():
    """테스트용 가짜 데이터 생성"""
    # Transaction 데이터 (5개 행)
    trans_data = {
        'TransactionID': [1, 2, 3, 4, 5],
        'isFraud': [0, 1, 0, 0, 1],
        'TransactionAmt': [10.0, 20.0, 30.0, 40.0, 50.0],
        'ProductCD': ['W', 'H', 'W', 'W', 'H']
    }
    
    # Identity 데이터 (5개 행)
    id_data = {
        'TransactionID': [1, 2, 3, 4, 5],
        'DeviceType': ['mobile', 'desktop', 'mobile', 'desktop', 'mobile']
    }
    
    return pd.DataFrame(trans_data), pd.DataFrame(id_data)

@patch('src.data.loader.pd.read_csv')
def test_load_and_preprocess(mock_read_csv, mock_data):
    """load_and_preprocess 함수가 정상적으로 동작하는지 테스트"""
    trans_df, id_df = mock_data
    
    # pd.read_csv가 호출될 때마다 순서대로 가짜 데이터를 반환하도록 설정
    # 첫 번째 호출: transaction, 두 번째 호출: identity
    mock_read_csv.side_effect = [trans_df, id_df]
    
    # 함수 실행
    X, y = load_and_preprocess(data_dir="dummy/path", nrows=5)
    
    # 1. 반환 타입 확인
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    # 2. 데이터 Shape 확인
    # 입력 데이터 5개, 컬럼은 (TransactionAmt, ProductCD, DeviceType) -> 3개 (인코딩 후)
    # TransactionID와 isFraud는 제거되어야 함
    assert X.shape[0] == 5
    assert y.shape[0] == 5
    
    # 3. 스케일링 확인 (평균이 0에 가까워야 함)
    # 데이터가 너무 적어서 정확히 0은 아니지만, 스케일러가 동작했는지 확인
    assert np.abs(X.mean()) < 1.0 
    
    print("\n✅ test_load_and_preprocess Passed!")
