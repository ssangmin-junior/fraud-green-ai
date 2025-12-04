import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.loader import load_and_preprocess

@patch('pandas.read_csv')
def test_load_and_preprocess(mock_read_csv):
    # Mock data
    mock_trans = pd.DataFrame({
        'TransactionID': [1, 2, 3],
        'isFraud': [0, 1, 0],
        'col1': [10, 20, 30],
        'col2': ['a', 'b', 'a']
    })
    mock_identity = pd.DataFrame({
        'TransactionID': [1, 2],
        'id_01': [1.0, 2.0]
    })
    
    # Configure mock to return these dataframes
    # load_and_preprocess calls read_csv twice
    mock_read_csv.side_effect = [mock_trans, mock_identity]
    
    X, y = load_and_preprocess(data_dir="dummy/path")
    
    assert X.shape[0] == 3 # 3 rows
    assert y.shape[0] == 3
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
