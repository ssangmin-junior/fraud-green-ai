import pytest
import sys
import os
import torch

# Add src to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.train import BaselineModel, LightModel, HeavyTransformer

def test_baseline_model_forward():
    input_dim = 10
    model = BaselineModel(input_dim)
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, 1)

def test_light_model_forward():
    input_dim = 10
    model = LightModel(input_dim)
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, 1)

def test_heavy_model_forward():
    input_dim = 10
    model = HeavyTransformer(input_dim)
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, 1)
