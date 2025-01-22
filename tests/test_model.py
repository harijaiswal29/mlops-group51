import pytest
from src.model import train_model

def test_model_training():
    model = train_model()
    assert hasattr(model, "predict"), "The model should have a predict method."
    assert model.coef_[0] != 0, "Coefficient should not be zero for this simple test."
