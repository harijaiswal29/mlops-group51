import pytest
import numpy as np
from src.model import train_and_log


def test_model_training():
    """
    Ensure the train_and_log function returns a model with a 'predict' method.
    """
    model = train_and_log()
    assert hasattr(model, "predict"), "Trained model must have predict method."


def test_model_accuracy():
    """
    Check if the MSE is near zero for a perfectly linear relationship.
    """
    model = train_and_log()
    X = np.array([[i] for i in range(10)])
    y = np.array([2 * i + 1 for i in range(10)])
    preds = model.predict(X)
    mse = ((preds - y) ** 2).mean()

    assert mse < 1e-10, f"MSE is {mse}, expected something very close to 0."


@pytest.mark.parametrize(
    "value,expected",
    [(0, 1), (1, 3), (2, 5)]
)
def test_single_prediction(value, expected):
    """
    Verify single-value predictions match the formula y = 2x + 1.
    """
    model = train_and_log()
    pred = model.predict([[value]])[0]
    assert abs(pred - expected) < 1e-7, f"Got {pred}, expected {expected}"