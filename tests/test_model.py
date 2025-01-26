import pytest
import numpy as np
from src.m2_model_training import manual_parameter_experiments as train_and_log


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
    X_test = np.array([
        [5.3,3.7,1.5,0.2],  
        [5.0,3.3,1.4,0.2],  
        [7.0,3.2,4.7,1.4],  
        [6.4,3.2,4.5,1.5]   
    ])
    y_test = np.array([2, 2, 2, 2]) 
    preds = model.predict(X_test)

    mse = ((preds - y_test) ** 2).mean()

    assert mse < 1e-10, f"MSE is {mse}, expected something very close to 0."


@pytest.mark.parametrize(
    "test_input,expected_output",
    [
        ([5.1, 3.5, 1.4, 0.2], 2),  # Class 0
        ([6.2, 2.8, 4.8, 1.8], 2),  # Class 1
        ([4.9, 3.0, 1.4, 0.2], 2),  # Class 0
    ],
)
def test_single_prediction(test_input, expected_output):
    """
    Verify single-value predictions for multi-feature input.
    """
    model = train_and_log()
    pred = model.predict([test_input])[0]  # Use test_input here
    assert pred == expected_output, f"Got {pred}, expected {expected_output}"

