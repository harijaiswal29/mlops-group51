# src/mlflow_training.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def train_and_log(run_name="default-run", alpha=1.0, l1_ratio=0.5):
    """
    Train a simple linear model and log metrics/params to MLflow.
    alpha, l1_ratio are placeholders in case you
    want to try a different regressor that uses them
    (e.g., ElasticNet).
    """

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Generate dummy data
        X = np.array([[i] for i in range(10)])
        y = np.array([2*i + 1 for i in range(10)])

        # For demonstration, let's just use LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions and compute metrics
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        # Log metrics
        mlflow.log_metric("mse", mse)

        # Log the model artifact
        mlflow.sklearn.log_model(model, "model_artifact")

        print(f"Run: {run_name}, MSE: {mse}")
        return model


if __name__ == "__main__":
    # Example: run the default training with no special parameters
    train_and_log()
