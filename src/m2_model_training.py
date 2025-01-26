import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error
)
from data_utils import setup_data


def manual_parameter_experiments():
    # Define parameter combinations to try manually
    param_combinations = [
        {"C": 0.1, "solver": "lbfgs", "max_iter": 300},
        {"C": 1.0, "solver": "lbfgs", "max_iter": 500},
        {"C": 10.0, "solver": "liblinear", "max_iter": 1000}
    ]

    X_train, X_test, y_train, y_test = setup_data()
    # 3. Scale the data => yields numpy arrays without column names
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlflow.set_experiment("Logistic Regresssion Experiment")
    # Loop through each parameter combination and log results
    for params in param_combinations:
        model = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            max_iter=params["max_iter"]
        )

        with mlflow.start_run(rclsun_name="Manual_Experiment_params"):
            model.fit(X_train_scaled, y_train)

            # Evaluate on the test set
            preds_test = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, preds_test)
            precision = precision_score(y_test, preds_test, average="weighted")
            recall = recall_score(y_test, preds_test, average="weighted")
            f1 = f1_score(y_test, preds_test, average="weighted")
            mse = mean_squared_error(y_test, preds_test)
            r2 = r2_score(y_test, preds_test)

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_r2_score", r2)

            # Log model
            sample_input = pd.DataFrame([X_test_scaled[0]], dtype="float64")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="manual_logreg_model",
                input_example=sample_input
            )

        print(f"Logged experiment with params: {params}")
        print(
            f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}, "
            f"MSE: {mse:.4f}, r2_score: {r2:.4f}"
        )


if __name__ == "__main__":
    manual_parameter_experiments()
