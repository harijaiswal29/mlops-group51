import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from data_utils import setup_data


def tune_logistic_regression():

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = setup_data()

    # 3. Scale the data => yields numpy arrays without column names
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define parameter grid
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],  # multi-class friendly solvers
        "max_iter": [300, 500, 1000]       # allow more iterations to converge
    }

    # 5. Base model + GridSearchCV
    lr_model = LogisticRegression()
    grid_search = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_acc = grid_search.best_score_
    joblib.dump(best_model, 'models/best_model.joblib')

    # 6. Evaluate on hold-out test
    preds_test = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, preds_test)
    precision = precision_score(y_test, preds_test, average="weighted")
    recall = recall_score(y_test, preds_test, average="weighted")
    f1 = f1_score(y_test, preds_test, average="weighted")
    conf_mat = confusion_matrix(y_test, preds_test)

    # 7. Log results to MLflow
    mlflow.set_experiment("Logistic Regresssion GridSearchCV Experiment")

    with mlflow.start_run(run_name="LogisticRegression_GridSearch_Updated"):
        mlflow.log_params(best_params)
        mlflow.log_artifact('models/best_model.joblib')
        mlflow.log_metric("best_cv_accuracy", best_cv_acc)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        # Create a input with no column names to match the training format
        sample_input_no_cols = pd.DataFrame([X_test_scaled[0]],
                                            dtype="float64")

        # Log best model passing input with float columns but no feat. names
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_logreg_model",
            input_example=sample_input_no_cols
        )

    # 8. Print results
    print(f"Best Params: {best_params}")
    print(f"Cross-Val Accuracy of Best Model: {best_cv_acc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_mat}")


if __name__ == "__main__":
    tune_logistic_regression()
