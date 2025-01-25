import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd


# Load dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Experiment with multiple models
model = LogisticRegression(max_iter=300, solver='lbfgs')

with mlflow.start_run():
    model_name = f"LogisticRegression_simple_model"

    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("max_iter", model.max_iter)
    mlflow.log_param("solver", model.solver)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions and log metrics
    predictions = model.predict(X_test)
        # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    conf_matrix = confusion_matrix(y_test, predictions)




    #accuracy = accuracy_score(y_test, predictions)
        # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Define an input example
    input_example = pd.DataFrame([X_test[0]], columns=data.feature_names)
    print(input_example)

    # Log the model with an input example
    mlflow.sklearn.log_model(
        model,
        model_name,
        input_example=input_example
    )

    # Print result
    print(f"Run Accuracy: {accuracy:.4f}")
    print(f"Metrics logged: Accuracy:{accuracy:.4f} , Precision{precision:.4f}, Recall{recall:.4f}, F1-score {f1:.4f}")
