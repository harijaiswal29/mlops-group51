import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Experiment with multiple models
model = LogisticRegression(max_iter=300)

with mlflow.start_run():
    model_name = f"LogisticRegression_simple_model"

     # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("max_iter", model.max_iter)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions and log metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, model_name)

    # Print result
    print(f"Run Accuracy: {accuracy:.4f}")
