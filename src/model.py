import numpy as np
from sklearn.linear_model import LinearRegression

def train_model():
    # Generate some dummy data
    X = np.array([[i] for i in range(10)])  # 10 samples, single feature
    y = np.array([2*i + 1 for i in range(10)])  # y = 2x + 1

    # Create and train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    model = train_model()
    print("Model trained successfully!")
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")
