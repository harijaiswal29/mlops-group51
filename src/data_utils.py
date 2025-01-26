import pandas as pd
from sklearn.model_selection import train_test_split


def read_data_from_file(file_path="data/iris_data.csv"):
    """
    Reads the Iris dataset from a file and returns X_df and y_df.
    """
    # Read the CSV file
    iris_df = pd.read_csv(file_path)
    # Separate features (X_df) and target (y_df)
    X_df = iris_df.drop(columns=["target"])
    y_df = iris_df["target"]
    return X_df, y_df


def setup_data():
    """
    Reads the Iris dataset from a file, splits it into train/test sets,
    and returns the split data.
    """
    # Read the dataset from the file
    X_df, y_df = read_data_from_file()
    # 2. Split into train/test
    return train_test_split(
        X_df, y_df, test_size=0.3, random_state=42
    )
