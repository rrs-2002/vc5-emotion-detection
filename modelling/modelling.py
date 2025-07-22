import os
import logging
import pickle
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_train_data(path: str) -> pd.DataFrame:
    """
    Load training data from a CSV file.
    """
    logging.info(f"Loading training data from {path}")
    return pd.read_csv(path)

def split_features_labels(df: pd.DataFrame, label_col: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
    """
    Split DataFrame into features and labels.
    """
    logging.info(f"Splitting features and label column '{label_col}'")
    X = df.drop(columns=[label_col]).values
    y = df[label_col].to_numpy()
    return X, y

def train_model(X: np.ndarray, y: np.ndarray) -> ClassifierMixin:
    """
    Train a RandomForestClassifier model.
    """
    logging.info("Training RandomForestClassifier model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model: ClassifierMixin, path: str) -> None:
    """
    Save the trained model to a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.info(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)

def main() -> None:
    """
    Main function to orchestrate model training and saving.
    """
    train_data_path = "data/interim/train_bow.csv"
    model_output_path = "models/random_forest_model.pkl"

    train_data = load_train_data(train_data_path)
    X_train, y_train = split_features_labels(train_data)
    model = train_model(X_train, y_train)
    save_model(model, model_output_path)

if __name__ == "__main__":
    main()