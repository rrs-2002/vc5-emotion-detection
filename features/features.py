import yaml
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import spmatrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train and test data from CSV files.
    """
    logging.info(f"Loading train data from {train_path}")
    train_data = pd.read_csv(train_path).dropna(subset=["content"])
    logging.info(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path).dropna(subset=["content"])
    return train_data, test_data

def extract_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from DataFrame.
    """
    X = df["content"].to_numpy()
    y = df["sentiment"].to_numpy()
    return X, y

def vectorize_text(
    X_train: np.ndarray, 
    X_test: np.ndarray
) -> Tuple[spmatrix, spmatrix, CountVectorizer]:
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    max_features = params["feature_engg"]["max_features"]
    """
    Fit CountVectorizer on training data and transform both train and test data.
    """
    logging.info("Fitting CountVectorizer on training data")
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow: spmatrix = vectorizer.fit_transform(X_train)
    X_test_bow: spmatrix = vectorizer.transform(X_test)
    return X_train_bow, X_test_bow, vectorizer

def save_feature_data(
    X_bow: np.ndarray, 
    y: np.ndarray, 
    output_path: str
) -> None:
    """
    Save feature vectors and labels to a CSV file.
    """
    # Convert to dense array if X_bow is a sparse matrix
    if hasattr(X_bow, "toarray"):
        data = X_bow.toarray()
    else:
        data = X_bow
    df = pd.DataFrame(data)
    df["label"] = y
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Saving features to {output_path}")
    df.to_csv(output_path, index=False)

def main() -> None:
    """
    Main function to orchestrate feature extraction and saving.
    """
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    train_output = "data/interim/train_bow.csv"
    test_output = "data/interim/test_bow.csv"

    train_data, test_data = load_data(train_path, test_path)
    X_train, y_train = extract_features_and_labels(train_data)
    X_test, y_test = extract_features_and_labels(test_data)

    X_train_bow, X_test_bow, _ = vectorize_text(X_train, X_test)

    save_feature_data(X_train_bow, y_train, train_output)
    save_feature_data(X_test_bow, y_test, test_output)

if __name__ == "__main__":
    main()