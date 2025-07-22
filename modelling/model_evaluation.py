import os
import logging
import json
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_model(model_path: str) -> BaseEstimator:
    """
    Load a trained model from a pickle file.
    """
    logging.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_test_data(test_data_path: str) -> pd.DataFrame:
    """
    Load test data from a CSV file.
    """
    logging.info(f"Loading test data from {test_data_path}")
    return pd.read_csv(test_data_path)

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate the model and return metrics.
    """
    logging.info("Evaluating model on test data")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Saving metrics to {output_path}")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

def main() -> None:
    """
    Main function to orchestrate model evaluation.
    """
    model_path = "models/random_forest_model.pkl"
    test_data_path = "data/interim/test_bow.csv"
    metrics_output_path = "reports/metrics.json"

    model = load_model(model_path)
    test_data = load_test_data(test_data_path)
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values

    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_output_path)

if __name__ == "__main__":
    main()