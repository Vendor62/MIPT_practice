import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import argparse
import pandas as pd

from sklearn.metrics            import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection    import train_test_split
from etl.config                 import CONFIG

def evaluate(data_path, model_path, metrics_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[CONFIG["target_column"]])
    y = df[CONFIG["target_column"]]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CONFIG["preprocessed_data_path"])
    parser.add_argument("--model-path", type=str, default=CONFIG["model_path"])
    parser.add_argument("--metrics-path", type=str, default=CONFIG["metrics_path"])
    args = parser.parse_args()

    evaluate(args.data_path, args.model_path, args.metrics_path)