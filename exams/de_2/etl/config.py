import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = {
    "raw_data_path": os.path.join(DATA_DIR, "raw_data.csv"),
    "preprocessed_data_path": os.path.join(CLEAN_DATA_DIR, "preprocessed_data.csv"),
    "model_path": os.path.join(RESULTS_DIR, "model.pkl"),
    "metrics_path": os.path.join(RESULTS_DIR, "metrics.json"),
    "target_column": "diagnosis"
}