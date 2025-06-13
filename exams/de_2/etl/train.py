import os
import sys
import joblib
import logging
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from etl.config                 import CONFIG, LOGS_PATH
from etl.logger_setup           import setup_logger
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import train_test_split

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH, exist_ok=True)

log_file = os.path.join(LOGS_PATH, "train.log")

logger = setup_logger(
    name="train_logger",
    log_file=log_file,
    to_stdout=True, 
    to_file=True,
    level=logging.INFO
)

logger.info("Начинаем этап обучения модели")


def train(data_path, model_path):
    try:
        logger.info("Начало обучения модели")
        df = pd.read_csv(data_path)
        logger.info(f"Загружено {df.shape[0]} строк и {df.shape[1]} столбцов из {data_path}")

        if CONFIG["target_column"] not in df.columns:
            raise ValueError(f"Целевая переменная '{CONFIG['target_column']}' не найдена в данных.")

        X = df.drop(columns=[CONFIG["target_column"]])
        y = df[CONFIG["target_column"]]

        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            raise ValueError("В данных есть пропущенные значения. Очисти их на этапе предобработки.")

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Размер обучающей выборки: {X_train.shape}")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        logger.info("Обучение модели завершено")

        joblib.dump(model, model_path)
        logger.info(f"Модель сохранена в: {model_path}")

    except Exception as e:
        logger.exception(f"Ошибка при обучении модели: {e}")
        raise 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CONFIG["preprocessed_data_path"])
    parser.add_argument("--model-path", type=str, default=CONFIG["model_path"])
    args = parser.parse_args()

    train(args.data_path, args.model_path)
