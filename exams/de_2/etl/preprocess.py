import os
import sys
import logging
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from etl.config                 import CONFIG, LOGS_PATH
from etl.logger_setup           import setup_logger
from sklearn.preprocessing      import StandardScaler


if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH, exist_ok=True)

log_file = os.path.join(LOGS_PATH, "preprocess.log")

logger = setup_logger(
    name="preprocess_logger",
    log_file=log_file,
    to_stdout=False,  
    to_file=True,
    level=logging.INFO
)

logger.info("Запуск этапа предобработки данных")


def preprocess(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            logger.error(f"Входной файл не найден: {input_path}")
            sys.exit(1)

        df = pd.read_csv(input_path)
        logger.info(f"Загружено {df.shape[0]} строк и {df.shape[1]} столбцов из {input_path}")

        if df.isnull().any().any():
            logger.warning("Обнаружены пропущенные значения в данных.")
        
        if CONFIG["target_column"] not in df.columns:
            logger.error(f"Целевая колонка '{CONFIG['target_column']}' не найдена в данных.")
            sys.exit(1)

        df = df.drop(columns=["target"], errors="ignore")
        df[CONFIG["target_column"]] = df[CONFIG["target_column"]].map({"B": 0, "M": 1})

        if df[CONFIG["target_column"]].isnull().any():
            logger.error(f"Ошибка при маппинге целевой переменной. Проверь значения в колонке '{CONFIG['target_column']}'.")
            sys.exit(1)

        X = df.drop(columns=[CONFIG["target_column"]])
        y = df[CONFIG["target_column"]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if X_scaled.shape != X.shape:
            logger.warning("Размерность после масштабирования не совпадает с исходной.")

        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        df_scaled[CONFIG["target_column"]] = y.values
        df_scaled.to_csv(output_path, index=False)
        logger.info(f"Преобразованные данные успешно сохранены в {output_path}")

    except Exception as e:
        logger.exception(f"Произошла ошибка при предобработке: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=CONFIG["raw_data_path"])
    parser.add_argument("--output-path", type=str, default=CONFIG["preprocessed_data_path"])
    args = parser.parse_args()

    preprocess(args.input_path, args.output_path)
