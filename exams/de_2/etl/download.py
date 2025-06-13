import os
import sys
import logging 
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from etl.config         import CONFIG
from etl.logger_setup   import setup_logger

LOGS_PATH = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_PATH, exist_ok=True)

log_file = os.path.join(LOGS_PATH, "download.log")

logger = setup_logger(
    name="download_logger",
    log_file=log_file,
    to_stdout=False,  
    to_file=True,
    level=logging.INFO
)

def download_and_save(output_path):
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame

        expected_columns = set(data.feature_names.tolist() + ["target"])
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют ожидаемые столбцы: {missing_columns}")

        if df.isnull().any().any():
            raise ValueError("В загруженных данных обнаружены пропущенные значения.")

        df[CONFIG["target_column"]] = df["target"].map({0: "B", 1: "M"})
        if df[CONFIG["target_column"]].isnull().any():
            raise ValueError("Ошибка при маппинге целевого признака. Проверь значения в колонке 'target'.")

        df.to_csv(output_path, index=False)
        logger.info(f"Данные успешно загружены из sklearn и сохранены в {output_path}")

    except Exception as e:
        logger.warning(f"Не удалось загрузить датасет из sklearn: {e}")
        backup_path = os.path.join(os.path.dirname(output_path), "backup_data.csv")

        if os.path.exists(backup_path):
            logger.info(f"Используется резервная копия датасета из {backup_path}")
            df = pd.read_csv(backup_path)

            if df.isnull().any().any():
                logger.error("Резервный датасет содержит пропущенные значения.")
                sys.exit(1)

            df.to_csv(output_path, index=False)
            logger.info(f"Резервные данные скопированы в {output_path}")
        else:
            logger.error(f"Резервный датасет не найден по пути {backup_path}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default=CONFIG["raw_data_path"],
        help="Путь для сохранения исходного датасета"
    )
    args = parser.parse_args()

    download_and_save(args.output_path)
