import os

from datetime                   import datetime, timedelta
from airflow                    import DAG
from airflow.operators.bash     import BashOperator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_PATH, exist_ok=True)

def notify_failure(context):
    task_instance = context.get('task_instance')
    exception = context.get('exception')
    dag_id = context.get('dag').dag_id
    task_id = task_instance.task_id
    log_url = task_instance.log_url

    msg = (
        f"Задача '{task_id}' в DAG '{dag_id}' упала!\n"
        f"Ошибка: {exception}\n"
        f"Логи: {log_url}\n"
    )
    print(msg)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 3,                  
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),  
    "start_date": datetime(2025, 6, 12),
    "on_failure_callback": notify_failure,     
}

with DAG(
    dag_id="breast_cancer_ml_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["breast_cancer", "ml"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            f'PYTHONPATH={BASE_DIR} python -m etl.download '
            f'>> {os.path.join(LOGS_PATH, "download.log")} 2>&1'
        ),
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=(
            f'PYTHONPATH={BASE_DIR} python -m etl.preprocess '
            f'>> {os.path.join(LOGS_PATH, "preprocess.log")} 2>&1'
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f'PYTHONPATH={BASE_DIR} python -m etl.train '
            f'>> {os.path.join(LOGS_PATH, "train.log")} 2>&1'
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f'PYTHONPATH={BASE_DIR} python -m etl.evaluate '
            f'>> {os.path.join(LOGS_PATH, "evaluate.log")} 2>&1'
        ),
    )

    download_data >> preprocess_data >> train_model >> evaluate_model
