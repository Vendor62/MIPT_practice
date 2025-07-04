# Breast Cancer ML Pipeline

---

## Задача машинного обучения

Цель данного проекта — построить автоматизированный пайплайн для решения **бинарной задачи классификации**: определить, является ли опухоль **злокачественной (malignant)** или **доброкачественной (benign)** на основе медицинских признаков, извлечённых из изображений клеток молочной железы.

Мы используем классический открытый датасет **Breast Cancer Wisconsin Diagnostic**, предоставленный библиотекой `scikit-learn`. Он содержит 30 числовых признаков, таких как радиус, текстура, компактность, симметрия и др.

---

## Декомпозиция задачи

### ML-задача

- **Тип задачи**: Бинарная классификация  
- **Модель**: `LogisticRegression`  
- **Цель**: Прогноз — злокачественная (malignant) или доброкачественная (benign) опухоль  
- **Датасет**: `Breast Cancer Wisconsin Diagnostic Dataset` из `sklearn.datasets`

### Этапы проекта

**1. Инфраструктура и запуск**

- Настроить рабочее окружение (venv / conda)
- Установить и сконфигурировать **Apache Airflow**
- Организовать структуру проекта с разнесением по директориям `data/`, `etl/`, `logs/`, `results/`
- Подключить DAG к Airflow через `dags/pipeline_dag.py`
- Добавить Makefile/ bash-скрипты для автоматизации запуска

**2. Этапы ETL-пайплайна (Airflow DAG)**

**2.1. `download_data`**
- Загрузка данных из `sklearn.datasets`
- Сохранение в `data/raw_data.csv`
- Обработка исключений при загрузке

**2.2. `preprocess_data`**
- Чтение сырых данных
- Стандартизация признаков с помощью `StandardScaler`
- Разделение на train/test
- Сохранение `clean_data/preprocessed_data.csv`
- Добавить логгирование и обработку NaN/ошибок

**2.3. `train_model`**
- Обучение `LogisticRegression` на тренировочной выборке
- Сохранение модели (`results/model.pkl`)
- Логгирование метрик качества
- Обработка ошибок при обучении

**2.4. `evaluate_model`**
- Прогнозирование и расчёт метрик (accuracy, precision, recall, f1-score)
- Сохранение в `results/metrics.json`

**3. Надёжность и контроль**

- Обработка исключений в каждом этапе пайпа
- Настройка логирования через `logger_setup.py` на каждый модуль
- Проверка корректности сохранения файлов (CSV, Pickle)
- Добавление `task_retries`, `retry_delay`, `timeout` в DAG-задачи

**4. MLOps и масштабируемость (опционально)**

- Хранение моделей с версионированием
- Отделение конфигов (`config.py`) от логики
- Вынос метрик в мониторинг
- Интеграция с GitHub и CI/CD пайплайном
---

## Структура проекта
```text
.
├── Makefile
├── README.md
├── clean_data
│   └── preprocessed_data.csv
├── dags
│   ├── __pycache__
│   └── pipeline_dag.py
├── data
│   ├── backup_data.csv
│   └── raw_data.csv
├── etl
│   ├── __init__.py
│   ├── __pycache__
│   ├── config.py
│   ├── download.py
│   ├── evaluate.py
│   ├── logger_setup.py
│   ├── preprocess.py
│   └── train.py
├── logs
│   ├── download.log
│   ├── evaluate.log
│   ├── preprocess.log
│   └── train.log
├── requirements.txt
└── results
    ├── metrics.json
    └── model.pkl
```
---

## Структура пайплайна

Пайплайн реализован с использованием **Apache Airflow** и включает в себя следующие этапы:

1. **Загрузка данных (`download_data`)**  
   Получение датасета из `sklearn.datasets` и сохранение в формате CSV.

2. **Предобработка данных (`preprocess_data`)**  
   - Стандартизация признаков с помощью `StandardScaler`
   - Разделение на обучающую и тестовую выборки
   - Сохранение предобработанных данных в формате `.сsv`

3. **Обучение модели (`train_model`)**  
   Обучение логистической регрессии (`LogisticRegression`) на обучающих данных. Сохранение модели.

4. **Оценка модели (`evaluate_model`)**  
   Расчёт метрик качества модели: Accuracy, Precision, Recall, F1-score.

Эти задачи объединены в DAG, где каждая последующая зависит от успешного завершения предыдущей.

---

## Схема пайплайна
```text
+----------------+       +-------------------+       +---------------+       +------------------+
|  Download Data | --->  |  Preprocess Data  | --->  |  Train Model  | --->  |  Evaluate Model  |
+----------------+       +-------------------+       +---------------+       +------------------+
```

## Структура и назначение скриптов

Проект состоит из набора Python-скриптов, каждый из которых отвечает за отдельный этап обработки данных и обучения модели. Скрипты находятся в папке `etl/` и выполняются поочерёдно в рамках DAG в Apache Airflow.

| Скрипт                  | Назначение                                                                 |
|-------------------------|----------------------------------------------------------------------------|
| `download.py`           | Загружает датасет Breast Cancer из `sklearn.datasets` и сохраняет его в CSV. |
| `preprocess.py`         | Загружает сырые данные, масштабирует числовые признаки, разбивает на train/test. |
| `train.py`              | Обучает модель логистической регрессии на подготовленных данных и сохраняет `model.pkl`. |
| `evaluate.py`           | Загружает модель и тестовые данные, рассчитывает метрики качества и сохраняет отчёт. |
| `config.py`             | Хранит пути к файлам и общие параметры конфигурации пайплайна. |
| `logger_setup.py`       | Настройки логгера. |
---

### Взаимосвязь скриптов

Поток данных между скриптами организован следующим образом:

1. `download.py` сохраняет исходные данные в `data/raw_data.csv`.
2. `preprocess.py` использует этот файл, обрабатывает данные и сохраняет `clean_data/train.csv`, `clean_data/test.csv`.
3. `train.py` загружает `train.csv`, обучает модель и сохраняет её в `results/model.pkl`.
4. `evaluate.py` загружает модель и `test.csv`, рассчитывает метрики и сохраняет отчёт в `results/metrics.json`.

## DAG и задачи

### Название DAG

DAG называется: `breast_cancer_ml_pipeline`

Он определён в файле: `dags/pipeline_dag.py`

---

### Зависимости между задачами

DAG состоит из четырёх последовательных задач, каждая из которых зависит от предыдущей:

1. `download_data` — загружает исходные данные;
2. `preprocess_data` — выполняет очистку и масштабирование данных;
3. `train_model` — обучает модель логистической регрессии;
4. `evaluate_model` — рассчитывает метрики и сохраняет отчёт.

---

### Запуск DAG вручную

Можно протестировать выполнение каждой отдельной задачи вручную с помощью CLI команды `airflow tasks test`. Пример:

```bash
# Протестировать задачу загрузки данных:
airflow tasks test breast_cancer_ml_pipeline download_data 2025-06-13

# Протестировать preprocessing:
airflow tasks test breast_cancer_ml_pipeline preprocess_data 2025-06-13

# Обучение модели:
airflow tasks test breast_cancer_ml_pipeline train_model 2025-06-13

# Оценка модели:
airflow tasks test breast_cancer_ml_pipeline evaluate_model 2025-06-13
```

## Интеграции и хранение данных

### Интеграция

Данный проект **не использует внешние источники данных или API**. Весь процесс построен на **локальной обработке датасета** `Breast Cancer Wisconsin Diagnostic`, загружаемого напрямую из библиотеки `sklearn.datasets`. Таким образом, нет необходимости в ключах доступа, конфигурациях подключения или внешней авторизации.

### Хранение ключей

Проект **не требует использования ключей доступа или переменных окружения**. Все операции происходят локально, без подключения к облачным сервисам или API.

---

## Структура хранения данных

Все промежуточные и финальные результаты работы пайплайна сохраняются на локальный диск в папку `/results`, которая создаётся автоматически при запуске соответствующего этапа.

### Структура папки:
```text
/results/
├── model.pkl        — сериализованная модель LogisticRegression
├── metrics.json     — метрики качества модели (accuracy, precision, recall, f1)
```

### Пример кода для получения данных:
```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
```

### Пример содержимого `metrics.json`:
```json
    "accuracy": 0.9736842105263158,
    "precision": 0.9722222222222222,
    "recall": 0.9859154929577465,
    "f1_score": 0.9790209790209791
```

## Где может упасть процесс?

- Загрузка данных из sklearn (`load_breast_cancer`)
- Проверка структуры данных (наличие нужных столбцов, пропуски)
- Сохранение данных на диск
- Предобработка: загрузка файла, проверка пропусков, маппинг целевого признака, масштабирование признаков
- Обучение модели: загрузка данных, проверка пропусков, разбиение на выборки, обучение LogisticRegression, сохранение модели

**Механизмы обработки ошибок:**

- Везде используется `try-except` с логированием ошибок
- При ошибке загрузки из sklearn — попытка подгрузить резервный файл
- При отсутствии файлов или критических ошибках — логирование и аварийное завершение (`sys.exit` или `raise`)
- Все ключевые этапы логируются в отдельные файлы для удобства диагностики

---

## Какие исключения могут возникнуть?

- `ValueError` при отсутствии ожидаемых колонок или при наличии пропусков
- Ошибки чтения/записи файлов (`FileNotFoundError`, `IOError`)
- Ошибки маппинга целевого признака (если значения некорректны)
- Исключения библиотеки sklearn (например, при обучении модели)
- Ошибки при отсутствии резервных данных
- Общие исключения (`Exception`) с последующим логированием и завершением работы

---

## Что произойдет при потере соединения с источником данных?

В текущем проекте источник — локальный sklearn-дataset, который не требует сети. Если бы источник был внешний (API, база), процесс мог бы упасть.

**Как реализовано сейчас:**

- При ошибке загрузки из sklearn происходит логирование предупреждения
- Затем запускается попытка загрузить резервный файл `backup_data.csv`
- Если резервный файл отсутствует или содержит пропуски — процесс завершится с ошибкой и логированием

---

## Что будет, если источник отдает невалидные данные?

Невалидные данные — отсутствие нужных столбцов, пропуски, некорректные значения целевой колонки.

**Поведение системы:**

- Проверка наличия всех обязательных колонок с выбросом ошибки (`ValueError`)
- Проверка пропусков с логированием предупреждений или ошибкой (в зависимости от этапа)
- Проверка корректности маппинга целевого признака — при ошибке процесс завершается
- При загрузке резервных данных происходит повторная валидация
- При обнаружении критических ошибок процесс аварийно завершается

---

## Что произойдет, если модель не обучается или выдает ошибку?

- Ошибка при загрузке данных (отсутствие целевой колонки, пропуски) вызывает исключение
- Ошибки во время обучения (несоответствие данных, параметры) логируются
- Исключения обучения пробрасываются выше для обработки вызывающим кодом
- При ошибке процесс останавливается, предотвращая использование невалидной модели
- Все этапы обучения тщательно логируются для диагностики

---

## Использование Airflow: retries, timeout, failure callback

В DAG `breast_cancer_ml_pipeline` реализованы важные механизмы устойчивости:

---

### Повторы задач (Retries)

- `retries=3`: при неудачном выполнении задача будет автоматически повторена до 3 раз.
- `retry_delay=timedelta(minutes=5)`: между попытками выдерживается пауза 5 минут.

Это особенно полезно при временных сбоях — например, загрузка файла может упасть из-за блокировки, но через 5 минут уже пройти успешно.

---

### Ограничение времени выполнения задач (Timeout)

- `execution_timeout=timedelta(minutes=15)`: если задача не завершится за 15 минут — она будет принудительно остановлена и помечена как failed.

Это защищает пайплайн от зависаний, бесконечных циклов или неэффективного кода.

---

### Callback при падении задачи

```python
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
```
При падении задачи вызывается функция `notify_failure`, которая:
- Получает информацию о задаче, DAG'е и исключении
- Формирует текст уведомления
- Потенциально может отправлять сообщение в Telegram, Slack или email

Текущая реализация выводит сообщение в консоль. Можно легко расширить под Slack API, SMTP или телеграм-бота.

## Архитектура: изолированные модули

- Каждый этап пайплайна оформлен как отдельный Python-модуль внутри папки etl/
- Запуск выполняется через `python -m etl.module_name`, что:
   - Обеспечивает изоляцию логики — каждый этап независим
   - Упрощает отладку и повторное использование кода
   - Делает модули переиспользуемыми и удобными для unit-тестов

Также используется `PYTHONPATH={BASE_DIR}` — это гарантирует, что Python найдёт модули, независимо от текущей рабочей директории.

## Идеи для развития ML-проекта на Airflow

1. **Интеграция с Telegram для алертов**
   - Уведомление о падениях задач, новых моделях, метриках — прямо в чат.

2. **Docker-контейнеризация**
   - Позволит разворачивать проект в любом окружении, подготовит инфраструктуру для CI/CD и деплоя.

3. **Добавить сравнение нескольких моделей**
   - Например, `RandomForest`, `XGBoost`, `SVM`, `CatBoost`.

4. **Хранение моделей и метрик в базе данных или S3**
   - Сейчас результаты сохраняются локально — можно уйти в централизованное хранилище.

5. **DashBoard мониторинга (например, через Grafana + Prometheus)**
   - Можно отслеживать историю запуска DAG'ов, частоту падений и длительность.

## Скриншот работающего DAG
![](https://raw.githubusercontent.com/Vendor62/MIPT_practice/refs/heads/main/exams/de_2/media/01.png)<br><br>
