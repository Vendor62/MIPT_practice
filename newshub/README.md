# NewsHub

Система мониторинга и агрегации новостей с NLP-обработкой
(NER + Relation Extraction) и графовым хранением историй в Neo4j.
Разработана в рамках выпускной работы МФТИ (2026).

---

## Архитектура

Сервисы запускаются через Docker Compose.

| Сервис | Назначение |
|--------|------------|
| `newshub-api` | FastAPI REST/WebSocket API |
| `newshub-bot` | Telegram-бот (aiogram) — дайджесты и подписки |
| `newshub-worker` | Celery worker — парсинг, обогащение, нотификации |
| `newshub-beat` | Celery beat — периодические задачи |
| `newshub-ie` | IE worker — NER/RE через GLiNER + mREBEL |
| `newshub-embeddings-worker` | Celery worker эмбеддингов |
| `newshub-embeddings-beat` | Планировщик эмбеддингов |
| `newshub-embeddings-scheduler` | Координатор очередей эмбеддингов |
| `newshub-storyline` | Сборка графа историй в Neo4j |
| `postgres` | PostgreSQL — основная реляционная БД |
| `redis` | Redis — брокер Celery и кэш |
| `neo4j` | Neo4j — граф сущностей и историй |

### Стек

- Python 3.11, FastAPI, aiogram 3, SQLAlchemy 2, Alembic
- Celery (брокер — Redis)
- PostgreSQL (статьи, источники, пользователи, подписки, заказы)
- Neo4j (граф сущностей и storylines)
- GLiNER (zero-shot NER), mREBEL (multilingual RE)
- Docker Compose

### IE Pipeline

`app/ie_pipeline.py` — обёртка над GLiNER (NER) и mREBEL (RE):

1. Текст статьи → GLiNER → именованные сущности (PER, ORG, LOC, …)
2. Текст + сущности → mREBEL → триплеты (субъект, отношение, объект)
3. Триплеты → Neo4j: сущности как узлы, отношения как рёбра
4. Связные компоненты графа → storylines

Модели работают на CPU (`IE_TORCH_DEVICE=cpu`).

---

## Запуск

```bash
cp .env.example .env
# Заполни .env: пароли БД, Telegram Bot Token и т.д.
docker compose up -d
```

Требования: Docker + Docker Compose v2, ~8 GB RAM.

---

## Структура

```
newshub/
├── app/                # Python-пакет приложения
│   ├── main.py         # FastAPI
│   ├── bot.py          # Telegram-бот
│   ├── tasks.py        # Celery-задачи
│   ├── models.py       # SQLAlchemy ORM
│   ├── ie_pipeline.py  # NER/RE pipeline
│   ├── storyline_*.py  # Сторилайнинг
│   ├── neo4j_*.py      # Работа с Neo4j
│   └── i18n.py         # Локализация ru/en
├── alembic/            # Миграции БД
├── compose.yaml        # Docker Compose
├── pyproject.toml      # Зависимости (Poetry)
├── .env.example
├── AUTHORSHIP.md       # Авторство по файлам
├── LICENSE             # Business Source License 1.1
└── LICENSE.ru.md       # Перевод лицензии
```

Метки `TODO[showcase]` в коде — намеренные заглушки на месте модулей,
не вошедших в публичную витрину.

---

## Лицензия

Репозиторий распространяется под **Business Source License 1.1** —
см. [LICENSE](LICENSE) и [LICENSE.ru.md](LICENSE.ru.md).

Кратко: код открыт для изучения и **некоммерческого** использования;
коммерческое использование требует письменного разрешения автора;
через 4 года после публикации код переходит под **Apache 2.0**.

© 2026 Поликарпов Дмитрий Александрович.
