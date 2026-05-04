# Эксперимент: Hybrid Story Tracking

Эта папка содержит компактную публичную версию кода эксперимента для оценки гибридного
story tracking в задаче агрегации новостей Telegram.

В эксперименте сравниваются:
- semantic-only baseline,
- гибридный скоринг (semantic + temporal + (graph) features),
- LLM baseline на базе DeepSeek Chat.

Публичный репозиторий **не** содержит приватные тексты Telegram, дампы БД, креды и внутренний
production graph pipeline. Здесь оставлены протокол оценки, реализация метрик, шаблон конфига,
итоговые агрегированные результаты и скрипт построения графиков.

## Итоговый бенчмарк

Manual gold датасет:
- 657 постов
- 300 gold stories
- 180 singleton stories
- 120 non-singleton stories

## Ключевой результат

Hybrid vs semantic-only на combined manual gold benchmark:

| Метод | Pairwise F1 | ARI | NMI | FMR | FSR |
|---|---:|---:|---:|---:|---:|
| Semantic-only | 0.121 | 0.106 | 0.792 | 0.078 | 0.556 |
| Hybrid | 0.269 | 0.259 | 0.876 | 0.021 | 0.500 |
| DeepSeek Chat | 0.295 | 0.285 | 0.871 | 0.022 | 0.481 |

Hybrid улучшает semantic-only на:
- +14.77 п.п. Pairwise F1
- +15.24 п.п. ARI
- +8.45 п.п. NMI

## Примечание про воспроизводимость

Код — это очищенная публичная reference-реализация. Оригинальный эксперимент использовал приватные
источники PostgreSQL и Neo4j, а также внутренний production graph pipeline. Чувствительные данные
и креды намеренно исключены.

## Быстрый старт (только графики)

Файлы `.png` в репозиторий не включены; их можно локально сгенерировать:

```bash
cd experiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/plotting.py
```

