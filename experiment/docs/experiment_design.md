# Дизайн эксперимента (публичная версия)

## Гипотеза

Гибридный story tracking, который объединяет семантическую близость с временными и графовыми сигналами,
снижает число ложных склеек (false merges) и улучшает качество кластеризации по сравнению с semantic-only baseline.

## Сравниваемые методы

- **semantic_only**: online assignment только по embedding similarity.
- **hybrid_v4_1_soft_entity_penalty**: hybrid scoring (semantic + temporal + entity/event/graph signals) с anti-merge bias.
- **deepseek_chat_llm**: LLM baseline, который выбирает attach vs new story по retrieval shortlist кандидатов-историй.

## Датасет

Manual gold benchmark **combined_manual_gold_v2**:
- 657 постов
- 300 gold stories
- 180 singleton stories
- 120 non-singleton stories

Gold получен из ручной разметки. Публичный репозиторий не содержит размеченных текстов.

## Сплиты

Хронологический сплит по времени публикации:
- 30% train
- 30% dev
- 40% test

## Предотвращение leakage

- Gold labels используются **только** для оценки и для dev-тюнинга порогов.
- Решение online assignment никогда не видит gold.
- В LLM prompt никогда не добавляются gold labels.

## Метрики

Основные:
- pairwise F1
- ARI
- NMI
- false merge rate (FMR)
- false split rate (FSR)

Дополнительные:
- post assignment accuracy (только справочно; может вводить в заблуждение для кластеризации)

## Протокол LLM baseline (DeepSeek)

- Online replay в хронологическом порядке.
- Candidate retrieval: top-K (5–8) candidate stories по semantic similarity к уже увиденным постам.
- Prompt: текст текущего поста + компактные представители candidate stories.
- Выход: строгий JSON `{decision, selected_story_id|null, confidence, short_reason}`.

