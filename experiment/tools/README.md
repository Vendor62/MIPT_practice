# `tools/` — служебные сценарии для проверки результатов

Сценарии для повторного расчёта и проверки итоговых чисел по сводному
размеченному набору **combined_manual_gold_v2** (commission audit v2).

Запуск предполагает локальный корень эксперимента с подготовленными данными
в `sync_out/data/prepared/` и наличие библиотеки оценки
`pilot_eval_v2_assignment_lib.py` в `PYTHONPATH`. Эта библиотека в публичный
репозиторий не входит.

## Что какой сценарий делает

| Сценарий | Назначение |
|----------|------------|
| `recompute_combined_manual_gold_v2_metrics_unified.py` | Сквозной пересчёт сводных метрик, доверительных интервалов по бутстрэпу, метрик по подгруппам и статуса гипотез — из сохранённых предсказаний, без обращения к LLM. |
| `trace_hybrid_v4_1_official_saved_provenance_v1.py` | Разбор причин, почему повтор формулы гибрида v4.1 не совпадает с официально сохранёнными предсказаниями (forensic-проверка, только чтение). |
| `search_combined_v2_generation_context_v1.py` | Поиск следов того, как именно был собран сводный набор v2: сценарии, манифесты, фрагменты транскриптов. |
| `audit_v3d_graph_feature_coverage.py` | Проверка покрытия графовых признаков (Neo4j / соседи / типы связей) на v3d и сводных парных признаках v2. |
| `rerun_v3d_eval_with_neo4j_graph_features_v2.py` | Пересборка парных признаков v3d с подмешиванием Neo4j и диагностическая оценка semantic/hybrid. |
| `develop_and_evaluate_graph_heavy_developmental_v3d_v1.py` | Поисковые варианты «графово-нагруженного» гибрида на строгом v3d (dev/test). |
| `build_manual_gold_expansion_v3d_strict_developmental_news.py` | Сбор набора для разметки v3d: только новостные «развивающиеся» сюжеты, белый список из 12 каналов, без пересечения с v2. |
| `validate_manual_gold_expansion_v3d_strict_developmental_labeling_set.py` | Проверка CSV для разметки v3d перед передачей разметчикам (форматы, диапазоны дат, запрещённые поля). |
| `audit_graph_advantage_extension_preview_v1_labeled.py` | Проверка размеченного предварительного набора по «графовому преимуществу» и решение, идти ли в полное расширение. |
| `build_same_source_graph_advantage_extension_preview_v1.py` | Сбор того же предварительного набора по «графовому преимуществу» по тому же протоколу, что и сводный v2. |
| `build_manual_gold_expansion_v3c_developmental_stories.py` | Сбор поискового расширения v3c («развивающиеся» сюжеты), не основная линия. |
| `audit_labeled_v3c_whitelist_and_prepare_clean_eval.py` | Чистка v3c gold и подготовка диагностической оценки. |

## Что не входит в синхронизацию

В публичной выгрузке отсутствуют сценарии, которые требуют приватных ключей или
тяжёлых внутренних артефактов:
- `run_deepseek_llm_baseline_on_combined_manual_gold_v2.py`;
- `evaluate_expansion_300_and_combined_manual_gold_v2.py`;
- `build_large_graph_features_v3.py`.

## Гигиена

В этой папке не должны появляться `.env`, входные приватные данные (`sync_in/`),
полные тексты постов, исходные эмбеддинги и продовые ключи доступа.
