# Story Tracking experiment tools (audit v2 sync)

Скрипты для воспроизведения и аудита **combined_manual_gold_v2** (commission audit v2).  
Запуск предполагает локальный experiment root с `sync_out/data/prepared/` и evaluator `pilot_eval_v2_assignment_lib.py` на `PYTHONPATH` (не входит в этот sync).

| Скрипт | Назначение |
|--------|------------|
| `recompute_combined_manual_gold_v2_metrics_unified.py` | Единый пересчёт unified metrics, bootstrap CI, subgroups, hypothesis status из saved predictions (без LLM). |
| `trace_hybrid_v4_1_official_saved_provenance_v1.py` | Forensic audit: почему replay hybrid v4.1 ≠ official saved predictions. |
| `search_combined_v2_generation_context_v1.py` | Поиск provenance генерации combined v2 bundle (скрипты, manifest, transcript hints). |
| `audit_v3d_graph_feature_coverage.py` | Покрытие Neo4j/graph/entity features на v3d и combined v2 pairwise. |
| `rerun_v3d_eval_with_neo4j_graph_features_v2.py` | Пересбор pairwise v3d с Neo4j + eval semantic/hybrid (diagnostic). |
| `develop_and_evaluate_graph_heavy_developmental_v3d_v1.py` | Exploratory graph-heavy variants на strict developmental v3d (dev/test). |
| `build_manual_gold_expansion_v3d_strict_developmental_news.py` | Сбор labeling set v3d (12-channel whitelist, без overlap с v2). |
| `validate_manual_gold_expansion_v3d_strict_developmental_labeling_set.py` | Валидация v3d labeling CSV до разметки. |
| `audit_graph_advantage_extension_preview_v1_labeled.py` | Audit размеченного graph-advantage preview + decision gate. |
| `build_same_source_graph_advantage_extension_preview_v1.py` | Preview extension для graph-advantage (same protocol). |
| `build_manual_gold_expansion_v3c_developmental_stories.py` | Сбор v3c developmental expansion (exploratory, не main line). |
| `audit_labeled_v3c_whitelist_and_prepare_clean_eval.py` | Clean v3c gold + diagnostic eval prep. |

**Не включены в sync (отсутствуют на сервере):** `run_deepseek_llm_baseline_on_combined_manual_gold_v2.py`, `evaluate_expansion_300_and_combined_manual_gold_v2.py`, `build_large_graph_features_v3.py`.

**Гигиена:** не коммитить `.env`, `sync_in/`, fulltext, raw embeddings, prod credentials.
