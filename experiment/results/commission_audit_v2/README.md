# Commission audit v2 — published metrics

Официальные агрегированные метрики для **combined_manual_gold_v2** (evaluator: `pilot_eval_v2_assignment_lib`, пороги frozen на dev).

| Файл | Содержание |
|------|------------|
| `unified_metrics_v2.csv` | Основные метрики по method × split (semantic, hybrid saved, deepseek saved, …). |
| `bootstrap_confidence_intervals_unified_v2.csv` | Stratified bootstrap 95% CI (gold-story), deltas hybrid−semantic / hybrid−deepseek. |
| `subgroup_metrics_unified_v2.csv` | Subgroup breakdown (story size, channel, …). |
| `subgroup_metrics_diff_vs_v1.csv` | Diff subgroup metrics vs legacy v1 export. |
| `hypothesis_status_v2.csv` | Статус гипотез H1/H2/H3 для текста ВКР. |

**Test reference (hybrid saved):** `hybrid_v4_1_soft_entity_penalty` pairwise F1 ≈ 0.269.  
**Note:** hybrid replay formula on current pairwise features does not reproduce saved predictions; use saved preds for commission numbers.

Сгенерировано: `recompute_combined_manual_gold_v2_metrics_unified.py` (2026-05-21 server run).
