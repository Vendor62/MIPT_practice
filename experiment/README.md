# Hybrid Story Tracking Experiment

This folder contains a compact, public version of the experiment code used to evaluate
hybrid story tracking for Telegram news aggregation.

The experiment compares:
- semantic-only baseline,
- hybrid semantic + temporal + (graph) features,
- DeepSeek Chat LLM baseline.

The public repository does **not** include private Telegram texts, database dumps, credentials,
or the internal production graph pipeline. It contains the evaluation protocol, metric
implementation, configuration template, final aggregate results, and figures.

## Final benchmark

Manual gold dataset:
- 657 posts
- 300 gold stories
- 180 singleton stories
- 120 non-singleton stories

## Key result

Hybrid vs semantic-only on the combined manual gold benchmark:

| Method | Pairwise F1 | ARI | NMI | FMR | FSR |
|---|---:|---:|---:|---:|---:|
| Semantic-only | 0.121 | 0.106 | 0.792 | 0.078 | 0.556 |
| Hybrid | 0.269 | 0.259 | 0.876 | 0.021 | 0.500 |
| DeepSeek Chat | 0.295 | 0.285 | 0.871 | 0.022 | 0.481 |

Hybrid improves over semantic-only by:
- +14.77 pp Pairwise F1
- +15.24 pp ARI
- +8.45 pp NMI

## Reproducibility note

The code is a cleaned public reference implementation. The original experiment used private
PostgreSQL and Neo4j data sources and an internal production graph pipeline. Sensitive data
and credentials are intentionally excluded.

## Quick start (plots only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiment/src/plotting.py
```

