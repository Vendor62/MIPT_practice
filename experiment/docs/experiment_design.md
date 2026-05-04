# Experiment design (public)

## Hypothesis

Hybrid story tracking that combines semantic similarity with temporal and graph-derived signals
reduces false merges and improves clustering quality over a semantic-only baseline.

## Compared methods

- **semantic_only**: online assignment using only embedding similarity.
- **hybrid_v4_1_soft_entity_penalty**: hybrid scoring (semantic + temporal + entity/event/graph signals) with anti-merge bias.
- **deepseek_chat_llm**: LLM baseline that decides attach vs new story using a retrieval shortlist of candidate stories.

## Dataset

Manual gold benchmark **combined_manual_gold_v2**:
- 657 posts
- 300 gold stories
- 180 singleton stories
- 120 non-singleton stories

Gold is created by human labeling. The public repo does not include the labeled texts.

## Splits

Chronological split by post time:
- 30% train
- 30% dev
- 40% test

## Leakage prevention

- Gold labels are used **only** for evaluation and dev-time tuning of thresholds.
- The online assignment decision never sees gold.
- The LLM prompt never includes gold labels.

## Metrics

Primary:
- pairwise F1
- ARI
- NMI
- false merge rate (FMR)
- false split rate (FSR)

Secondary:
- post assignment accuracy (informational only; can be misleading for clustering)

## LLM baseline protocol (DeepSeek)

- Online replay in chronological order.
- Candidate retrieval: top-K (5–8) candidate stories by semantic similarity to already-seen posts.
- Prompt includes: current post text + compact representatives of candidate stories.
- Output: strict JSON `{decision, selected_story_id|null, confidence, short_reason}`.

