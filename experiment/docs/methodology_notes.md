# Methodology notes

## Why not optimize raw accuracy?

For story clustering, raw post-level accuracy can look high even when the model makes
structurally bad errors (e.g., merging two unrelated stories into one cluster).
Therefore the evaluation focuses on structural clustering metrics.

## Structural objective and anti-merge constraints

Thresholds are tuned on **dev** only.
The tuning objective rewards cluster quality (pairwise F1, ARI, NMI) while penalizing
false merges and false splits.

Additionally, anti-merge constraints ensure the tuned configuration does not collapse
everything into a few large clusters.

## Namespace audit for gold_story_id

When combining two independently labeled datasets, raw IDs can collide (both can start from `1`).
The experiment enforces separate namespaces:
- old gold: `OLD_*`
- expansion gold: `EXP2_*`

This prevents accidental cross-source story merges.

## No gold leakage

- Gold labels are never used as features in the assignment decision.
- The LLM baseline never receives gold labels in the prompt.

## Limitations

- The public repo omits private data sources and internal graph feature computation.
- Graph-derived features are represented as an interface/hook to keep the evaluation logic clear.

