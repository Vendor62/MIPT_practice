from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figs = root / "figures"

    df = pd.read_csv(results / "final_metrics_summary.csv")

    # final_metrics_f1_ari_nmi.png
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = range(len(df))
    ax.bar([i - 0.25 for i in x], df["pairwise_f1"], width=0.25, label="Pairwise F1")
    ax.bar([i for i in x], df["ari"], width=0.25, label="ARI")
    ax.bar([i + 0.25 for i in x], df["nmi"], width=0.25, label="NMI")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, figs / "final_metrics_f1_ari_nmi.png")

    # hybrid_delta_vs_semantic_target.png
    sem = df[df["method"] == "semantic_only"].iloc[0]
    hyb = df[df["method"] == "hybrid_v4_1_soft_entity_penalty"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(
        ["pairwise_f1", "ari", "nmi"],
        [hyb["pairwise_f1"] - sem["pairwise_f1"], hyb["ari"] - sem["ari"], hyb["nmi"] - sem["nmi"]],
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Hybrid − Semantic-only (test)")
    _save(fig, figs / "hybrid_delta_vs_semantic_target.png")

    # false_merge_false_split_comparison.png
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.bar([i - 0.2 for i in x], df["false_merge_rate"], width=0.4, label="False merge rate")
    ax.bar([i + 0.2 for i in x], df["false_split_rate"], width=0.4, label="False split rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    _save(fig, figs / "false_merge_false_split_comparison.png")

    # dataset_story_composition.png
    ds = json.loads((results / "dataset_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.pie([ds["singleton_stories"], ds["non_singleton_stories"]], labels=["singleton", "non-singleton"], autopct="%1.1f%%")
    ax.set_title("Gold story composition")
    _save(fig, figs / "dataset_story_composition.png")

    # gold_story_size_distribution.png (placeholder)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.text(0.5, 0.5, "Gold story size distribution\n(not included in public repo)", ha="center", va="center")
    ax.set_axis_off()
    _save(fig, figs / "gold_story_size_distribution.png")

    # experiment_progression_summary.png (schematic)
    fig, ax = plt.subplots(figsize=(7.8, 2.6))
    ax.text(
        0.5,
        0.5,
        "Progression: weak labels → manual gold v1 → expansion v2 → combined v2\nMetrics: semantic vs hybrid vs LLM baseline",
        ha="center",
        va="center",
    )
    ax.set_axis_off()
    _save(fig, figs / "experiment_progression_summary.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

