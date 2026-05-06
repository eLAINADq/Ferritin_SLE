#!/usr/bin/env python

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def p_label(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return f"p = {p_value:.3f}"
    if p_value < 0.05:
        return f"p = {p_value:.3f}"
    return f"p = {p_value:.3f}"


def draw_violin(scores, stats, args):
    data = scores[[args.condition_col, args.score_col]].dropna().copy()
    data = data[data[args.condition_col].isin([args.ctrl_label, args.case_label])]
    if data.empty:
        raise ValueError("no AUCell scores available for the selected groups")
    group_counts = data[args.condition_col].value_counts()
    missing_groups = [
        group for group in (args.ctrl_label, args.case_label)
        if group_counts.get(group, 0) == 0
    ]
    if missing_groups:
        raise ValueError(f"AUCell scores are missing groups: {missing_groups}")

    if "P_value" not in stats.columns:
        raise KeyError("statistics table is missing P_value")
    if len(stats) != 1:
        raise ValueError("statistics table must contain exactly one comparison row")
    p_value = float(stats["P_value"].iloc[0])
    palette = [args.ctrl_color, args.case_color]
    fig, ax = plt.subplots(figsize=(4.0, 5.0), dpi=300)

    sns.violinplot(
        data=data,
        x=args.condition_col,
        y=args.score_col,
        hue=args.condition_col,
        order=[args.ctrl_label, args.case_label],
        hue_order=[args.ctrl_label, args.case_label],
        palette=palette,
        cut=0,
        inner="box",
        linewidth=1.1,
        saturation=0.85,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=args.condition_col,
        y=args.score_col,
        order=[args.ctrl_label, args.case_label],
        color="black",
        alpha=0.25,
        jitter=0.25,
        size=1.5,
        linewidth=0,
        ax=ax,
    )

    y_min = data[args.score_col].min()
    y_max = data[args.score_col].max()
    y_range = max(y_max - y_min, 1e-9)
    y = y_max + 0.08 * y_range
    h = 0.03 * y_range
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.1, color="black")
    ax.text(0.5, y + 1.2 * h, p_label(p_value), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    fig.savefig(os.path.join(args.outdir, "aucell_ferroptosis_violin.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "aucell_ferroptosis_violin.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="paper_tables/aucell_ferroptosis/aucell_ferroptosis_scores.tsv")
    parser.add_argument("--stats", default="paper_tables/aucell_ferroptosis/statistical_results.tsv")
    parser.add_argument("--outdir", default="paper_figures/aucell_ferroptosis")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--score-col", default="AUCell_FERROPTOSIS")
    parser.add_argument("--ctrl-label", default="Ctrl_WT")
    parser.add_argument("--case-label", default="Pristane_WT")
    parser.add_argument("--ctrl-color", default="#4ecd9c")
    parser.add_argument("--case-color", default="#ff8861")
    parser.add_argument("--ylabel", default="AUCell: GOBP_FERROPTOSIS")
    parser.add_argument("--title", default="Neutrophil ferroptosis pathway activity")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, sep="\t", index_col=0)
    stats = pd.read_csv(args.stats, sep="\t")
    draw_violin(scores, stats, args)


if __name__ == "__main__":
    main()
