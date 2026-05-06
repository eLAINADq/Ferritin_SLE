#!/usr/bin/env python

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scanpy as sc


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

BASE_COLORS = [
    "#88CCEE", "#44AA99", "#117733", "#332288", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#999933", "#E69F00",
    "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#77AADD", "#99DDFF", "#44BB99", "#BBCC33",
]


def mm_to_inch(value):
    return value / 25.4


def build_palette(categories, desat):
    if not 0.0 <= desat <= 1.0:
        raise ValueError("desat must be between 0 and 1")
    if len(categories) > len(BASE_COLORS):
        raise ValueError("number of categories exceeds the predefined color palette")
    colors = []
    for idx, _ in enumerate(categories):
        rgb = np.array(matplotlib.colors.to_rgb(BASE_COLORS[idx % len(BASE_COLORS)]))
        rgb = (1.0 - desat) * rgb + desat
        colors.append(matplotlib.colors.to_hex(rgb))
    return dict(zip(categories, colors))


def compute_centroids(x_values, y_values, labels, min_count):
    data = pd.DataFrame({"x": x_values, "y": y_values, "label": labels})
    centroids = {}
    for label, subset in data.groupby("label", observed=False):
        if len(subset) >= min_count:
            centroids[label] = (
                float(subset["x"].median()),
                float(subset["y"].median()),
                int(len(subset)),
            )
    return centroids


def draw_umap(adata, args):
    adata.obs[args.group_col] = adata.obs[args.group_col].astype("category")
    coordinates = adata.obsm[args.embedding_key]
    if coordinates.shape[1] < 2:
        raise ValueError(f"{args.embedding_key} must contain at least two dimensions")
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    categories = adata.obs[args.group_col].cat.categories.tolist()
    labels = adata.obs[args.group_col].values
    color_map = build_palette(categories, args.desat)
    counts = pd.Series(labels).value_counts()
    centroids = compute_centroids(x_values, y_values, labels, args.label_min_count)

    fig, ax = plt.subplots(
        figsize=(mm_to_inch(args.width_mm), mm_to_inch(args.height_mm)),
        dpi=300,
        constrained_layout=True,
    )
    ax.scatter(x_values, y_values, s=args.point_size, c="#D9D9D9", linewidths=0, rasterized=True)
    for category in categories:
        mask = labels == category
        if np.any(mask):
            ax.scatter(
                x_values[mask],
                y_values[mask],
                s=args.point_size,
                c=[color_map[category]],
                edgecolors="none",
                linewidths=0,
                rasterized=True,
            )

    label_order = counts.index.tolist()
    if args.label_topn >= 0:
        label_order = label_order[:args.label_topn]
    for label in label_order:
        if label not in centroids:
            continue
        x_coord, y_coord, _ = centroids[label]
        text = ax.text(
            x_coord,
            y_coord,
            str(label),
            fontsize=9,
            ha="center",
            va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=0, foreground="white")],
        )
        text.set_clip_on(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor=color_map[category], markersize=6, label=str(category))
        for category in categories[: min(18, len(categories))]
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=10, title=args.group_col, title_fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(args.outdir, exist_ok=True)
    fig.savefig(os.path.join(args.outdir, "figure1_umap_celltypes.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "figure1_umap_celltypes.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "category": categories,
        "color_hex": [color_map[category] for category in categories],
        "count": [int(counts.get(category, 0)) for category in categories],
        "centroid_x": [centroids.get(category, (np.nan, np.nan, 0))[0] for category in categories],
        "centroid_y": [centroids.get(category, (np.nan, np.nan, 0))[1] for category in categories],
    }).to_csv(os.path.join(args.outdir, "figure1_umap_celltype_centroids.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", default="fullblood0822.h5ad")
    parser.add_argument("--outdir", default="paper_figures/figure1")
    parser.add_argument("--group-col", default="celltype_fine")
    parser.add_argument("--embedding-key", default="X_umap")
    parser.add_argument("--width-mm", type=float, default=183.0)
    parser.add_argument("--height-mm", type=float, default=183.0)
    parser.add_argument("--point-size", type=float, default=0.9)
    parser.add_argument("--desat", type=float, default=0.25)
    parser.add_argument("--label-topn", type=int, default=30)
    parser.add_argument("--label-min-count", type=int, default=0)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    if args.group_col not in adata.obs:
        raise KeyError(f"obs is missing {args.group_col}")
    if args.embedding_key not in adata.obsm:
        raise KeyError(f"obsm is missing {args.embedding_key}")
    draw_umap(adata, args)


if __name__ == "__main__":
    main()
