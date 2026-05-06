#!/usr/bin/env python

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def norm(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return (values - values.min()) / (np.ptp(values) + 1e-9)


def compute_io_strength(edge_df):
    required_columns = {"source", "target", "weight", "n_pairs"}
    missing_columns = required_columns - set(edge_df.columns)
    if missing_columns:
        raise KeyError(f"edge table is missing columns: {sorted(missing_columns)}")
    if edge_df.empty:
        raise ValueError("edge table is empty")
    outgoing = edge_df.groupby("source", as_index=False)["weight"].sum().rename(
        columns={"source": "celltype", "weight": "outgoing"}
    )
    incoming = edge_df.groupby("target", as_index=False)["weight"].sum().rename(
        columns={"target": "celltype", "weight": "incoming"}
    )
    n_out = edge_df.groupby("source", as_index=False)["n_pairs"].sum().rename(
        columns={"source": "celltype", "n_pairs": "n_out"}
    )
    n_in = edge_df.groupby("target", as_index=False)["n_pairs"].sum().rename(
        columns={"target": "celltype", "n_pairs": "n_in"}
    )
    data = outgoing.merge(incoming, on="celltype", how="outer").fillna(0.0)
    data = data.merge(n_out, on="celltype", how="left").merge(n_in, on="celltype", how="left").fillna(0)
    data["total"] = data["outgoing"] + data["incoming"]
    data["n_links"] = data["n_out"] + data["n_in"]
    return data


def plot_role_scatter(io_df, title, output):
    if io_df.empty:
        raise ValueError(f"no signaling role data for {title}")
    max_links = max(float(io_df["n_links"].max()), 1.0)
    sizes = 50 + 650 * io_df["n_links"] / max_links
    limit = max(float(io_df["outgoing"].max()), float(io_df["incoming"].max()), 1e-9) * 1.08

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=0.8, color="#777777", alpha=0.7)
    scatter = ax.scatter(
        io_df["outgoing"],
        io_df["incoming"],
        s=sizes,
        c=io_df["total"],
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in io_df.iterrows():
        ax.text(
            row["outgoing"],
            row["incoming"],
            str(row["celltype"]),
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.5),
        )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Total strength")
    ax.set_xlabel("Outgoing strength")
    ax.set_ylabel("Incoming strength")
    ax.set_title(title)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_role_scatter_joint(io_ctrl, io_case, output):
    all_cells = sorted(set(io_ctrl["celltype"]).union(set(io_case["celltype"])))
    ctrl = io_ctrl.set_index("celltype").reindex(all_cells).fillna(0.0).reset_index()
    case = io_case.set_index("celltype").reindex(all_cells).fillna(0.0).reset_index()
    ctrl.columns = ["celltype", "out_ctrl", "in_ctrl", "n_out_ctrl", "n_in_ctrl", "total_ctrl", "links_ctrl"]
    case.columns = ["celltype", "out_case", "in_case", "n_out_case", "n_in_case", "total_case", "links_case"]
    data = ctrl.merge(case, on="celltype", how="outer").fillna(0.0)
    limit = max(data[["out_ctrl", "in_ctrl", "out_case", "in_case"]].to_numpy().max(), 1e-9) * 1.08
    link_max = max(data["links_ctrl"].max(), data["links_case"].max(), 1.0)

    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=300)
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=0.8, color="#777777", alpha=0.7, zorder=1)
    for _, row in data.iterrows():
        ax.annotate(
            "",
            xy=(row["out_case"], row["in_case"]),
            xytext=(row["out_ctrl"], row["in_ctrl"]),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333", alpha=0.55),
            zorder=2,
        )
    ax.scatter(
        data["out_ctrl"],
        data["in_ctrl"],
        s=60 + 500 * data["links_ctrl"] / link_max,
        c="#4C78A8",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label="Ctrl_WT",
        zorder=3,
    )
    ax.scatter(
        data["out_case"],
        data["in_case"],
        s=60 + 500 * data["links_case"] / link_max,
        c="#E45756",
        marker="s",
        edgecolors="black",
        linewidths=0.5,
        label="Pristane_WT",
        zorder=4,
    )
    for _, row in data.iterrows():
        ax.text(
            row["out_case"],
            row["in_case"],
            str(row["celltype"]),
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.5),
            zorder=5,
        )
    ax.set_xlabel("Outgoing strength")
    ax.set_ylabel("Incoming strength")
    ax.set_title("Signaling roles")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def draw_network(edge_df, title, output, difference=False):
    required_columns = {"source", "target", "weight"}
    if difference:
        required_columns.add("diff")
    missing_columns = required_columns - set(edge_df.columns)
    if missing_columns:
        raise KeyError(f"edge table is missing columns: {sorted(missing_columns)}")
    if edge_df.empty:
        raise ValueError(f"edge table is empty for {title}")
    nodes = sorted(set(edge_df["source"]).union(set(edge_df["target"])))
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    weights = norm(edge_df["weight"].to_numpy())
    diffs = norm(np.abs(edge_df["diff"].to_numpy())) if difference and "diff" in edge_df else None

    for idx, (_, row) in enumerate(edge_df.iterrows()):
        if difference:
            color = "#C44E52" if row["diff"] >= 0 else "#4C72B0"
            alpha = 0.25 + 0.75 * float(diffs[idx])
        else:
            color = "#444444"
            alpha = 0.20 + 0.75 * float(weights[idx])
        graph.add_edge(row["source"], row["target"], color=color, alpha=alpha, width=0.5 + 5.5 * float(weights[idx]))

    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=1300, node_color="#F2F2F2", edgecolors="#555555", linewidths=1.0)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight="bold")
    for source, target, data in graph.edges(data=True):
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(source, target)],
            ax=ax,
            width=data["width"],
            edge_color=data["color"],
            alpha=data["alpha"],
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.08",
        )
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def bubbleplot_lr(result_df, source, targets, top_n, q_threshold, title, output):
    required_columns = {"source", "target", "ligand", "receptor", "lr_means", "qval_bh"}
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise KeyError(f"LIANA result table is missing columns: {sorted(missing_columns)}")
    rows = []
    for target in targets:
        subset = result_df[(result_df["source"] == source) & (result_df["target"] == target)].copy()
        subset = subset[subset["qval_bh"] <= q_threshold]
        if subset.empty:
            continue
        subset = subset.sort_values("lr_means", ascending=False).head(top_n)
        subset["cell_pair"] = f"{source}->{target}"
        subset["lr_pair"] = subset["ligand"].astype(str) + "->" + subset["receptor"].astype(str)
        rows.append(subset)
    if not rows:
        raise ValueError(f"no significant ligand-receptor pairs for {source} to {targets}")
    data = pd.concat(rows, axis=0)
    data["neglog10q"] = -np.log10(data["qval_bh"].replace(0, 1e-300))

    fig, ax = plt.subplots(figsize=(7.0, max(3.5, 0.23 * data["lr_pair"].nunique())), dpi=300)
    sns.scatterplot(
        data=data,
        x="cell_pair",
        y="lr_pair",
        size="neglog10q",
        sizes=(12, 260),
        hue="lr_means",
        palette="rocket",
        edgecolor="none",
        ax=ax,
    )
    ax.set_xlabel("Cell pair")
    ax.set_ylabel("Ligand-receptor pair")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="paper_tables/cell_cell_communication")
    parser.add_argument("--figures", default="paper_figures/cell_cell_communication")
    parser.add_argument("--focus-source", default="Neutrophil")
    parser.add_argument("--focus-targets", nargs="+", default=["Classical monocyte", "Non-classical monocyte"])
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.figures, exist_ok=True)
    edges_ctrl = pd.read_csv(os.path.join(args.tables, "edges_ctrl.tsv"), sep="\t")
    edges_case = pd.read_csv(os.path.join(args.tables, "edges_case.tsv"), sep="\t")
    edges_diff = pd.read_csv(os.path.join(args.tables, "edges_diff.tsv"), sep="\t")
    cpdb_ctrl = pd.read_csv(os.path.join(args.tables, "cpdb_ctrl.tsv"), sep="\t")
    cpdb_case = pd.read_csv(os.path.join(args.tables, "cpdb_case.tsv"), sep="\t")

    draw_network(edges_ctrl, "Ctrl_WT", os.path.join(args.figures, "cell_communication_network_ctrl.pdf"))
    draw_network(edges_case, "Pristane_WT", os.path.join(args.figures, "cell_communication_network_pristane.pdf"))
    draw_network(
        edges_diff,
        "Pristane_WT - Ctrl_WT",
        os.path.join(args.figures, "cell_communication_network_difference.pdf"),
        difference=True,
    )

    io_ctrl = compute_io_strength(edges_ctrl)
    io_case = compute_io_strength(edges_case)
    plot_role_scatter(io_ctrl, "Ctrl_WT signaling roles", os.path.join(args.figures, "cell_communication_role_ctrl.pdf"))
    plot_role_scatter(io_case, "Pristane_WT signaling roles", os.path.join(args.figures, "cell_communication_role_pristane.pdf"))
    plot_role_scatter_joint(io_ctrl, io_case, os.path.join(args.figures, "cell_communication_role_joint.pdf"))

    bubbleplot_lr(
        cpdb_ctrl,
        args.focus_source,
        args.focus_targets,
        args.top_n,
        args.q_threshold,
        "Ctrl_WT ligand-receptor pairs",
        os.path.join(args.figures, "cell_communication_lr_bubble_ctrl.pdf"),
    )
    bubbleplot_lr(
        cpdb_case,
        args.focus_source,
        args.focus_targets,
        args.top_n,
        args.q_threshold,
        "Pristane_WT ligand-receptor pairs",
        os.path.join(args.figures, "cell_communication_lr_bubble_pristane.pdf"),
    )


if __name__ == "__main__":
    main()
