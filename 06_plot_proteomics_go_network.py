#!/usr/bin/env python

import argparse
import math
import os
import re
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def strip_go_id(term):
    return re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term)).strip()


def wrap_label(term, width):
    return "\n".join(textwrap.wrap(strip_go_id(term), width=width))


def score_column(nodes):
    for column in ("neglog10_pval", "neglog10_padj"):
        if column in nodes.columns:
            return column
    raise KeyError("nodes table must include neglog10_pval or neglog10_padj")


def load_graph(nodes_path, edges_path):
    nodes = pd.read_csv(nodes_path, sep="\t")
    edges = pd.read_csv(edges_path, sep="\t")
    required_node_columns = {"term", "module"}
    required_edge_columns = {"source", "target", "weight"}
    missing_node_columns = required_node_columns - set(nodes.columns)
    missing_edge_columns = required_edge_columns - set(edges.columns)
    if missing_node_columns:
        raise KeyError(f"nodes table is missing columns: {sorted(missing_node_columns)}")
    if missing_edge_columns:
        raise KeyError(f"edges table is missing columns: {sorted(missing_edge_columns)}")
    if edges.empty:
        raise ValueError("edges table contains no network edges")
    score_col = score_column(nodes)

    if nodes["term"].duplicated().any():
        duplicated_terms = sorted(nodes.loc[nodes["term"].duplicated(), "term"].astype(str).unique())
        raise ValueError(f"nodes table contains duplicated terms: {duplicated_terms[:10]}")
    edge_nodes = set(edges["source"]).union(set(edges["target"]))
    missing_edge_nodes = sorted(edge_nodes - set(nodes["term"]))
    if missing_edge_nodes:
        raise ValueError(f"edges table references nodes absent from nodes table: {missing_edge_nodes[:10]}")
    nodes = nodes.loc[nodes["term"].isin(edge_nodes)].copy()
    if nodes.empty:
        raise ValueError("network contains no nodes referenced by edges")

    graph = nx.Graph()
    for _, row in nodes.iterrows():
        if pd.isna(row["term"]) or pd.isna(row["module"]) or pd.isna(row[score_col]):
            raise ValueError("nodes table contains missing term, module, or score values")
        graph.add_node(
            row["term"],
            module=int(row["module"]),
            score=float(row[score_col]),
        )
    for _, row in edges.iterrows():
        if pd.isna(row["source"]) or pd.isna(row["target"]) or pd.isna(row["weight"]):
            raise ValueError("edges table contains missing source, target, or weight values")
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    return graph


def representatives(graph):
    module_nodes = {}
    for node, data in graph.nodes(data=True):
        module_nodes.setdefault(data["module"], []).append(node)
    reps = []
    for nodes in module_nodes.values():
        reps.append(max(nodes, key=lambda node: graph.nodes[node]["score"]))
    return set(reps)


def normalize_sizes(values, low=260, high=1500):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span <= 0:
        return np.full(values.shape, (low + high) / 2)
    return low + (high - low) * (values - values.min()) / span


def draw_graph(graph, output_pdf, output_png, args):
    n_nodes = max(1, graph.number_of_nodes())
    layout_k = args.k_factor / math.sqrt(n_nodes)
    pos = nx.spring_layout(graph, seed=args.seed, weight="weight", k=layout_k, iterations=args.iterations)

    modules = sorted({data["module"] for _, data in graph.nodes(data=True)})
    cmap = plt.get_cmap("tab20")
    module_colors = {module: cmap(i % cmap.N) for i, module in enumerate(modules)}
    node_colors = [module_colors[graph.nodes[node]["module"]] for node in graph.nodes()]
    scores = [graph.nodes[node]["score"] for node in graph.nodes()]
    node_sizes = normalize_sizes(scores)
    edge_widths = [0.4 + 2.4 * graph[u][v].get("weight", 0.0) for u, v in graph.edges()]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, constrained_layout=True)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, edge_color="#B0BEC5", alpha=0.50)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.6,
        edgecolors="#333333",
        alpha=0.90,
    )

    labels = {node: wrap_label(node, args.label_width) for node in representatives(graph)}
    texts = nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color="#2C3E50", ax=ax)
    for text in texts.values():
        text.set_path_effects([pe.withStroke(linewidth=0, foreground="white")])

    ax.set_axis_off()
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default="paper_tables/proteomics_go_network/network_nodes.tsv")
    parser.add_argument("--edges", default="paper_tables/proteomics_go_network/network_edges.tsv")
    parser.add_argument("--outdir", default="paper_figures/proteomics_go_network")
    parser.add_argument("--seed", type=int, default=1018)
    parser.add_argument("--k-factor", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--label-width", type=int, default=32)
    args = parser.parse_args()

    graph = load_graph(args.nodes, args.edges)
    if graph.number_of_nodes() == 0:
        raise ValueError("network contains no nodes")

    draw_graph(
        graph,
        os.path.join(args.outdir, "proteomics_go_network.pdf"),
        os.path.join(args.outdir, "proteomics_go_network.png"),
        args,
    )


if __name__ == "__main__":
    main()
