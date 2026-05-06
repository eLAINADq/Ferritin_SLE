#!/usr/bin/env python

import argparse
import os

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

cmap_gray_cyan = mpl.colors.LinearSegmentedColormap.from_list(
    "gray_cyan", ["#E0E0E0", "#19735b"], N=256
)

PAPER_W_MM = 183.0
PAPER_H_MM = 120.0
MARKERS = {
    "B cell": ["Ms4a1", "Cd79a", "Cd74"],
    "Basophil": ["Mcpt8", "Fcer1a", "Il3ra"],
    "CD4+ T cell": ["Il7r", "Ccr7", "Tcf7"],
    "CD8+ T cell": ["Cd8a", "Gzmb", "Prf1"],
    "Classical monocyte": ["Ly6c2", "Ccr2", "S100a8"],
    "Dendritic cell": ["Itgax", "H2-Ab1", "Xcr1"],
    "Erythrocyte": ["Hbb-bs", "Hba-a1", "Alas2"],
    "Natural killer cell": ["Ncr1", "Klrd1", "Nkg7"],
    "Neutrophil": ["Ly6g", "S100a8", "S100a9"],
    "Non-classical monocyte": ["Cx3cr1", "Nr4a1", "Fcgr3"],
}


def mm2inch(mm):
    return mm / 25.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", default="fullblood0822.h5ad")
    parser.add_argument("--groupby", default="celltype_fine")
    parser.add_argument("--outdir", default="paper_figures/figure2")
    parser.add_argument("--width-mm", type=float, default=PAPER_W_MM)
    parser.add_argument("--height-mm", type=float, default=PAPER_H_MM)
    parser.add_argument("--dot-max", type=float, default=0.6)
    parser.add_argument("--mean-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outpdf = os.path.join(args.outdir, "figure2_marker_gene_dotplot.pdf")
    outpng = os.path.join(args.outdir, "figure2_marker_gene_dotplot.png")
    outrep = os.path.join(args.outdir, "figure2_marker_gene_match_report.csv")

    adata = sc.read_h5ad(args.h5ad)
    if args.groupby not in adata.obs:
        raise KeyError(f"obs is missing {args.groupby}")

    wanted = list(MARKERS.keys())
    adata.obs[args.groupby] = adata.obs[args.groupby].astype("category")
    categories = set(adata.obs[args.groupby].cat.categories)
    missing_groups = [group for group in wanted if group not in categories]
    if missing_groups:
        raise ValueError(f"target groups are absent from {args.groupby}: {missing_groups}")

    adata = adata[adata.obs[args.groupby].isin(wanted), :].copy()
    adata.obs[args.groupby] = (
        adata.obs[args.groupby]
        .cat.remove_unused_categories()
        .cat.reorder_categories(wanted, ordered=True)
    )

    upper_to_var = {str(var).upper(): str(var) for var in adata.var_names}
    report_rows = []
    positions = []
    varlist_for_plot = []
    missing_markers = []
    idx = 0

    for group in wanted:
        genes = MARKERS[group]
        keep = []
        for symbol in genes:
            hit = upper_to_var.get(symbol.upper())
            report_rows.append({
                "celltype": group,
                "input_symbol": symbol,
                "matched": hit is not None,
                "used_token": hit or "",
                "match_mode": "var_names",
            })
            if hit is not None:
                keep.append(hit)
            else:
                missing_markers.append((group, symbol))
        varlist_for_plot.extend(keep)
        positions.append((idx, idx + len(keep) - 1))
        idx += len(keep)

    pd.DataFrame(report_rows).to_csv(outrep, index=False)
    if missing_markers:
        missing = ", ".join([f"{group}:{symbol}" for group, symbol in missing_markers])
        raise ValueError(f"marker genes are absent from var_names: {missing}")

    width = mm2inch(args.width_mm)
    height = mm2inch(args.height_mm)
    with plt.rc_context({"figure.figsize": (width, height)}):
        dotplot = sc.pl.dotplot(
            adata,
            varlist_for_plot,
            groupby=args.groupby,
            categories_order=wanted,
            var_group_positions=positions,
            var_group_labels=wanted,
            standard_scale=None if args.mean_only else "var",
            dot_max=args.dot_max,
            color_map=cmap_gray_cyan,
            return_fig=True,
            figsize=(width, height),
        )
        dotplot.make_figure()
        for ax in getattr(dotplot, "axes_dict", {}).values():
            ax.tick_params(axis="x", labelrotation=45)
        dotplot.savefig(outpdf, bbox_inches="tight")
        dotplot.savefig(outpng, dpi=600, bbox_inches="tight")

    print(f"Saved {outpdf}")
    print(f"Saved {outpng}")
    print(f"Saved {outrep}")


if __name__ == "__main__":
    main()
