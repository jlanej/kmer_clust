"""Stage: static analytical figures (PNG) for the README and the report page.

Chrome follows the dataviz token set (light surface, recessive grid, muted ink).
Censat classes use the track's own RGB convention so T2T people recognize them;
clusters use the categorical slots cycled with lightness steps, assigned in
cluster-size order; continuous values use a single-hue ramp.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUT, Params
from .annotate import CENSAT_CLASSES, censat_class

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
NON_SAT = "#c9c8c2"
NOISE = "#dddcd6"

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

FIG_DIR = OUT / "figs"


def style_ax(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def censat_palette(params: Params) -> dict:
    """class -> hex from the censat track's own RGB column (median per class)."""
    from .annotate import load_bed

    bed = load_bed(params.censat, [3, 8], ["name", "rgb"])
    bed["cls"] = bed["name"].map(censat_class)
    pal = {}
    for cls, g in bed[bed["cls"] != ""].groupby("cls"):
        rgb = np.median(
            np.array([[int(x) for x in r.split(",")] for r in g["rgb"]]), axis=0
        ).astype(int)
        pal[cls] = "#{:02x}{:02x}{:02x}".format(*rgb)
    pal["non_sat"] = NON_SAT
    pal.setdefault("ct", "#b9b8b2")
    pal["rDNA"] = "#8a6d3b"  # track's near-black would vanish on dark surfaces
    return pal


def cluster_palette(labels: np.ndarray) -> dict:
    """cluster id -> hex; biggest clusters get the clean slots, then stepped."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.argsort(counts)[::-1]]
    pal = {-1: NOISE}
    for rank, cid in enumerate(order):
        base = SLOTS[rank % len(SLOTS)]
        cycle = rank // len(SLOTS)
        r, g, b = (int(base[i : i + 2], 16) for i in (1, 3, 5))
        if cycle:
            f = (0.72, 1.28, 0.5, 1.55)[(cycle - 1) % 4]
            r, g, b = (min(255, max(0, int(v * f + (28 if f > 1 else 0)))) for v in (r, g, b))
        pal[int(cid)] = f"#{r:02x}{g:02x}{b:02x}"
    return pal


def fig_umap(bins, annot, pal_censat, pal_cluster):
    color_sat = np.array([
        pal_censat.get(c if c else "non_sat", NON_SAT) for c in annot["censat_class"]
    ])
    color_clu = np.array([pal_cluster[int(c)] for c in bins["cluster"]])
    for tag, colors, title in (
        ("censat", color_sat, "UMAP of 100 kb bins — censat annotation (not an input)"),
        ("clusters", color_clu, "UMAP of 100 kb bins — HDBSCAN clusters"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160, facecolor=SURFACE)
        style_ax(ax, title)
        ax.scatter(bins["x"], bins["y"], s=1.4, c=colors, linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        if tag == "censat":
            present = [c for c in CENSAT_CLASSES + ["non_sat"] if c in pal_censat]
            handles = [
                plt.Line2D([], [], marker="o", ls="", ms=5, color=pal_censat[c], label=c)
                for c in present
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=6.5, frameon=False,
                      labelcolor=INK, ncols=2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"umap_{tag}.png", facecolor=SURFACE)
        plt.close(fig)


def fig_ribbon(bins, pal_cluster):
    chroms = list(dict.fromkeys(bins["chrom"]))
    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, chrom in enumerate(chroms):
        g = bins[bins["chrom"] == chrom]
        y = len(chroms) - i
        colors = [pal_cluster[int(c)] for c in g["cluster"]]
        ax.bar(
            g["start"] / 1e6, 0.72, width=(g["end"] - g["start"]) / 1e6,
            bottom=y - 0.36, color=colors, align="edge", linewidth=0,
        )
        ax.text(-2, y, chrom.replace("chr", ""), ha="right", va="center",
                fontsize=7, color=INK)
    ax.set_ylim(0.3, len(chroms) + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("position (Mb)", color=MUTED, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("The genome painted by vocabulary cluster", color=INK,
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ribbon.png", facecolor=SURFACE)
    plt.close(fig)


def fig_confusions():
    an = np.load(OUT / "analysis.npz")
    names = json.loads((OUT / "conf_names.json").read_text())
    metrics = json.loads((OUT / "metrics.json").read_text())
    panels = [p for p in ("alpha", "rdna", "taxonomy") if f"conf_{p}" in an]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6),
                             dpi=160, facecolor=SURFACE)
    if len(panels) == 1:
        axes = [axes]
    titles = {
        "alpha": f"αSat HOR bins: which chromosome?\nkNN acc {metrics.get('alpha_chrom_knn_acc', 0):.1%} (chance {metrics.get('alpha_chrom_chance', 0):.1%})",
        "rdna": f"rDNA bins: which chromosome?\nkNN acc {metrics.get('rdna_chrom_knn_acc', 0):.1%} (chance {metrics.get('rdna_chrom_chance', 0):.1%})",
        "taxonomy": f"satellite bins: which family?\nkNN acc {metrics.get('sat_taxonomy_knn_acc', 0):.1%}",
    }
    for ax, p in zip(axes, panels):
        conf = an[f"conf_{p}"].astype(float)
        row = conf.sum(axis=1, keepdims=True)
        conf = np.divide(conf, row, out=np.zeros_like(conf), where=row > 0)
        labs = [n.replace("chr", "").replace("asat_", "α") for n in names[p]]
        ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labs)), labs, fontsize=6, rotation=90, color=MUTED)
        ax.set_yticks(range(len(labs)), labs, fontsize=6, color=MUTED)
        ax.set_title(titles[p], color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel("predicted", fontsize=7, color=MUTED)
        if p == panels[0]:
            ax.set_ylabel("true", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusions.png", facecolor=SURFACE)
    plt.close(fig)


def fig_sweep():
    an = np.load(OUT / "analysis.npz")
    ari, grid = an["sweep_ari"], an["sweep_grid"]
    labs = [f"mcs{m}/ms{s}" for m, s in grid]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=160, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(ari, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)), labs, fontsize=6, rotation=90, color=MUTED)
    ax.set_yticks(range(len(labs)), labs, fontsize=6, color=MUTED)
    ax.set_title("Cluster agreement (ARI) across HDBSCAN settings",
                 color=INK, fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=7, colors=MUTED)
    ax = axes[1]
    style_ax(ax, "clusters found vs noise, per setting")
    ax.scatter(an["sweep_n_clusters"], an["sweep_noise"] * 100, s=30,
               c="#2a78d6", zorder=3)
    for (m, s), x, y in zip(grid, an["sweep_n_clusters"], an["sweep_noise"] * 100):
        ax.annotate(f"{m}/{s}", (x, y), fontsize=6, color=MUTED,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("n clusters", fontsize=8, color=MUTED)
    ax.set_ylabel("% noise", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep.png", facecolor=SURFACE)
    plt.close(fig)


def fig_ab_agreement():
    ab = np.load(OUT / "ab_agreement.npz")
    metrics = json.loads((OUT / "metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=160, facecolor=SURFACE)
    style_ax(
        ax,
        "Track A (cosine, SVD of weighted matrix) vs\n"
        f"Track B (exact 1−Jaccard), 1 Mb bins — ρ={metrics['trackA_vs_trackB_spearman']:.3f}",
    )
    hb = ax.hexbin(ab["d_jacc"], ab["cos_d"], gridsize=60, cmap="Blues", bins="log",
                   linewidths=0)
    ax.set_xlabel("1 − Jaccard (exact)", fontsize=8, color=MUTED)
    ax.set_ylabel("cosine distance (model)", fontsize=8, color=MUTED)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="log pairs").ax.tick_params(
        labelsize=7, colors=MUTED
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ab_agreement.png", facecolor=SURFACE)
    plt.close(fig)


def fig_seeds(bins, annot, pal_censat):
    se = np.load(OUT / "seed_embeds.npz")
    metrics = json.loads((OUT / "metrics.json").read_text())
    color_sat = np.array([
        pal_censat.get(c if c else "non_sat", NON_SAT) for c in annot["censat_class"]
    ])
    embeds = [("seed 42 (main)", bins[["x", "y"]].to_numpy())] + [
        (k.replace("seed", "seed "), se[k]) for k in se.files
    ]
    fig, axes = plt.subplots(1, len(embeds), figsize=(3.1 * len(embeds), 3.3),
                             dpi=160, facecolor=SURFACE)
    for ax, (name, E) in zip(axes, embeds):
        ax.set_facecolor(SURFACE)
        ax.scatter(E[:, 0], E[:, 1], s=0.5, c=color_sat, linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(name, color=INK, fontsize=8, loc="left")
    fig.suptitle(
        f"UMAP under different seeds — mean kNN overlap {metrics['umap_seed_knn_overlap']:.2f}",
        color=INK, fontsize=9, x=0.01, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "seeds.png", facecolor=SURFACE)
    plt.close(fig)


def fig_pairwise_heatmap():
    pw = np.load(OUT / "pairwise.npz")
    parents = pd.read_parquet(OUT / "pairwise_bins.parquet")
    sim = np.clip(1 - pw["d_jacc"], 0, 1) ** (1 / 3)  # cbrt-stretched exact Jaccard
    leaf = pw["leaf_order"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), dpi=160, facecolor=SURFACE)
    chroms = parents["chrom"].to_numpy()
    bounds = np.flatnonzero(chroms[1:] != chroms[:-1]) + 1
    for ax, order, title in (
        (axes[0], np.arange(len(parents)), "genome order"),
        (axes[1], leaf, "dendrogram order (average linkage, OLO)"),
    ):
        ax.imshow(sim[np.ix_(order, order)], cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"1 Mb exact Jaccard (∛-stretch) — {title}", color=INK,
                     fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        if title == "genome order":
            for b in bounds:
                ax.axhline(b - 0.5, color=SURFACE, lw=0.4)
                ax.axvline(b - 0.5, color=SURFACE, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pairwise_heatmap.png", facecolor=SURFACE)
    plt.close(fig)


def run(params: Params) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    pal_censat = censat_palette(params)
    pal_cluster = cluster_palette(bins["cluster"].to_numpy())
    with open(OUT / "palette.json", "w") as fh:
        json.dump({"censat": pal_censat,
                   "cluster": {str(k): v for k, v in pal_cluster.items()}}, fh)
    fig_umap(bins, annot, pal_censat, pal_cluster)
    fig_ribbon(bins, pal_cluster)
    fig_confusions()
    fig_sweep()
    fig_ab_agreement()
    fig_seeds(bins, annot, pal_censat)
    fig_pairwise_heatmap()
    fig_acro(params)
    print(f"figures -> {FIG_DIR}")


def fig_acro(params: Params):
    """Acro p-arms: map position + where along each arm identity survives."""
    acro_path = OUT / "acro.parquet"
    kl_path = OUT / "kladder.npz"
    if not (acro_path.exists() and kl_path.exists()):
        return
    from .analyze import ACROS

    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    ac = pd.read_parquet(acro_path)
    xy = np.load(kl_path)[f"k{params.k}"]
    COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7"]
    GREY = np.array([201, 200, 194]) / 255
    col = np.tile(np.array([232, 230, 223]) / 255 * 0.93, (len(bins), 1))
    acro_map = np.zeros(len(bins), int)
    prom = np.zeros(len(bins))
    acro_map[ac["bin"]] = [ACROS.index(c) + 1 for c in ac["chrom"]]
    prom[ac["bin"]] = ac["promiscuity"]
    for ci, c in enumerate(COLS):
        m = acro_map == ci + 1
        base = np.array([int(c[j : j + 2], 16) for j in (1, 3, 5)]) / 255
        t = (prom[m] * 0.85)[:, None]
        col[m] = base[None, :] * (1 - t) + GREY[None, :] * t
    fig = plt.figure(figsize=(11.5, 5.6), dpi=160, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    order = np.argsort(acro_map > 0)
    ax.scatter(xy[order, 0], xy[order, 1], s=np.where(acro_map[order] > 0, 8, 1.1),
               c=col[order], linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(GRID)
    ax.set_title("Acrocentric p-arms on the map — saturated = knows its chromosome,\n"
                 "washed out = pan-acrocentric commons", color=INK, fontsize=10, loc="left")
    axr = fig.add_subplot(gs[1])
    axr.set_facecolor(SURFACE)
    for ci, chrom in enumerate(ACROS):
        g = ac[ac["chrom"] == chrom].sort_values("start")
        y = len(ACROS) - ci
        base = np.array([int(COLS[ci][j : j + 2], 16) for j in (1, 3, 5)]) / 255
        for _, r in g.iterrows():
            t = r["promiscuity"] * 0.85
            axr.bar(r["start"] / 1e6, 0.7, width=0.1, bottom=y - 0.35,
                    color=base * (1 - t) + GREY * t, align="edge", linewidth=0)
        axr.text(-0.4, y, chrom.replace("chr", "") + "p", ha="right", va="center",
                 fontsize=9, color=INK)
    axr.set_ylim(0.4, len(ACROS) + 0.8)
    axr.set_yticks([])
    axr.set_xlabel("position on p-arm (Mb)", fontsize=8, color=MUTED)
    for s_ in ("top", "right", "left"):
        axr.spines[s_].set_visible(False)
    axr.spines["bottom"].set_color(GRID)
    axr.tick_params(colors=MUTED, labelsize=8)
    axr.set_title("Where along each arm identity survives", color=INK, fontsize=10, loc="left")
    fig.savefig(FIG_DIR / "acro_focus.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
