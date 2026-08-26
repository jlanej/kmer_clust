"""Stage: assess the map. Annotation is the judge here, never the input.

Produces out/metrics.json, out/cluster_table.parquet, out/analysis.npz with:
- cluster <-> satellite-annotation agreement (AMI/ARI, purity, fragmentation)
- the two-way acrocentric test: alpha-HOR bins should be chromosome-IDENTIFIABLE
  from vocabulary alone while rDNA bins should be chromosome-CONFUSED (both
  directions predicted by biology; a trivially-positional method fails one,
  a chromosome-blind method fails the other)
- satellite taxonomy recovery by kNN on SVD coords
- robustness: UMAP seed stability, HDBSCAN parameter sweep ARI, FracMinHash
  subsampling stability, Track A (cosine/SVD) vs Track B (exact Jaccard) accord
"""

import json
import time

import numpy as np
import pandas as pd

from .config import OUT, Params
from .embed_run import cluster_hdbscan, umap_embed
from .matrix import build_matrix, downsample_store, gram_rsvd
from .pairwise import merge_to_parents
from .sketch_run import load_store

SAT_CLASSES_FOR_PURITY = [
    "asat_hor_live", "asat_hor", "asat_dhor", "asat_mon",
    "hsat1A", "hsat1B", "hsat2", "hsat3", "bsat", "gsat", "other_sat", "rDNA",
]


def knn_indices(Z, k=15, metric="cosine", seed=42):
    from pynndescent import NNDescent

    index = NNDescent(Z, metric=metric, n_neighbors=k + 1, random_state=seed)
    idx, _ = index.neighbor_graph
    return idx[:, 1 : k + 1]


def semantic_acc(E, labels, min_n=10, metric="euclidean"):
    """kNN CV accuracy over classes with at least min_n members."""
    counts = pd.Series(labels).value_counts()
    keep = np.isin(labels, counts[counts >= min_n].index)
    acc, *_ = knn_cv_accuracy(E[keep], labels[keep], k=10, metric=metric)
    return acc


def knn_cv_accuracy(Z, labels, k=10, folds=5, seed=42, metric="cosine"):
    """Stratified kNN cross-validated prediction. Returns (acc, y_true, y_pred)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    y = pd.factorize(labels)[0]
    names = pd.factorize(labels)[1]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred = np.full_like(y, -1)
    for tr, te in skf.split(Z, y):
        clf = KNeighborsClassifier(n_neighbors=min(k, len(tr)), metric=metric)
        clf.fit(Z[tr], y[tr])
        y_pred[te] = clf.predict(Z[te])
    return float(np.mean(y_pred == y)), y, y_pred, list(names)


def neighbor_overlap(idx_a, idx_b):
    """Mean Jaccard of per-point kNN sets between two graphs."""
    k = idx_a.shape[1]
    ov = np.empty(idx_a.shape[0])
    for i in range(idx_a.shape[0]):
        inter = np.intersect1d(idx_a[i], idx_b[i], assume_unique=False).size
        ov[i] = inter / (2 * k - inter)
    return float(ov.mean())


def cluster_purity_table(bins, annot, label_col="cluster"):
    """Per-cluster summary: size, Mb, dominant censat class, chrom spread."""
    df = pd.concat([bins.reset_index(drop=True), annot.reset_index(drop=True)], axis=1)
    df = df[df[label_col] >= 0]
    rows = []
    for c, g in df.groupby(label_col):
        cls = g["censat_class"].replace("", "non_sat")
        dom_cls = cls.mode().iloc[0]
        dom_frac = float((cls == dom_cls).mean())
        chrom_counts = g["chrom"].value_counts()
        dom_chrom = chrom_counts.index[0]
        rows.append({
            "cluster": int(c),
            "n_bins": len(g),
            "mb": round(float((g["end"] - g["start"]).sum() / 1e6), 1),
            "dom_class": dom_cls,
            "dom_class_frac": round(dom_frac, 3),
            "n_chroms": int((chrom_counts > 0).sum()),
            "dom_chrom": dom_chrom,
            "dom_chrom_frac": round(float(chrom_counts.iloc[0] / len(g)), 3),
            "gc": round(float(g["gc"].mean()), 3),
            "sd_frac": round(float(g["sd_frac"].mean()), 3),
            "private_frac": round(float(g["private_frac"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("n_bins", ascending=False)


def name_clusters(table, annot_cols):
    """Human names: dominant satellite class, else repeat/GC descriptor."""
    names = {}
    for _, r in table.iterrows():
        if r["dom_class"] != "non_sat" and r["dom_class_frac"] >= 0.4:
            name = r["dom_class"]
        elif r["sd_frac"] >= 0.4:
            name = "segdup"
        else:
            name = f"euchromatin {r['gc']*100:.0f}% GC"
        loc = r["dom_chrom"] if r["dom_chrom_frac"] >= 0.8 else f"{r['n_chroms']} chroms"
        names[r["cluster"]] = f"{name} · {loc}"
    return names


def run(params: Params) -> dict:
    t0 = time.time()
    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    z = np.load(OUT / f"svd_s{params.embed_scaled}.npz")
    Z = z["Z"]
    labels = bins["cluster"].to_numpy()
    censat = annot["censat_class"].replace("", "non_sat").to_numpy()

    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    clustered = labels >= 0
    metrics = {
        "n_bins": int(len(bins)),
        "n_clusters": int(labels.max() + 1),
        "noise_frac": round(float(np.mean(~clustered)), 4),
        "ami_vs_censat": round(
            float(adjusted_mutual_info_score(censat[clustered], labels[clustered])), 4
        ),
        "ari_vs_censat": round(
            float(adjusted_rand_score(censat[clustered], labels[clustered])), 4
        ),
    }

    table = cluster_purity_table(bins, annot)
    names = name_clusters(table, annot.columns)
    table["name"] = table["cluster"].map(names)
    sat_bins = np.isin(censat, SAT_CLASSES_FOR_PURITY)
    sat_clustered = clustered & sat_bins
    if sat_clustered.sum():
        per_cluster_dom = table.set_index("cluster")
        sat_tbl = table[table["dom_class"].isin(SAT_CLASSES_FOR_PURITY)]
        w = sat_tbl["n_bins"].to_numpy(np.float64)
        metrics["sat_cluster_weighted_purity"] = round(
            float((sat_tbl["dom_class_frac"] * w).sum() / max(w.sum(), 1)), 4
        )
        metrics["sat_bins_recovered_frac"] = round(
            float(sat_clustered.sum() / max(sat_bins.sum(), 1)), 4
        )

    # ---- two-way acrocentric test ------------------------------------------
    alpha_mask = (annot["cov_asat_hor_live"] + annot["cov_asat_hor"]).to_numpy() >= 0.5
    rdna_mask = annot["cov_rDNA"].to_numpy() >= 0.5
    chroms = bins["chrom"].to_numpy()
    results_conf = {}
    for tag, mask in (("alpha", alpha_mask), ("rdna", rdna_mask)):
        if mask.sum() < 25:
            continue
        keep_cls = pd.Series(chroms[mask]).value_counts()
        keep_cls = keep_cls[keep_cls >= 5].index
        m2 = mask & np.isin(chroms, keep_cls)
        acc, y, yp, names_c = knn_cv_accuracy(Z[m2], chroms[m2], k=10)
        n_cls = len(names_c)
        conf = np.zeros((n_cls, n_cls), np.int32)
        np.add.at(conf, (y, yp), 1)
        results_conf[tag] = {"conf": conf, "names": names_c}
        metrics[f"{tag}_chrom_knn_acc"] = round(acc, 4)
        metrics[f"{tag}_chrom_n_bins"] = int(m2.sum())
        metrics[f"{tag}_chrom_n_chroms"] = n_cls
        metrics[f"{tag}_chrom_chance"] = round(
            float(pd.Series(chroms[m2]).value_counts(normalize=True).max()), 4
        )

    # satellite taxonomy recovery
    if sat_bins.sum() >= 100:
        cls_counts = pd.Series(censat[sat_bins]).value_counts()
        keep_cls = cls_counts[cls_counts >= 20].index
        m2 = sat_bins & np.isin(censat, keep_cls)
        acc, y, yp, names_s = knn_cv_accuracy(Z[m2], censat[m2], k=10)
        conf = np.zeros((len(names_s), len(names_s)), np.int32)
        np.add.at(conf, (y, yp), 1)
        results_conf["taxonomy"] = {"conf": conf, "names": names_s}
        metrics["sat_taxonomy_knn_acc"] = round(acc, 4)
        metrics["sat_taxonomy_chance"] = round(
            float(pd.Series(censat[m2]).value_counts(normalize=True).max()), 4
        )

    print(f"agreement + classification done t={time.time()-t0:.0f}s")

    # ---- robustness ---------------------------------------------------------
    # (a) UMAP seed stability on the 2-D display embedding
    seeds = [1, 2, 3]
    embeds = [umap_embed(Z, 2, params, seed=s) for s in seeds]
    base_idx = knn_indices(bins[["x", "y"]].to_numpy(np.float32), k=15, metric="euclidean")
    seed_overlaps = []
    for E in embeds:
        idx = knn_indices(E.astype(np.float32), k=15, metric="euclidean")
        seed_overlaps.append(neighbor_overlap(base_idx, idx))
    metrics["umap_seed_knn_overlap"] = round(float(np.mean(seed_overlaps)), 4)
    # semantic stability: micro-neighbors shuffle between seeds (dense regions
    # have interchangeable neighbors), but can each seed's 2-d map still tell
    # satellite families apart in place?
    sem = [
        semantic_acc(E[sat_bins], censat[sat_bins])
        for E in [bins[["x", "y"]].to_numpy(np.float64)]
        + [e.astype(np.float64) for e in embeds]
    ]
    metrics["umap_seed_semantic_acc"] = [round(a, 4) for a in sem]
    np.savez_compressed(
        OUT / "seed_embeds.npz", **{f"seed{s}": e for s, e in zip(seeds, embeds)}
    )
    print(f"seed stability done t={time.time()-t0:.0f}s")

    # (b) HDBSCAN parameter sweep on the fixed 12-D clustering embedding
    from sklearn.metrics import adjusted_rand_score as ars

    Yc = np.load(OUT / "umap12.npy")
    grid = [(mcs, ms) for mcs in (10, 25, 50, 100) for ms in (5, 10, 20)]
    sweep_labels = []
    for mcs, ms in grid:
        lab, _ = cluster_hdbscan(Yc, params, min_cluster_size=mcs, min_samples=ms)
        sweep_labels.append(lab)
    n = len(grid)
    ari = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            both = (sweep_labels[i] >= 0) & (sweep_labels[j] >= 0)
            ari[i, j] = ari[j, i] = ars(sweep_labels[i][both], sweep_labels[j][both])
    metrics["hdbscan_sweep_median_ari"] = round(float(np.median(ari[np.triu_indices(n, 1)])), 4)
    # agreement at comparable granularity: same mcs with different min_samples,
    # and adjacent mcs at fixed min_samples (the full grid spans 10x cluster
    # scales, where low ARI mostly measures granularity, not instability)
    near = []
    for i in range(n):
        for j in range(i + 1, n):
            (m1, s1), (m2, s2) = grid[i], grid[j]
            if m1 == m2 or (s1 == s2 and abs([10, 25, 50, 100].index(m1) - [10, 25, 50, 100].index(m2)) == 1):
                near.append(ari[i, j])
    metrics["hdbscan_sweep_adjacent_ari"] = round(float(np.mean(near)), 4)
    sweep_meta = pd.DataFrame(grid, columns=["min_cluster_size", "min_samples"])
    sweep_meta["n_clusters"] = [int(l.max() + 1) for l in sweep_labels]
    sweep_meta["noise"] = [float(np.mean(l < 0)) for l in sweep_labels]
    print(f"hdbscan sweep done t={time.time()-t0:.0f}s")

    # (c) FracMinHash subsampling stability: SVD-space kNN overlap across scaled
    bins_raw, indptr, hashes, counts = load_store(params)
    base_knn = knn_indices(Z, k=15)
    sub_overlaps, sub_sat_overlaps, sub_sem = {}, {}, {}
    for scl in (params.embed_scaled * 2, params.embed_scaled * 4):
        ip, h, c = downsample_store(indptr, hashes, counts, scl)
        X, _, _, _ = build_matrix(ip, h, c, params.min_df)
        Zs, _ = gram_rsvd(X, params.svd_dims, seed=params.seed)
        sub_knn = knn_indices(Zs, k=15)
        sub_overlaps[scl] = neighbor_overlap(base_knn, sub_knn)
        sub_sat_overlaps[scl] = neighbor_overlap(base_knn[sat_bins], sub_knn[sat_bins])
        sub_sem[scl] = semantic_acc(Zs[sat_bins], censat[sat_bins], metric="cosine")
        print(f"  scaled={scl} overlap={sub_overlaps[scl]:.3f} t={time.time()-t0:.0f}s")
    metrics["subsample_knn_overlap"] = {str(k): round(v, 4) for k, v in sub_overlaps.items()}
    metrics["subsample_knn_overlap_sat"] = {str(k): round(v, 4) for k, v in sub_sat_overlaps.items()}
    metrics["subsample_semantic_acc"] = {str(k): round(v, 4) for k, v in sub_sem.items()}

    # (d) Track A vs Track B at 1 Mb
    pw = np.load(OUT / "pairwise.npz")
    parents = pd.read_parquet(OUT / "pairwise_bins.parquet")
    _, p_ip, p_h, p_c = merge_to_parents(
        bins_raw, indptr, hashes, counts, params.pairwise_bin_bp
    )
    Xp, _, _, _ = build_matrix(p_ip, p_h, p_c, params.min_df)
    Zp, _ = gram_rsvd(Xp, params.svd_dims, seed=params.seed)
    Zn = Zp / np.maximum(np.linalg.norm(Zp, axis=1, keepdims=True), 1e-9)
    cos_d = 1.0 - Zn @ Zn.T
    d_j = pw["d_jacc"].astype(np.float64)
    iu = np.triu_indices(len(parents), 1)
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    samp = rng.choice(iu[0].size, size=min(300_000, iu[0].size), replace=False)
    rho = spearmanr(cos_d[iu][samp], d_j[iu][samp]).statistic
    metrics["trackA_vs_trackB_spearman"] = round(float(rho), 4)
    # of the 1% of pairs closest by exact Jaccard, how many does the model
    # place in its closest 2%? Rank rho is dragged down by the near-equidistant
    # euchromatic mass; the close pairs are where structure lives.
    j_flat = d_j[iu]
    c_flat = cos_d[iu]
    k_top = max(1, int(0.01 * j_flat.size))
    top_j = np.argpartition(j_flat, k_top)[:k_top]
    c_cut = np.partition(c_flat, int(0.02 * c_flat.size))[int(0.02 * c_flat.size)]
    metrics["ab_close_pair_recall"] = round(float(np.mean(c_flat[top_j] <= c_cut)), 4)
    parent_id = pd.factorize(
        pd.Series(list(zip(bins["chrom"], bins["start"] // params.pairwise_bin_bp)))
    )[0]
    a_child = bins.groupby(parent_id)["cluster"].agg(
        lambda s: s[s >= 0].mode().iloc[0] if (s >= 0).any() else -1
    )
    b_lab = parents["cluster"].to_numpy()
    both = (a_child.to_numpy() >= 0) & (b_lab >= 0)
    metrics["trackA_vs_trackB_ari"] = round(
        float(ars(a_child.to_numpy()[both], b_lab[both])), 4
    )
    np.savez_compressed(
        OUT / "ab_agreement.npz",
        cos_d=cos_d[iu][samp].astype(np.float32),
        d_jacc=d_j[iu][samp].astype(np.float32),
    )
    print(f"A/B agreement done t={time.time()-t0:.0f}s")

    table.to_parquet(OUT / "cluster_table.parquet", index=False)
    np.savez_compressed(
        OUT / "analysis.npz",
        sweep_ari=ari,
        sweep_grid=np.array(grid, np.int32),
        sweep_n_clusters=sweep_meta["n_clusters"].to_numpy(np.int32),
        sweep_noise=sweep_meta["noise"].to_numpy(np.float32),
        **{
            f"conf_{k}": v["conf"] for k, v in results_conf.items()
        },
    )
    with open(OUT / "conf_names.json", "w") as fh:
        json.dump({k: v["names"] for k, v in results_conf.items()}, fh)
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics
