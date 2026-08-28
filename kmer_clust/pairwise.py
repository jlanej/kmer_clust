"""Stage: boosted bins (default 1 Mb) -> exact pairwise sketch distances.

At 1 Mb a bin's FracMinHash (scaled=20) holds tens of thousands of hashes, so
Jaccard and containment between every pair of bins are essentially exact and
cheap: one sparse intersection matmul over the bin x hash incidence matrix.
Distances: 1 - Jaccard and 1 - cANI, with cANI = max-containment^(1/k)
(sourmash's containment-to-ANI point estimate).
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .config import OUT, Params
from .sketch_run import load_store


def merge_to_parents(bins: pd.DataFrame, indptr, hashes, counts, parent_bp: int):
    """Union child sketches into parent bins; counts sum, duplicates collapse."""
    parent_key = pd.Series(list(zip(bins["chrom"], bins["start"] // parent_bp)))
    parent_of_bin = pd.factorize(parent_key)[0]  # order of appearance = genome order
    n_parents = parent_of_bin.max() + 1
    row_parent = np.repeat(parent_of_bin, np.diff(indptr)).astype(np.int64)
    order = np.lexsort((hashes, row_parent))
    b = row_parent[order]
    h = hashes[order]
    c = counts[order].astype(np.int64)
    new = np.empty(b.size, bool)
    new[0] = True
    new[1:] = (b[1:] != b[:-1]) | (h[1:] != h[:-1])
    starts = np.flatnonzero(new)
    c_sum = np.add.reduceat(c, starts)
    b_u, h_u = b[starts], h[starts]
    p_indptr = np.zeros(n_parents + 1, np.int64)
    np.cumsum(np.bincount(b_u, minlength=n_parents), out=p_indptr[1:])

    tmp = bins[["chrom", "start", "end", "acgt"]].copy()
    tmp["gc_bases"] = bins["gc"].to_numpy() * bins["acgt"].to_numpy()
    agg = tmp.groupby(parent_of_bin, sort=False).agg(
        chrom=("chrom", "first"), start=("start", "min"), end=("end", "max"),
        acgt=("acgt", "sum"), gc_bases=("gc_bases", "sum"),
    )
    parents = agg.reset_index(drop=True)
    parents["gc"] = parents["gc_bases"] / np.maximum(parents["acgt"], 1)
    parents = parents.drop(columns=["gc_bases"])
    parents["sketch_size"] = np.diff(p_indptr)
    return parents, p_indptr, h_u, c_sum.astype(np.uint32)


def pairwise_distances(p_indptr, p_hashes, k: int):
    """Exact Jaccard / max-containment / cANI over all parent-bin pairs."""
    universe, inv = np.unique(p_hashes, return_inverse=True)
    n = p_indptr.size - 1
    B = sp.csr_matrix(
        (
            np.ones(p_hashes.size, np.float32),
            inv.astype(np.int32),
            p_indptr.astype(np.int64),
        ),
        shape=(n, universe.size),
    )
    t0 = time.time()
    I = np.asarray((B @ B.T).todense(), dtype=np.float64)
    print(f"  intersections: {n}x{n} in {time.time()-t0:.0f}s")
    sizes = np.diff(p_indptr).astype(np.float64)
    denom_j = sizes[:, None] + sizes[None, :] - I
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = np.where(denom_j > 0, I / denom_j, 0.0)
        cont = np.where(sizes[:, None] > 0, I / sizes[:, None], 0.0)
    maxc = np.maximum(cont, cont.T)
    cani = np.clip(maxc, 0, 1) ** (1.0 / k)
    np.fill_diagonal(jacc, 1.0)
    np.fill_diagonal(cani, 1.0)
    return jacc.astype(np.float32), cani.astype(np.float32), sizes.astype(np.int64)


def run(params: Params) -> None:
    t0 = time.time()
    bins, indptr, hashes, counts = load_store(params)
    parents, p_indptr, p_hashes, p_counts = merge_to_parents(
        bins, indptr, hashes, counts, params.pairwise_bin_bp
    )
    print(
        f"pairwise: {len(parents)} bins of {params.pairwise_bin_bp//1000} kb, "
        f"median sketch {int(np.median(parents['sketch_size']))} hashes"
    )
    jacc, cani, sizes = pairwise_distances(p_indptr, p_hashes, params.k)
    d_ani = (1.0 - cani).astype(np.float32)
    d_jacc = (1.0 - jacc).astype(np.float32)

    cond = squareform(d_ani.astype(np.float64), checks=False)
    L = linkage(cond, method="average")
    L = optimal_leaf_ordering(L, cond)
    leaf_order = leaves_list(L).astype(np.int32)

    from sklearn.cluster import HDBSCAN

    h = HDBSCAN(min_cluster_size=5, min_samples=3, metric="precomputed").fit(
        d_ani.astype(np.float64)
    )
    labels = h.labels_.astype(np.int32)

    import warnings

    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="precomputed",
            random_state=params.seed,
        ).fit_transform(d_ani.astype(np.float64))

    parents["cluster"] = labels
    parents["x"] = xy[:, 0].astype(np.float32)
    parents["y"] = xy[:, 1].astype(np.float32)
    OUT.mkdir(exist_ok=True)
    parents.to_parquet(OUT / "pairwise_bins.parquet", index=False)
    np.savez_compressed(
        OUT / "pairwise.npz",
        d_ani=d_ani, d_jacc=d_jacc, sizes=sizes, leaf_order=leaf_order,
        linkage=L.astype(np.float32),
        meta=np.array([params.k, params.base_scaled, params.pairwise_bin_bp], np.int64),
    )
    n_clust = labels.max() + 1
    print(
        f"pairwise: {n_clust} clusters, {np.mean(labels<0):.1%} noise, "
        f"t={time.time()-t0:.0f}s -> pairwise.npz"
    )
