"""Stage: SVD coords -> UMAP (2-D for display, higher-D for clustering) -> HDBSCAN."""

import time
import warnings

import numpy as np
import pandas as pd

from .config import OUT, Params


def umap_embed(Z, n_components, params: Params, min_dist=None, seed=None):
    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return umap.UMAP(
            n_neighbors=params.umap_neighbors,
            min_dist=params.umap_min_dist if min_dist is None else min_dist,
            n_components=n_components,
            metric="cosine",
            random_state=params.seed if seed is None else seed,
        ).fit_transform(Z)


def cluster_hdbscan(Y, params: Params, min_cluster_size=None, min_samples=None):
    from sklearn.cluster import HDBSCAN

    h = HDBSCAN(
        min_cluster_size=min_cluster_size or params.hdbscan_min_cluster_size,
        min_samples=min_samples or params.hdbscan_min_samples,
    ).fit(Y)
    return h.labels_, h.probabilities_


def run(params: Params) -> pd.DataFrame:
    t0 = time.time()
    z = np.load(params.svd_npz())
    Z = z["Z"]
    bins = pd.read_parquet(params.bins_parquet)  # the 1.7 GB store is not needed here
    print(f"umap: {Z.shape[0]} bins from {Z.shape[1]}-d SVD")
    xy = umap_embed(Z, 2, params)
    print(f"  2-d done t={time.time()-t0:.0f}s")
    Yc = umap_embed(Z, 12, params, min_dist=0.0)
    print(f"  12-d done t={time.time()-t0:.0f}s")
    labels, probs = cluster_hdbscan(Yc, params)
    n_clust = labels.max() + 1
    noise = float(np.mean(labels < 0))
    print(f"hdbscan: {n_clust} clusters, {noise:.1%} noise, t={time.time()-t0:.0f}s")

    bins = bins.copy()
    bins["x"] = xy[:, 0].astype(np.float32)
    bins["y"] = xy[:, 1].astype(np.float32)
    bins["cluster"] = labels.astype(np.int32)
    bins["cluster_prob"] = probs.astype(np.float32)
    bins["private_frac"] = z["private_frac"]
    out = OUT / "bins_embedded.parquet"
    bins.to_parquet(out, index=False)
    np.save(OUT / "umap12.npy", Yc.astype(np.float32))
    print(f"-> {out.name}")
    return bins
