"""The k-ladder: one embedding per word length, aligned for morphing.

For each k the genome is re-sketched (cached 2-bit codes make this ~2.5 min),
the same matrix -> SVD -> UMAP path runs, and the 2-D layout is Procrustes-
aligned (rotation/reflection/scale) to the k=21 baseline so that morphing
between layouts shows structural change, not UMAP's arbitrary orientation.

Per-k judge-side metrics ship with the views so the atlas slider can display
them live — all under ONE protocol (mainland R^2 on the 12-D embedding,
alpha->chromosome by adjacency-excluded neighbor vote), so chips from
different labs stay comparable.

k=15 is included on purpose: 4^15 is about the genome's own size, so it sits
at the random-collision floor -- the ladder lets you watch that happen.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .analyze import SAT_CLASSES_FOR_PURITY, excluded_neighbor_vote, semantic_acc
from .config import DATA, OUT, Params, PARAMS
from .embed_run import umap_embed
from .matrix import build_matrix, downsample_store, gram_rsvd

KS = [15, 17, 19, 21, 23, 25]


def procrustes_to(base: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Align Y to base by centering, isotropic scale, rotation/reflection."""
    from scipy.linalg import orthogonal_procrustes

    mb, my = base.mean(0), Y.mean(0)
    B = base - mb
    A = Y - my
    sb = np.linalg.norm(B)
    sa = np.linalg.norm(A) or 1.0
    R, _ = orthogonal_procrustes(A / sa, B / sb)
    return (A / sa) @ R * sb + mb


def ensure_sketch(k: int) -> Params:
    """Sketch the genome at word length k if its store is absent.

    write_bins=False keeps the canonical bin-stats parquet untouched — the
    multi-k labs only need the sketch store, and the parquet's sketch_size
    columns must stay those of the default k."""
    p = Params(k=k)
    if not p.sketch_npz.exists():
        from . import sketch_run

        sketch_run.run(p, write_bins=False)
    return p


def run(params: Params = PARAMS) -> None:
    t0 = time.time()
    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    censat = annot["censat_class"].replace("", "non_sat").to_numpy()
    sat = np.isin(censat, SAT_CLASSES_FOR_PURITY)
    eu = (~sat) & (annot["ct_frac"].to_numpy() < 0.5)
    chroms = bins["chrom"].to_numpy()
    alpha_mask = (annot["cov_asat_hor_live"] + annot["cov_asat_hor"]).to_numpy() >= 0.5
    COV = {
        "gc": bins["gc"].to_numpy(np.float64),
        "alu": annot["rm_alu"].to_numpy(np.float64),
        "l1": annot["rm_l1"].to_numpy(np.float64),
        "sd": annot["sd_frac"].to_numpy(np.float64),
    }

    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    starts = bins["start"].to_numpy()

    def eval_layout(xy, Z, y12):
        r = {"sat_sem_2d": round(semantic_acc(xy[sat].astype(np.float64), censat[sat]), 4)}
        keep = pd.Series(chroms[alpha_mask]).value_counts()
        keep = keep[keep >= 5].index
        m = alpha_mask & np.isin(chroms, keep)
        r["alpha_chrom"] = round(excluded_neighbor_vote(
            Z[m], chroms[m], chroms[m], starts[m], params.bin_bp, k=10)[0], 4)
        for name, v in COV.items():
            s = cross_val_score(
                KNeighborsRegressor(n_neighbors=15), y12[eu], v[eu], cv=5, scoring="r2"
            ).mean()
            r[f"r2_{name}"] = round(float(s), 4)
        r["mainland_score"] = round(
            np.mean([r["r2_alu"], r["r2_l1"], r["r2_sd"], r["r2_gc"]]), 4
        )
        return r

    layouts, metrics = {}, {}
    for k in KS:
        t1 = time.time()
        pk = ensure_sketch(k)
        z = np.load(pk.sketch_npz)
        ip, h, c = downsample_store(z["indptr"], z["hashes"], z["counts"], params.embed_scaled)
        X, _, _, _ = build_matrix(ip, h, c, params.min_df)
        Z, _ = gram_rsvd(X, params.svd_dims, seed=params.seed)
        xy = umap_embed(Z, 2, params)
        y12 = umap_embed(Z, 12, params, min_dist=0.0)
        layouts[k] = xy
        metrics[k] = eval_layout(xy, Z, y12)
        metrics[k]["t_s"] = int(time.time() - t1)
        print(f"k={k}: {json.dumps(metrics[k])}", flush=True)

    base = layouts[21]
    out = {}
    for k in KS:
        out[f"k{k}"] = procrustes_to(base, layouts[k]).astype(np.float32)
    # align the dual-vocabulary view from the structure lab onto the same frame
    lab_npz = OUT / "structure_lab.npz"
    if lab_npz.exists():
        store = np.load(lab_npz)
        if "concat_xy" in store:
            out["concat"] = procrustes_to(base, store["concat_xy"].astype(np.float64)).astype(
                np.float32
            )
    np.savez_compressed(OUT / "kladder.npz", **out)
    with open(OUT / "kladder.json", "w") as fh:
        json.dump({"ks": KS, "metrics": {str(k): v for k, v in metrics.items()}}, fh, indent=2)
    print(f"k-ladder complete t={time.time()-t0:.0f}s -> kladder.npz")


if __name__ == "__main__":
    run()
