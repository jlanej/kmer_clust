"""Multi-k vocabulary lab: which combination of horizons wins?

Candidates are concatenations of per-k SVD blocks (each whitened alpha=0.35,
L2 row-normalized, so the concat cosine is the mean of per-k cosines).
Evaluated with structure_lab.eval-style metrics for direct comparability.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from kmer_clust.analyze import SAT_CLASSES_FOR_PURITY, knn_cv_accuracy, semantic_acc
from kmer_clust.config import OUT, Params, PARAMS
from kmer_clust.embed_run import cluster_hdbscan, umap_embed
from kmer_clust.matrix import build_matrix, downsample_store, gram_rsvd

p = PARAMS
t0 = time.time()
bins = pd.read_parquet(OUT / "bins_embedded.parquet")
annot = pd.read_parquet(OUT / f"annot_{p.bin_bp}.parquet")
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
ALPHA = 0.35
KS = [15, 17, 19, 21, 23, 25]

def l2rows(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)

blocks = {}
for k in KS:
    pk = Params(k=k)
    z = np.load(pk.sketch_npz)
    ip, h, c = downsample_store(z["indptr"], z["hashes"], z["counts"], p.embed_scaled)
    X, _, _, _ = build_matrix(ip, h, c, p.min_df)
    Z, sig = gram_rsvd(X, p.svd_dims, seed=p.seed)
    s = np.where(sig > 0, sig, 1.0)
    blocks[k] = l2rows((Z.astype(np.float64) / s[None, :] ** ALPHA)).astype(np.float32)
    print(f"block k={k} ready t={time.time()-t0:.0f}s", flush=True)

def eval_embedding(tag, Zf):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    t1 = time.time()
    xy = umap_embed(Zf, 2, p)
    y12 = umap_embed(Zf, 12, p, min_dist=0.0)
    lab, _ = cluster_hdbscan(y12, p)
    r = {"tag": tag, "n_clusters": int(lab.max() + 1),
         "noise": round(float((lab < 0).mean()), 4)}
    r["sat_sem_2d"] = round(semantic_acc(xy[sat].astype(np.float64), censat[sat]), 4)
    keep = pd.Series(chroms[alpha_mask]).value_counts()
    keep = keep[keep >= 5].index
    m = alpha_mask & np.isin(chroms, keep)
    r["alpha_chrom"] = round(knn_cv_accuracy(Zf[m], chroms[m], k=10)[0], 4)
    for name, v in COV.items():
        s = cross_val_score(KNeighborsRegressor(n_neighbors=15), y12[eu], v[eu],
                            cv=5, scoring="r2").mean()
        r[f"r2_{name}"] = round(float(s), 4)
    r["mainland_score"] = round(np.mean([r["r2_alu"], r["r2_l1"], r["r2_sd"], r["r2_gc"]]), 4)
    r["t_s"] = int(time.time() - t1)
    print(json.dumps(r), flush=True)
    return r, xy

CANDS = {
    "k17+k21 (shipped)": [17, 21],
    "k15+k21": [15, 21],
    "k15+k25 ends": [15, 25],
    "k15+k21+k25": [15, 21, 25],
    "all six 15..25": KS,
}
results, layouts = [], {}
for tag, ks in CANDS.items():
    Zc = np.hstack([blocks[k] for k in ks]).astype(np.float32)
    r, xy = eval_embedding(tag, Zc)
    r["ks"] = ks
    results.append(r)
    layouts[tag] = xy

np.savez_compressed(OUT / "multik_lab.npz",
                    **{t.replace(" ", "_"): v for t, v in layouts.items()})
with open(OUT / "multik_lab.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"multik lab complete t={time.time()-t0:.0f}s", flush=True)

# align the information view (k15+k21) onto the ladder's k21 frame and store
# it in kladder.npz so the site picks it up as a view chip
kl_path = OUT / "kladder.npz"
if kl_path.exists() and "k15+k21" in layouts:
    from kmer_clust.kladder import procrustes_to

    store = dict(np.load(kl_path))
    if "k21" in store:
        store["duo1521"] = procrustes_to(
            store["k21"].astype(np.float64), layouts["k15+k21"].astype(np.float64)
        ).astype(np.float32)
        np.savez_compressed(kl_path, **store)
        print("duo1521 view aligned into kladder.npz")
