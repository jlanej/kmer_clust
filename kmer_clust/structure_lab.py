"""Structure lab: annotation-free levers for articulating the euchromatic
mainland, judged (never built) by annotation, with satellite health as a floor.

Candidates:
  a000/a035/a070/a100 : same k=21 SVD, components scaled by 1/sigma^alpha
  k17                 : fresh k=17 vocabulary, alpha=best from sweep
  concat              : L2-normalized [k21, k17] blocks side by side
"""

import json
import shutil
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from kmer_clust.analyze import SAT_CLASSES_FOR_PURITY, knn_cv_accuracy, semantic_acc
from kmer_clust.config import DATA, OUT, Params, PARAMS
from kmer_clust.embed_run import cluster_hdbscan, umap_embed
from kmer_clust.matrix import build_matrix, downsample_store, gram_rsvd

p = PARAMS
t0 = time.time()
bins = pd.read_parquet(OUT / "bins_embedded.parquet")
annot = pd.read_parquet(OUT / f"annot_{p.bin_bp}.parquet")
z = np.load(OUT / f"svd_s{p.embed_scaled}.npz")
Z21, sig21 = z["Z"].astype(np.float64), z["sigma"].astype(np.float64)

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


def whiten(Z, sigma, alpha):
    s = np.where(sigma > 0, sigma, 1.0)
    return (Z / s[None, :] ** alpha).astype(np.float32)


def alpha_chrom_acc(Zf):
    keep = pd.Series(chroms[alpha_mask]).value_counts()
    keep = keep[keep >= 5].index
    m = alpha_mask & np.isin(chroms, keep)
    return knn_cv_accuracy(Zf[m], chroms[m], k=10)[0]


def eval_embedding(tag, Zf, store):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    t1 = time.time()
    xy = umap_embed(Zf, 2, p)
    y12 = umap_embed(Zf, 12, p, min_dist=0.0)
    lab, _ = cluster_hdbscan(y12, p)
    r = {"tag": tag, "n_clusters": int(lab.max() + 1), "noise": round(float((lab < 0).mean()), 4)}
    # satellite health (hard floor)
    r["sat_sem_2d"] = round(semantic_acc(xy[sat].astype(np.float64), censat[sat]), 4)
    r["alpha_chrom"] = round(alpha_chrom_acc(Zf), 4)
    # mainland organization: embedding-geometry R2 (kNN regression, 12-d, CV)
    for name, v in COV.items():
        s = cross_val_score(
            KNeighborsRegressor(n_neighbors=15), y12[eu], v[eu], cv=5, scoring="r2"
        ).mean()
        r[f"r2_{name}"] = round(float(s), 4)
    # cluster-explained variance on clustered euchromatic bins
    m = eu & (lab >= 0)
    for name, v in COV.items():
        vv = v[m]
        groups = pd.Series(vv).groupby(lab[m])
        ssw = float(((vv - groups.transform("mean")) ** 2).sum())
        sst = float(((vv - vv.mean()) ** 2).sum())
        r[f"cr2_{name}"] = round(1 - ssw / max(sst, 1e-9), 4)
    n_eu_cl = len({c for c in np.unique(lab[m]) if c >= 0})
    r["eu_clusters"] = n_eu_cl
    r["mainland_score"] = round(np.mean([r["r2_alu"], r["r2_l1"], r["r2_sd"], r["r2_gc"]]), 4)
    r["t_s"] = int(time.time() - t1)
    store[f"{tag}_xy"] = xy.astype(np.float32)
    store[f"{tag}_lab"] = lab.astype(np.int16)
    print(json.dumps(r), flush=True)
    return r


results, store = [], {}
for a in (0.0, 0.35, 0.7, 1.0):
    results.append(eval_embedding(f"a{int(a*100):03d}", whiten(Z21, sig21, a), store))

base = results[0]
ok = [
    r for r in results
    if r["sat_sem_2d"] >= base["sat_sem_2d"] - 0.02
    and r["alpha_chrom"] >= base["alpha_chrom"] - 0.02
]
best = max(ok, key=lambda r: r["mainland_score"])
a_best = int(best["tag"][1:]) / 100
print(f"## best alpha = {a_best} ({best['tag']})", flush=True)

# ---- k=17 vocabulary ------------------------------------------------------
p17 = Params(k=17)
if not p17.sketch_npz.exists():
    bak = DATA / "bins_100000.parquet.bak"
    shutil.copy(p17.bins_parquet, bak)
    from kmer_clust import sketch_run

    sketch_run.run(p17)
    shutil.move(bak, p17.bins_parquet)  # keep k=21 bin stats canonical
zz = np.load(p17.sketch_npz)
ip, h, c = downsample_store(zz["indptr"], zz["hashes"], zz["counts"], p.embed_scaled)
X17, _, _, _ = build_matrix(ip, h, c, p.min_df)
Z17, sig17 = gram_rsvd(X17, p.svd_dims, seed=p.seed)
Z17, sig17 = Z17.astype(np.float64), sig17.astype(np.float64)
print(f"## k17 svd done t={time.time()-t0:.0f}s", flush=True)

results.append(eval_embedding("k17", whiten(Z17, sig17, a_best), store))


def l2rows(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)


concat = np.hstack([
    l2rows(whiten(Z21, sig21, a_best)), l2rows(whiten(Z17, sig17, a_best))
]).astype(np.float32)
results.append(eval_embedding("concat", concat, store))

np.savez_compressed(OUT / "structure_lab.npz", **store)
with open(OUT / "structure_lab.json", "w") as fh:
    json.dump({"results": results, "a_best": a_best}, fh, indent=2)
print(f"## structure lab complete t={time.time()-t0:.0f}s", flush=True)
