"""Stage: project new sequence onto the frozen CHM13 word-space (prototype).

The map's model is frozen: the shared-hash universe, the IDF weights, and the
SVD basis. Any 100 kb window of any assembly can then be sketched with the
same lottery, matched against that universe, weighted the same way, and
dropped into the same 128-d space — no aligner, no coordinates. Its nearest
T2T bins by cosine are its "hits"; adjacent hits merge into loci, so a window
that belongs to several places (the acrocentric commons) says so honestly.

Validation is self-projection: T2T windows offset by 50 kb (so no query
equals any training bin) must find their own overlapping bins.
"""

import json
import time

import numpy as np
import pandas as pd

from .config import DATA, OUT, Params, PARAMS
from .fasta import iter_chrom_codes, iter_fasta_codes
from .fracminhash import sketch_codes
from .matrix import build_matrix, downsample_store
from .sketch_run import load_store

KMER_DUST_TESTDATA = DATA.parent.parent / "kmer_dust" / "data" / "testdata"
QUERY_SETS = [
    ("chm13_slice", "CHM13 chr21 slice (control)", "chm13v2.0.fa.gz"),
    ("hg00097_h1", "HG00097 hap1 (HPRC)", "HG00097_hap1_hprc_r2_v1.0.1.fa.gz"),
    ("hg00097_h2", "HG00097 hap2 (HPRC)", "HG00097_hap2_hprc_r2_v1.0.1.fa.gz"),
    ("na19909_h2", "NA19909 hap2 (HPRC)", "NA19909_hap2_hprc_r2_v1.0.1.fa.gz"),
]
TOP_HITS = 8


class Kit:
    """Frozen projection operators.

    Two signals, on purpose: the MODEL (shared-vocabulary SVD space) places a
    query in word-space — the dialect neighborhood — while EXACT sketch
    Jaccard against the full store (private words included) decides which
    locus it is. Euchromatic locus identity lives in the private vocabulary
    the model deliberately excludes, so loci must not rely on cosine alone.
    """

    def __init__(self, params: Params):
        import scipy.sparse as sp

        t0 = time.time()
        self.params = params
        bins, indptr, hashes, counts = load_store(params)
        ip, h, c = downsample_store(indptr, hashes, counts, params.embed_scaled)
        # exact side: every bin's full sketch (private words included)
        self.universe_full, inv = np.unique(h, return_inverse=True)
        self.Bfull = sp.csr_matrix(
            (np.ones(h.size, np.float32), inv.astype(np.int32), ip.astype(np.int64)),
            shape=(len(bins), self.universe_full.size),
        ).tocsc()
        self.bin_sizes = np.diff(ip).astype(np.float64)
        # model side: shared-vocabulary basis
        X, universe, df, _ = build_matrix(ip, h, c, params.min_df)
        self.universe = universe
        self.idf = np.log1p(X.shape[0] / df.astype(np.float64)).astype(np.float32)
        z = np.load(params.svd_npz())
        Z, sigma = z["Z"].astype(np.float64), z["sigma"].astype(np.float64)
        W = X.T.tocsr() @ (Z / np.maximum(sigma, 1e-9) ** 2)  # V = X^T Z Sigma^-2
        self.V = W.astype(np.float32)
        self.Zn = (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)).astype(
            np.float32
        )
        self.bins = bins
        print(
            f"kit ready: shared universe {universe.size/1e6:.2f}M, "
            f"full universe {self.universe_full.size/1e6:.1f}M, t={time.time()-t0:.0f}s"
        )

    def project_window(self, codes: np.ndarray):
        """codes (2-bit) -> model hits (cosine), exact hits (Jaccard), coverage."""
        import scipy.sparse as sp

        p = self.params
        _, hs = sketch_codes(codes, p.k, p.embed_scaled)
        if hs.size < 10:
            return None
        uniq, cts = np.unique(hs, return_counts=True)

        # exact locus evidence: Jaccard vs every bin's full sketch
        posf = np.searchsorted(self.universe_full, uniq)
        okf = (posf < self.universe_full.size) & (
            self.universe_full[np.minimum(posf, self.universe_full.size - 1)] == uniq
        )
        cover = float(okf.sum() / uniq.size)
        qv = sp.csc_matrix(
            (np.ones(int(okf.sum()), np.float32),
             (posf[okf].astype(np.int64), np.zeros(int(okf.sum()), np.int64))),
            shape=(self.universe_full.size, 1),
        )
        inter = np.asarray((self.Bfull @ qv).todense()).ravel()
        jacc = inter / np.maximum(self.bin_sizes + uniq.size - inter, 1.0)
        etop = np.argpartition(-jacc, TOP_HITS)[:TOP_HITS]
        etop = etop[np.argsort(-jacc[etop])]

        # model placement: shared-vocabulary cosine in SVD space
        pos = np.searchsorted(self.universe, uniq)
        ok = (pos < self.universe.size) & (
            self.universe[np.minimum(pos, self.universe.size - 1)] == uniq
        )
        hits, sims = [], []
        if ok.sum() >= 5:
            w = np.zeros(self.V.shape[0], np.float32)
            w[pos[ok]] = np.log1p(cts[ok]).astype(np.float32) * self.idf[pos[ok]]
            zq = (w / max(np.linalg.norm(w), 1e-9)) @ self.V
            zq /= max(np.linalg.norm(zq), 1e-9)
            csims = self.Zn @ zq
            top = np.argpartition(-csims, TOP_HITS)[:TOP_HITS]
            top = top[np.argsort(-csims[top])]
            hits = top.tolist()
            sims = [round(float(csims[i]), 4) for i in top]
        return {
            "cover": cover, "hits": hits, "sims": sims,
            "ehits": etop.tolist(),
            "ejacc": [round(float(jacc[i]), 4) for i in etop],
        }


def loci_of(kit: Kit, hits, sims, gap_bins=3, max_loci=3):
    """Merge genomically adjacent hits into loci — transitively, so one
    contiguous array forms ONE locus rather than fragments that would evict
    genuinely different loci from the max_loci slots. Loci are ordered by
    their best hit's score."""
    rows = kit.bins.iloc[hits].reset_index(drop=True)
    order = sorted(range(len(hits)),
                   key=lambda m: (rows["chrom"][m], int(rows["start"][m])))
    chains = [[order[0]]]
    for m in order[1:]:
        prev = chains[-1][-1]
        if (rows["chrom"][m] == rows["chrom"][prev]
                and rows["start"][m] - rows["start"][prev] <= gap_bins * kit.params.bin_bp):
            chains[-1].append(m)
        else:
            chains.append([m])
    loci = []
    for members in chains:
        best = max(sims[m] for m in members)
        loci.append({
            "chrom": rows["chrom"][members[0]],
            "start_mb": round(float(min(rows["start"][m] for m in members) / 1e6), 1),
            "end_mb": round(float(max(rows["end"][m] for m in members) / 1e6), 1),
            "sim": best,
            "bins": [hits[m] for m in members],
            "bins_j": [[hits[m], sims[m]] for m in members],
        })
    loci.sort(key=lambda l: -l["sim"])
    return loci[:max_loci]


def selftest(kit: Kit, n_easy=100, n_hard=60, offset=50_000, seed=7):
    rng = np.random.default_rng(seed)
    annot = pd.read_parquet(OUT / f"annot_{kit.params.bin_bp}.parquet")
    bins = kit.bins
    sat = (annot["censat_class"] != "").to_numpy() | annot["acro_p"].to_numpy()
    ok_start = (bins["start"] + offset + kit.params.bin_bp).to_numpy() <= bins.groupby(
        "chrom")["end"].transform("max").to_numpy()
    easy_pool = np.flatnonzero(~sat & ok_start)
    hard_pool = np.flatnonzero(sat & ok_start)
    picks = {
        "easy": rng.choice(easy_pool, size=min(n_easy, easy_pool.size), replace=False),
        "hard": rng.choice(hard_pool, size=min(n_hard, hard_pool.size), replace=False),
    }
    by_chrom = {}
    for grp, ids in picks.items():
        for b in ids:
            by_chrom.setdefault(bins.iloc[b]["chrom"], []).append((grp, int(b)))
    results = {"easy": [], "hard": []}
    for chrom, codes in iter_chrom_codes(kit.params):
        if chrom not in by_chrom:
            del codes
            continue
        for grp, b in by_chrom[chrom]:
            s = int(bins.iloc[b]["start"]) + offset
            r = kit.project_window(codes[s : s + kit.params.bin_bp])
            if not r or not r["ehits"]:
                continue
            truth = {b, b + 1}  # the two bins the offset window overlaps
            results[grp].append({
                "top1_ok": r["ehits"][0] in truth,
                "top3_ok": any(h in truth for h in r["ehits"][:3]),
                "cos1_ok": bool(r["hits"]) and r["hits"][0] in truth,
                "jacc": r["ejacc"][0],
                "cover": round(r["cover"], 3),
            })
        del codes
    out = {}
    for grp, rows in results.items():
        n = len(rows)
        out[grp] = {
            "n": n,
            "top1_acc": round(sum(x["top1_ok"] for x in rows) / max(n, 1), 4),
            "top3_acc": round(sum(x["top3_ok"] for x in rows) / max(n, 1), 4),
            "cosine_top1_acc": round(sum(x["cos1_ok"] for x in rows) / max(n, 1), 4),
            "median_jacc": round(float(np.median([x["jacc"] for x in rows])), 4),
            "median_cover": round(float(np.median([x["cover"] for x in rows])), 4),
        }
    return out


def project_set(kit: Kit, path, window=None):
    window = window or kit.params.bin_bp
    out = []
    for name, codes in iter_fasta_codes(path):
        n_win = codes.size // window
        for wdx in range(n_win):
            r = kit.project_window(codes[wdx * window : (wdx + 1) * window])
            if not r:
                continue
            entry = {
                "label": f"{name}:{wdx*window/1e6:.1f}–{(wdx+1)*window/1e6:.1f} Mb",
                "pos_mb": round(wdx * window / 1e6, 2),
                "cover": round(r["cover"], 3),
            }
            if r["ehits"]:
                # word-space placement from the model; loci from exact Jaccard
                entry["hits"] = [[int(h), s] for h, s in zip(r["hits"], r["sims"])] or \
                    [[int(h), s] for h, s in zip(r["ehits"], r["ejacc"])]
                entry["loci"] = loci_of(kit, r["ehits"], r["ejacc"])
            out.append(entry)
    return out


def run(params: Params = PARAMS) -> None:
    t0 = time.time()
    kit = Kit(params)
    st = selftest(kit)
    print("selftest:", json.dumps(st))
    sets = []
    for sid, label, fname in QUERY_SETS:
        path = KMER_DUST_TESTDATA / fname
        if not path.exists():
            print(f"  ({fname} not found; skipping)")
            continue
        windows = project_set(kit, path)
        sets.append({"id": sid, "label": label, "windows": windows})
        med = np.median([w["hits"][0][1] for w in windows if "hits" in w])
        print(f"  {label}: {len(windows)} windows, median top-1 sim {med:.3f}")
    with open(OUT / "projection.json", "w") as fh:
        json.dump({"selftest": st, "sets": sets,
                   "params": {"k": params.k, "scaled": params.embed_scaled,
                              "window_bp": params.bin_bp, "top_hits": TOP_HITS}}, fh)
    print(f"projection -> out/projection.json, t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
