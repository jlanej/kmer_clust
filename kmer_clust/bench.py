"""Benchmarks that turn the projection/triage claims into numbers.

1. Misassembly detection (run_misassembly): synthetic contigs cut from
   CHM13 itself (offset 50 kb from the bin grid, so no window equals a
   training bin), corrupted with the classic draft-assembly errors —
   an inverted middle block, an interchromosomal misjoin, a distant
   intrachromosomal misjoin — then read back through the projector.
   Detection statistics are alignment-free: order jumps between adjacent
   window placements, and the most-negative local (sliding-window) tau.
   Reported per stratum (euchromatic / satellite) as ROC AUC plus a
   fixed-operating-point table against intact controls.

2. Ultralong-read recruitment (run_reads): fragments of CHM13 at read
   lengths 50/100/200 kb with uniform substitution errors injected at
   0/2/5% — the ONT regime — placed by the two-signal projector. Truth =
   the fragment's overlapped bins. Reports top-1 locus accuracy, median
   exact J, and measured novelty (error k-mers read as novel vocabulary,
   expectation 1-(1-p)^k).

Run: python -m kmer_clust.bench [mis|reads|all]
Outputs: out/bench_misassembly.json, out/bench_reads.json
"""

import json
import time

import numpy as np
import pandas as pd

from .config import OUT, Params, PARAMS
from .fasta import iter_chrom_codes
from .project import Kit

RNG_SEED = 7
SLIDE = 7          # windows per local-tau span
JUMP_MB = 5.0      # adjacent-placement leap that counts as an order jump
INV_TAU = -0.7     # local-tau operating point for inversion flags
WIN = PARAMS.bin_bp  # windows mirror the store's bin size


# ---------------------------------------------------------------- helpers
def revcomp_codes(c: np.ndarray) -> np.ndarray:
    return np.where(c < 4, 3 - c, 4)[::-1].copy()


def inject_errors(c: np.ndarray, p: float, rng) -> np.ndarray:
    """Uniform substitutions at rate p (ACGT positions only)."""
    if p <= 0:
        return c
    c = c.copy()
    ok = np.flatnonzero(c < 4)
    hit = ok[rng.random(ok.size) < p]
    c[hit] = (c[hit] + 1 + rng.integers(0, 3, hit.size)) % 4
    return c


def order_stats(chrom_idx, mb, tie=0.01):
    """(jumps, min local tau, global tau) over placed windows in order.

    Local tau spans only same-chromosome stretches; cross-chromosome
    adjacency is what `jumps` counts."""
    n = len(chrom_idx)
    jumps = 0
    for k in range(1, n):
        if chrom_idx[k] != chrom_idx[k - 1] or abs(mb[k] - mb[k - 1]) > JUMP_MB:
            jumps += 1

    def tau(vals):
        m = len(vals)
        if m < 2:
            return 0.0
        S = sum(np.sign(vals[j] - vals[i])
                if abs(vals[j] - vals[i]) > tie else 0
                for i in range(m) for j in range(i + 1, m))
        return float(S / (m * (m - 1) / 2))

    local = []
    for k in range(n - SLIDE + 1):
        ci = chrom_idx[k:k + SLIDE]
        if (ci == ci[0]).all():
            local.append(tau(list(mb[k:k + SLIDE])))
    return jumps, (min(local) if local else 0.0), tau(list(mb))


def _auc(neg, pos):
    """Rank-based AUC of score separating pos from neg."""
    xs = np.concatenate([neg, pos])
    ys = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(xs, kind="mergesort")
    ranks = np.empty(len(xs))
    # average ranks for ties
    sx = xs[order]
    r, i = np.empty(len(xs)), 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = r
    n1, n0 = ys.sum(), (1 - ys).sum()
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Genome:
    """Cached CHM13 codes plus bin bookkeeping for truth and strata."""

    def __init__(self, kit: Kit, params: Params):
        self.params = params
        self.codes = dict(iter_chrom_codes(params))
        bins = kit.bins
        self.chroms = list(dict.fromkeys(bins["chrom"]))
        self.chrom_idx_of_bin = np.array(
            [self.chroms.index(c) for c in bins["chrom"]])
        self.mb_of_bin = bins["start"].to_numpy() / 1e6 + WIN / 2e6
        self.bin0 = {}
        i = 0
        for c in self.chroms:
            self.bin0[c] = i
            i += int((bins["chrom"] == c).sum())
        annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
        self.sat = (annot["censat_class"] != "").to_numpy()

    def global_bin(self, chrom, pos):
        return self.bin0[chrom] + int(pos // WIN)

    def sample_span(self, rng, n_win, stratum):
        """Random (chrom, start) for an n_win-window span, offset 50 kb
        from the bin grid, whose bins are >=60% satellite (stratum='sat')
        or fully non-satellite ('eu')."""
        chroms = [c for c in self.chroms if self.codes[c].size > (n_win + 2) * WIN]
        for _ in range(4000):
            c = chroms[rng.integers(len(chroms))]
            b = rng.integers(0, self.codes[c].size // WIN - n_win - 1)
            start = b * WIN + WIN // 2
            g0 = self.global_bin(c, start)
            sf = self.sat[g0:g0 + n_win + 1].mean()
            if (stratum == "sat") == (sf >= 0.6) and (stratum == "sat" or sf == 0):
                return c, start
        raise RuntimeError(f"no span found for {stratum}")

    def truth_bins(self, chrom, pos):
        g = self.global_bin(chrom, pos)
        return {g, g + 1}


def project_fragment(kit, codes):
    """Per-window top-1 placements for a fragment (list of (bin, J))."""
    out = []
    for w in range(codes.size // WIN):
        r = kit.project_window(codes[w * WIN:(w + 1) * WIN])
        if r and r["ehits"]:
            out.append((r["ehits"][0], r["ejacc"][0], r["cover"]))
    return out


# ------------------------------------------------------- 1. misassembly
def run_misassembly(kit, G, n_per=40, n_win=30, params=PARAMS):
    rng = np.random.default_rng(RNG_SEED)
    types = ["control", "inversion", "misjoin_inter", "misjoin_intra"]
    rows = []
    t0 = time.time()
    for stratum in ("eu", "sat"):
        for typ in types:
            for _ in range(n_per):
                c, s = G.sample_span(rng, n_win, stratum)
                frag = G.codes[c][s:s + n_win * WIN].copy()
                if typ == "inversion":
                    a, b = (n_win // 3) * WIN, (2 * n_win // 3) * WIN
                    frag[a:b] = revcomp_codes(frag[a:b])
                elif typ == "misjoin_inter":
                    while True:
                        c2, s2 = G.sample_span(rng, n_win, stratum)
                        if c2 != c:
                            break
                    h = (n_win // 2) * WIN
                    frag = np.concatenate([frag[:h], G.codes[c2][s2:s2 + (n_win * WIN - h)]])
                elif typ == "misjoin_intra":
                    for _ in range(200):
                        s2 = rng.integers(0, G.codes[c].size // WIN - n_win) * WIN + WIN // 2
                        if abs(s2 - s) > 20e6:
                            break
                    h = (n_win // 2) * WIN
                    frag = np.concatenate([frag[:h], G.codes[c][s2:s2 + (n_win * WIN - h)]])
                pl = project_fragment(kit, frag)
                ci = np.array([G.chrom_idx_of_bin[b] for b, _, _ in pl])
                mb = np.array([G.mb_of_bin[b] for b, _, _ in pl])
                jumps, mloc, gtau = order_stats(ci, mb)
                rows.append({"stratum": stratum, "type": typ, "jumps": jumps,
                             "min_local_tau": round(mloc, 3),
                             "tau": round(gtau, 3),
                             "j_med": round(float(np.median([j for _, j, _ in pl])), 3)})
    print(f"  {len(rows)} synthetic contigs in {time.time()-t0:.0f}s")

    res = {"n_per": n_per, "n_win": n_win, "rows": rows, "summary": {}}
    for stratum in ("eu", "sat"):
        S = [r for r in rows if r["stratum"] == stratum]
        ctrl = [r for r in S if r["type"] == "control"]
        summ = {}
        for typ in types[1:]:
            P = [r for r in S if r["type"] == typ]
            score = (lambda r: (1 - r["min_local_tau"]) / 2) if typ == "inversion" \
                else (lambda r: r["jumps"])
            auc = _auc(np.array([score(r) for r in ctrl], float),
                       np.array([score(r) for r in P], float))
            if typ == "inversion":
                det = np.mean([r["min_local_tau"] <= INV_TAU for r in P])
                fpr = np.mean([r["min_local_tau"] <= INV_TAU for r in ctrl])
            else:
                det = np.mean([r["jumps"] >= 1 for r in P])
                fpr = np.mean([r["jumps"] >= 1 for r in ctrl])
            summ[typ] = {"auc": round(auc, 3), "detected": round(float(det), 3),
                         "control_fpr": round(float(fpr), 3)}
        res["summary"][stratum] = summ
        print(f"  [{stratum}] " + "  ".join(
            f"{t}: AUC {v['auc']} det {v['detected']:.0%} (FPR {v['control_fpr']:.0%})"
            for t, v in summ.items()))
    with open(OUT / "bench_misassembly.json", "w") as fh:
        json.dump(res, fh)
    return res


# ---------------------------------------------------------- 2. UL reads
def run_reads(kit, G, n_per=50, params=PARAMS):
    rng = np.random.default_rng(RNG_SEED + 1)
    lengths = [50_000, 100_000, 200_000]
    errs = [0.0, 0.02, 0.05]
    cells = []
    t0 = time.time()
    for stratum in ("eu", "sat"):
        for L in lengths:
            for p in errs:
                accs, chrom_accs, js, covs = [], [], [], []
                for _ in range(n_per):
                    nw = max(L // WIN, 1)
                    c, s = G.sample_span(rng, max(nw, 1) + 1, stratum)
                    frag = G.codes[c][s:s + L].copy()
                    if rng.random() < 0.5:
                        frag = revcomp_codes(frag)  # random strand
                    frag = inject_errors(frag, p, rng)
                    r = kit.project_window(frag)
                    if not r or not r["ehits"]:
                        accs.append(0.0)
                        chrom_accs.append(0.0)
                        continue
                    truth = set()
                    for off in range(0, L, WIN):
                        truth |= G.truth_bins(c, s + off)
                    top = r["ehits"][0]
                    accs.append(float(top in truth))
                    chrom_accs.append(float(
                        G.chrom_idx_of_bin[top] == G.chroms.index(c)))
                    js.append(r["ejacc"][0])
                    covs.append(r["cover"])
                cells.append({"stratum": stratum, "len_kb": L // 1000,
                              "err": p, "top1": round(float(np.mean(accs)), 3),
                              "chrom_acc": round(float(np.mean(chrom_accs)), 3),
                              "j_med": round(float(np.median(js)), 3) if js else None,
                              "cover_med": round(float(np.median(covs)), 3) if covs else None})
                print(f"  [{stratum}] {L//1000:3d} kb @ {p:.0%}: "
                      f"top-1 {cells[-1]['top1']:.0%}  chrom {cells[-1]['chrom_acc']:.0%}  "
                      f"J {cells[-1]['j_med']}  cover {cells[-1]['cover_med']}")
    print(f"  reads bench in {time.time()-t0:.0f}s")
    res = {"n_per": n_per, "cells": cells}
    with open(OUT / "bench_reads.json", "w") as fh:
        json.dump(res, fh)
    return res


def run(which="all", params: Params = PARAMS):
    kit = Kit(params)
    G = Genome(kit, params)
    if which in ("mis", "all"):
        print("== misassembly detection ==")
        run_misassembly(kit, G, params=params)
    if which in ("reads", "all"):
        print("== ultralong-read recruitment ==")
        run_reads(kit, G, params=params)


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "all")
