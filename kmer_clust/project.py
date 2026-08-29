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
    ("chm13_slice", "CHM13 chr21 slice (control)", "chm13v2.0.fa.gz",
     "the reference projected onto itself — every window must come home (and does: 40/40 to chr21)"),
    ("hg00097_h1", "HG00097 hap1 · chr21 acro (HPRC)", "HG00097_hap1_hprc_r2_v1.0.1.fa.gz",
     "a hard acrocentric slice — the pan-acrocentric commons pulls some windows toward chr13/14"),
    ("hg00097_h2", "HG00097 hap2 · chr21 acro (HPRC)", "HG00097_hap2_hprc_r2_v1.0.1.fa.gz",
     "same locus, other haplotype — more of its vocabulary finds its best exact home on chr13"),
    ("na19909_h2", "NA19909 hap2 · chr21 acro (HPRC)", "NA19909_hap2_hprc_r2_v1.0.1.fa.gz",
     "the extreme case: its best homes hop between chr13/14/21 — assembly order fully scrambled (τ −0.19)"),
]

# famous T2T regions the showcase fishes out of a whole projected haplotype
SHOWCASES = [
    ("mhc", "chr6", 28.5, 33.5, "MHC / HLA",
     "the genome's most polymorphic region — divergent in sequence, colinear in structure; all windows land on chr6 (J≈0.6)"),
    ("mapt", "chr17", 45.0, 47.5, "MAPT / 17q21.31",
     "the 17q21.31 H1/H2 inversion polymorphism — the most word-divergent locus showcased (J≈0.35), yet placed exactly"),
    ("igh", "chr14", 99.0, 101.2, "IGH",
     "the immunoglobulin heavy-chain locus — germline-variable between people (window J spans 0.25–0.95)"),
    ("def8p", "chr8", 6.5, 13.0, "8p23.1 defensins",
     "an inversion flanked by defensin-cluster segdups — the one showcase locus whose window order shuffles (τ +0.67)"),
    ("smn", "chr5", 70.5, 71.8, "SMN1/SMN2 · 5q13",
     "the SMA locus: 7 of 8 windows carry TWO strong chr5 homes ~0.9 Mb apart — order shuffles between the twins (τ +0.46)"),
    ("lcr22", "chr22", 19.2, 22.5, "22q11.2 · DiGeorge/VCFS",
     "the DiGeorge microdeletion region — LCR22 segdups multi-map while the reverse-stored contig inverts the axis (τ −0.91)"),
    ("lrc_kir", "chr19", 57.15, 58.1, "KIR / LRC · 19q13.4",
     "NK-cell immunity genes: the KIR window is 20% novel to CHM13 (cover 0.80) — KIR haplotypes differ in gene content"),
    ("yq12", "chrY", 30.0, 60.0, "Yq12 heterochromatin",
     "the giant DYZ satellite ocean; CHM13's chrY IS HG002's — near-self control (J≈0.9, runner-ups elsewhere in Yq12)"),
]
# hand-cut contiguous contig walks (data-driven discoveries; windows of one
# contig, chosen by a stated rule) — built by run_showcase when the scanned
# haplotype contains the contig
WALKS = [
    ("cen13_hg002", "HG002#1#JAHKSE010000070.1", 944, 973,
     "chr13 centromere entry",
     "into the chr13 centromere: coverage crashes to 0.55 as personal HOR variants appear — the contig dies in the array"),
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

    def project_hashes(self, uniq: np.ndarray, cts: np.ndarray):
        """Sketch hashes (unique + counts) -> model hits, exact hits, coverage.

        Uses index gathers (rows of V, columns of Bfull) so a whole haplotype
        of windows projects in minutes, not hours."""
        # exact locus evidence: Jaccard vs every bin's full sketch
        posf = np.searchsorted(self.universe_full, uniq)
        okf = (posf < self.universe_full.size) & (
            self.universe_full[np.minimum(posf, self.universe_full.size - 1)] == uniq
        )
        cover = float(okf.sum() / max(uniq.size, 1))
        cols = posf[okf].astype(np.int64)
        inter = np.asarray(self.Bfull[:, cols].sum(axis=1)).ravel()
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
            colw = pos[ok]
            wv = np.log1p(cts[ok]).astype(np.float32) * self.idf[colw]
            wv /= max(np.linalg.norm(wv), 1e-9)
            zq = wv @ self.V[colw]
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

    def project_window(self, codes: np.ndarray):
        """codes (2-bit) -> model hits (cosine), exact hits (Jaccard), coverage."""
        p = self.params
        _, hs = sketch_codes(codes, p.k, p.embed_scaled)
        if hs.size < 10:
            return None
        uniq, cts = np.unique(hs, return_counts=True)
        return self.project_hashes(uniq, cts)


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
    for sid, label, fname, blurb in QUERY_SETS:
        path = KMER_DUST_TESTDATA / fname
        if not path.exists():
            print(f"  ({fname} not found; skipping)")
            continue
        windows = project_set(kit, path)
        sets.append({"id": sid, "label": label, "blurb": blurb, "windows": windows})
        med = np.median([w["hits"][0][1] for w in windows if "hits" in w])
        print(f"  {label}: {len(windows)} windows, median top-1 sim {med:.3f}")
    with open(OUT / "projection.json", "w") as fh:
        json.dump({"selftest": st, "sets": sets,
                   "params": {"k": params.k, "scaled": params.embed_scaled,
                              "window_bp": params.bin_bp, "top_hits": TOP_HITS}}, fh)
    print(f"projection -> out/projection.json, t={time.time()-t0:.0f}s")


def scan_haplotype(kit: Kit, fasta_path) -> list:
    """Project every 100 kb window of a whole assembly; cached as parquet so
    re-tuning the showcase selection does not re-pay the ~5 min scan."""
    from pathlib import Path

    tag = Path(fasta_path).name.removesuffix(".gz").removesuffix(".fa")
    cache = OUT / f"hapscan_{tag}.parquet"
    if cache.exists() and cache.stat().st_mtime > Path(fasta_path).stat().st_mtime:
        df = pd.read_parquet(cache)
        print(f"  hapscan cache hit: {len(df)} windows")
        return [(row.contig, int(row.w),
                 {"cover": float(row.cover),
                  "hits": [int(x) for x in row.hits],
                  "sims": [float(x) for x in row.sims],
                  "ehits": [int(x) for x in row.ehits],
                  "ejacc": [float(x) for x in row.ejacc]})
                for row in df.itertuples()]

    p = kit.params
    win = p.bin_bp
    results = []
    t0 = time.time()
    for name, codes in iter_fasta_codes(fasta_path):
        if codes.size < win:
            continue
        pos, hs = sketch_codes(codes, p.k, p.embed_scaled)
        wids = (pos // win).astype(np.int64)
        order = np.lexsort((hs, wids))
        wids, hs2 = wids[order], hs[order]
        bounds = np.flatnonzero(np.diff(wids)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [wids.size]))
        for si in range(starts.size):
            w = int(wids[starts[si]])
            if (w + 1) * win > codes.size:
                continue
            hw = hs2[starts[si]:ends[si]]
            if hw.size < 10:
                continue
            uniq, cts = np.unique(hw, return_counts=True)
            r = kit.project_hashes(uniq, cts)
            if r and r["ehits"]:
                results.append((name, w, r))
        del codes
    print(f"  projected {len(results)} windows in {time.time()-t0:.0f}s")
    pd.DataFrame({
        "contig": [n for n, _, _ in results],
        "w": [w for _, w, _ in results],
        "cover": [r["cover"] for _, _, r in results],
        "hits": [r["hits"] for _, _, r in results],
        "sims": [r["sims"] for _, _, r in results],
        "ehits": [r["ehits"] for _, _, r in results],
        "ejacc": [r["ejacc"] for _, _, r in results],
    }).to_parquet(cache)
    return results


def window_entry(kit: Kit, contig: str, w: int, r: dict, pos_mb: float) -> dict:
    win = kit.params.bin_bp
    return {
        "label": f"{contig}:{w*win/1e6:.1f}–{(w+1)*win/1e6:.1f} Mb",
        "pos_mb": round(pos_mb, 2),
        "cover": round(r["cover"], 3),
        "hits": [[int(h), s2] for h, s2 in zip(r["hits"], r["sims"])] or
                [[int(h), s2] for h, s2 in zip(r["ehits"], r["ejacc"])],
        "loci": loci_of(kit, r["ehits"], r["ejacc"]),
    }


def whole_chrom_set(kit: Kit, results: list, sample: str, chrom: str = "chrY",
                    sid: str = "", blurb: str = "", min_hits: int = 15,
                    cap: int = 120, exclude: set | None = None) -> dict | None:
    """One-contig chromosome example, same shape as the region showcases:
    among contigs with >= min_hits windows whose best exact locus is on
    `chrom` (skipping contigs already featured in another set), pick the one
    maximizing hits x reference-span walked, and show its contiguous span."""
    cm = (kit.bins["chrom"] == chrom).to_numpy()
    lo_b, hi_b = int(np.flatnonzero(cm).min()), int(np.flatnonzero(cm).max())
    win = kit.params.bin_bp
    by_contig = {}
    for n, w, r in results:
        by_contig.setdefault(n, {})[w] = r
    best = None  # (hits x ref-span score, contig, hit windows, span)
    for n, ws in by_contig.items():
        if exclude and n in exclude:
            continue
        hits = [w for w, r in ws.items() if lo_b <= r["ehits"][0] <= hi_b]
        if len(hits) < min_hits:
            continue
        mbs = [(ws[w]["ehits"][0] - lo_b) * win / 1e6 for w in hits]
        span0 = max(mbs) - min(mbs)
        key = span0 * len(hits)
        if best is None or key > best[0]:
            best = (key, n, hits, span0)
    if best is None:
        print(f"  (whole-{chrom}: no contig with >= {min_hits} hits)")
        return None
    _, dom, hits, span = best
    lo, hi = min(hits), max(hits)
    members = sorted(w for w in by_contig[dom] if lo <= w <= hi)
    n_all = len(members)
    if n_all > cap:
        idx = np.unique(np.linspace(0, n_all - 1, cap).round().astype(int))
        members = [members[i] for i in idx]
    windows = [window_entry(kit, dom, w, by_contig[dom][w], w * win / 1e6)
               for w in members]
    label = f"{sample} · chrY contig ({span:.0f} Mb walk)"
    print(f"  whole-{chrom}: {dom} — {len(windows)} windows"
          f"{f' of {n_all}' if n_all > cap else ''}, walks {span:.0f} Mb")
    out = {"id": sid or f"y_{sample.split()[0].lower()}", "label": label,
           "blurb": blurb, "windows": windows}
    if n_all > cap:
        out["n_win_all"] = n_all
        out["trim_kind"] = "uniform"
    return out


def contig_walk_set(kit: Kit, by_contig: dict, sample: str, sid: str,
                    contig: str, w0: int, w1: int, label: str,
                    blurb: str) -> dict | None:
    """A hand-cut contiguous walk of one contig (windows w0..w1)."""
    ws = by_contig.get(contig)
    if not ws:
        return None
    members = sorted(w for w in ws if w0 <= w <= w1)
    if len(members) < 8:
        return None
    win = kit.params.bin_bp
    windows = [window_entry(kit, contig, w, ws[w], w * win / 1e6)
               for w in members]
    print(f"  walk {sid}: {len(windows)} windows of {contig}")
    return {"id": sid, "label": f"{sample} · {label}", "blurb": blurb,
            "windows": windows}


def showcase(kit: Kit, fasta_path, sample: str, cap: int = 44) -> list:
    """Project a WHOLE haplotype and pull out the windows whose best exact
    locus lands in famous T2T regions — the projector locating landmarks in
    an unannotated assembly by vocabulary alone. Each set is one CONTIGUOUS
    assembly segment: the span between the region's first and last hit on the
    dominant contig, in-span non-hit windows included, so the assembly axis
    has no artificial holes. Sets over `cap` windows are trimmed to the best
    contiguous run (never interior-dropped); the trim is recorded in
    n_win_all/trim_kind and disclosed on the axis."""
    results = scan_haplotype(kit, fasta_path)
    by_contig = {}
    for n, w, r in results:
        by_contig.setdefault(n, {})[w] = r

    win = kit.params.bin_bp
    sets = []
    for sid, chrom, mb0, mb1, label, blurb in SHOWCASES:
        m = (kit.bins["chrom"] == chrom) & (kit.bins["start"] >= mb0 * 1e6) \
            & (kit.bins["start"] < mb1 * 1e6)
        idset = set(np.flatnonzero(m.to_numpy()).tolist())
        hit_ws = {}
        for n, w, r in results:
            if r["ehits"][0] in idset:
                hit_ws.setdefault(n, []).append(w)
        if not hit_ws or max(len(v) for v in hit_ws.values()) < 8:
            got = max((len(v) for v in hit_ws.values()), default=0)
            print(f"  ({label}: only {got} windows hit "
                  f"{chrom}:{mb0}-{mb1}; skipped)")
            continue
        dom = max(hit_ws, key=lambda n: len(hit_ws[n]))
        hs = set(hit_ws[dom])
        lo, hi = min(hs), max(hs)
        members = sorted(w for w in by_contig[dom] if lo <= w <= hi)
        n_span = len(members)
        if n_span > cap:
            # trim to the contiguous run keeping the most in-region hits
            # (never punch interior holes by dropping low-J windows)
            best = (-1, -1.0, 0)
            for i0 in range(len(members) - cap + 1):
                run = members[i0:i0 + cap]
                nh = sum(1 for w in run if w in hs)
                tj = sum(by_contig[dom][w]["ejacc"][0] for w in run)
                if (nh, tj) > (best[0], best[1]):
                    best = (nh, tj, i0)
            members = members[best[2]:best[2] + cap]
        windows = []
        for w in members:
            windows.append(window_entry(kit, dom, w, by_contig[dom][w],
                                        w * win / 1e6))
        entry = {"id": sid, "label": f"{sample} · {label}",
                 "blurb": blurb, "windows": windows}
        if n_span > cap:
            entry["n_win_all"] = n_span
            entry["trim_kind"] = "contiguous-run"
        sets.append(entry)
        med_cov = float(np.median([w2["cover"] for w2 in windows]))
        print(f"  {label}: {len(windows)} windows from {dom}, "
              f"median cover {med_cov:.2f}")
    return sets


def _write_projection(pj: dict) -> None:
    """Serialize fully before touching the file — a bad value must never
    leave projection.json truncated."""
    text = json.dumps(pj)
    tmp = OUT / "projection.json.tmp"
    tmp.write_text(text)
    tmp.replace(OUT / "projection.json")


def run_showcase(fasta_path, sample: str, params: Params = PARAMS) -> None:
    kit = Kit(params)
    new_sets = showcase(kit, fasta_path, sample)
    results = scan_haplotype(kit, fasta_path)
    by_contig = {}
    for n, w, r in results:
        by_contig.setdefault(n, {})[w] = r
    for sid, contig, w0, w1, label, blurb in WALKS:
        st = contig_walk_set(kit, by_contig, sample, sid, contig, w0, w1,
                             label, blurb)
        if st:
            new_sets.append(st)
    pj = json.loads((OUT / "projection.json").read_text())
    have = {s2["id"] for s2 in new_sets}
    pj["sets"] = [s2 for s2 in pj["sets"] if s2["id"] not in have] + new_sets
    _write_projection(pj)
    print(f"projection.json now holds {len(pj['sets'])} sets")


# ---------------------------------------------------------------- triage
TRIAGE_MIN_WIN = 3       # report contigs with at least this many windows
TRIAGE_ORIENT_WIN = 5    # orientation verdicts need this many placed windows
TRIAGE_NOVEL_COVER = 0.90
TRIAGE_JUMP_MB = 5.0


def _tau_a(order_vals, tie=0.01):
    n = len(order_vals)
    if n < 2:
        return 0.0
    S = sum(np.sign(order_vals[j] - order_vals[i])
            if abs(order_vals[j] - order_vals[i]) > tie else 0
            for i in range(n) for j in range(i + 1, n))
    return float(S / (n * (n - 1) / 2))


def triage_rows(chrom_idx, bin_mb, sat_mask, results, bin_bp,
                min_windows: int = TRIAGE_MIN_WIN):
    """Per-contig compass rows from a whole-assembly scan.

    chrom_idx[b], bin_mb[b]: reference chromosome index and within-chromosome
    Mb of bin b; sat_mask[b]: censat != '' (judge only). results: the
    scan_haplotype list. Pure function — unit-testable on synthetic data."""
    by_contig = {}
    for n, w, r in results:
        by_contig.setdefault(n, []).append((w, r))
    rows = []
    for contig, ws in sorted(by_contig.items()):
        ws.sort()
        if len(ws) < min_windows:
            continue
        j1 = np.array([r["ejacc"][0] for _, r in ws])
        top = np.array([r["ehits"][0] for _, r in ws])
        cov = np.array([r["cover"] for _, r in ws])
        ci = chrom_idx[top]
        placed = j1 >= 0.1
        # dominant chromosome among placed windows
        if placed.sum():
            vals, cnts = np.unique(ci[placed], return_counts=True)
            dom = int(vals[cnts.argmax()])
            dom_n = int(cnts.max())
        else:
            dom, dom_n = -1, 0
        on_dom = placed & (ci == dom)
        mbs = bin_mb[top]
        span_lo = float(mbs[on_dom].min()) if dom_n else float("nan")
        span_hi = float(mbs[on_dom].max()) if dom_n else float("nan")
        # orientation: assembly order vs reference position on the dominant chrom
        if placed.sum() == 0:
            tau, orient = float("nan"), "unplaced"
        elif dom_n >= TRIAGE_ORIENT_WIN:
            tau = _tau_a(list(mbs[on_dom]))
            orient = ("forward" if tau >= 0.85 else
                      "reverse" if tau <= -0.85 else
                      "mostly+" if tau >= 0.5 else
                      "mostly-" if tau <= -0.5 else "scrambled")
        else:
            tau, orient = float("nan"), "short"
        # order breaks: adjacent placed windows jumping chromosome or > 5 Mb
        jumps = 0
        prev = None
        for k in range(len(ws)):
            if not placed[k]:
                continue
            cur = (int(ci[k]), float(mbs[k]))
            if prev is not None and (cur[0] != prev[0]
                                     or abs(cur[1] - prev[1]) > TRIAGE_JUMP_MB):
                jumps += 1
            prev = cur
        # novelty runs: contiguous windows (step 1) below the coverage bar
        runs, cur0 = [], None
        wids = [w for w, _ in ws]
        for k in range(len(ws)):
            low = cov[k] < TRIAGE_NOVEL_COVER
            step = k > 0 and wids[k] == wids[k - 1] + 1
            if low and cur0 is not None and not step:
                runs.append((cur0, k - 1))  # window-id gap splits the run
                cur0 = k
            elif low and cur0 is None:
                cur0 = k
            if not low and cur0 is not None:
                runs.append((cur0, k - 1))
                cur0 = None
        if cur0 is not None:
            runs.append((cur0, len(ws) - 1))
        runs = [(a, b) for a, b in runs if b - a + 1 >= 3]
        best_run = max(runs, key=lambda ab: ab[1] - ab[0], default=None)
        nov = {"novel_frac": round(float((cov < TRIAGE_NOVEL_COVER).mean()), 3)}
        if best_run is not None:
            a, b = best_run
            nov.update(novel_run_mb0=round(wids[a] * bin_bp / 1e6, 1),
                       novel_run_mb1=round((wids[b] + 1) * bin_bp / 1e6, 1),
                       novel_run_mincov=round(float(cov[a:b + 1].min()), 2))
        def terrain(k):
            if not placed[k]:
                return "unplaced"
            return "satellite" if sat_mask[top[k]] else "non-sat"
        rows.append({
            "contig": contig, "n_win": len(ws),
            "mb": round(len(ws) * bin_bp / 1e6, 1),
            "dom_chrom": int(dom), "dom_frac": round(dom_n / max(placed.sum(), 1), 2),
            "span_lo": round(span_lo, 1) if dom_n else None,
            "span_hi": round(span_hi, 1) if dom_n else None,
            "tau": round(tau, 2) if tau == tau else None,
            "orient": orient, "jumps": int(jumps),
            "j_med": round(float(np.median(j1)), 2),
            "cover_med": round(float(np.median(cov)), 2),
            "end5": terrain(0), "end3": terrain(len(ws) - 1), **nov,
        })
    return rows


def triage_summary(rows, results, n_ref_bins, bin_bp):
    j1 = np.array([r["ejacc"][0] for _, _, r in results])
    cov = np.array([r["cover"] for _, _, r in results])
    top = np.array([r["ehits"][0] for _, _, r in results])
    big = [r for r in rows if r["n_win"] >= TRIAGE_ORIENT_WIN]
    ends = [r[e] for r in big for e in ("end5", "end3")]
    orient = {}
    for r in big:
        orient[r["orient"]] = orient.get(r["orient"], 0) + 1
    return {
        "n_windows": len(results),
        "assembly_mb": round(len(results) * bin_bp / 1e6, 1),
        "n_contigs_reported": len(rows),
        "placed_confident": round(float((j1 >= 0.3).mean()), 3),
        "placed_weak": round(float(((j1 >= 0.1) & (j1 < 0.3)).mean()), 3),
        "unplaced": round(float((j1 < 0.1).mean()), 3),
        "ref_breadth": round(len(np.unique(top)) / n_ref_bins, 3),
        "novel_mb": round(float((cov < TRIAGE_NOVEL_COVER).sum()) * bin_bp / 1e6, 1),
        "orientation_census": orient,
        "ends_in_satellite": round(ends.count("satellite") / max(len(ends), 1), 3),
        "ends_unplaced": round(ends.count("unplaced") / max(len(ends), 1), 3),
    }


def run_triage(fasta_path, sample: str, params: Params = PARAMS) -> None:
    from pathlib import Path

    kit = Kit(params)
    results = scan_haplotype(kit, fasta_path)
    chroms = list(dict.fromkeys(kit.bins["chrom"]))
    chrom_idx = np.array([chroms.index(c) for c in kit.bins["chrom"]])
    bin_mb = (kit.bins["start"].to_numpy() / 1e6) + params.bin_bp / 2e6
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    sat_mask = (annot["censat_class"] != "").to_numpy()
    rows = triage_rows(chrom_idx, bin_mb, sat_mask, results, params.bin_bp)
    if not rows:
        print("triage: no contig reached the minimum window count; nothing to report")
        return
    summ = triage_summary(rows, results, len(kit.bins), params.bin_bp)

    tag = Path(fasta_path).name.removesuffix(".gz").removesuffix(".fa")
    df = pd.DataFrame(rows)
    df["dom_chrom"] = [chroms[c] if c >= 0 else "" for c in df["dom_chrom"]]
    tsv = OUT / f"triage_{tag}.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    print(f"\n== triage: {sample} ==")
    print(f"{summ['n_windows']} windows ({summ['assembly_mb']} Mb) in "
          f"{summ['n_contigs_reported']} contigs (>= {TRIAGE_MIN_WIN} windows)")
    print(f"placement: {summ['placed_confident']:.0%} confident (J>=0.3), "
          f"{summ['placed_weak']:.0%} weak, {summ['unplaced']:.0%} unplaced; "
          f"reference breadth {summ['ref_breadth']:.0%}")
    print(f"orientation census (>= {TRIAGE_ORIENT_WIN} windows): "
          + ", ".join(f"{k} {v}" for k, v in sorted(summ["orientation_census"].items())))
    print(f"novel vs reference: {summ['novel_mb']} Mb below cover "
          f"{TRIAGE_NOVEL_COVER}; contig ends in satellite: "
          f"{summ['ends_in_satellite']:.0%} (unplaced {summ['ends_unplaced']:.0%})")
    print(f"table: {tsv}")
    with open(OUT / f"triage_{tag}.json", "w") as fh:
        json.dump({"sample": sample, "summary": summ, "rows": rows}, fh)
    try:
        from .figures import fig_triage
        png = fig_triage(rows, summ, chroms, kit.bins, sat_mask, params,
                         sample, tag)
        print(f"figure: {png}")
    except Exception as e:  # figure is a bonus, never blocks the table
        print(f"(figure skipped: {e})")


def run_ychrom(fasta_path, sample: str, blurb: str = "",
               params: Params = PARAMS) -> None:
    kit = Kit(params)
    results = scan_haplotype(kit, fasta_path)
    pj = json.loads((OUT / "projection.json").read_text())
    sid = f"y_{sample.split()[0].lower()}"
    used = {w["label"].rsplit(":", 1)[0]
            for s2 in pj["sets"] if s2["id"] != sid
            for w in s2["windows"]}
    st = whole_chrom_set(kit, results, sample, "chrY", blurb=blurb,
                         exclude=used)
    if st is None:
        return
    pj["sets"] = [s2 for s2 in pj["sets"] if s2["id"] != st["id"]] + [st]
    _write_projection(pj)
    print(f"projection.json now holds {len(pj['sets'])} sets")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "showcase":
        run_showcase(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "HPRC")
    elif len(sys.argv) >= 3 and sys.argv[1] == "triage":
        run_triage(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "assembly")
    elif len(sys.argv) >= 3 and sys.argv[1] == "ychrom":
        run_ychrom(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "HPRC",
                   sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        run()
