"""Real ultralong ONT reads vs the projector, with minimap2 as comparator.

Reads: a streamed subset of the GIAB HG002 ultralong PromethION run
(2019 basecalls, ~7-10% error — a deliberately harsh test), >=50 kb.
Each read is placed twice: by this repo's two-signal projector
(vocabulary -> exact-J locus + coverage), and by minimap2 (map-ont,
via mappy) against the same CHM13v2.0.

What is measured (out/bench_real.json + per-read parquet):
1. Concordance where the field standard is confident: reads whose
   primary minimap2 alignment has mapq >= 50 in non-satellite terrain —
   our chromosome- and bin-level agreement.
2. Behavior where alignment struggles: per terrain (satellite vs not),
   the fraction of reads each method places confidently.
3. The error meter: per-read error estimated from vocabulary coverage
   alone, err = 1 - cover^(1/k), against minimap2's alignment identity.

Run: python -m kmer_clust.bench_real [fastq.gz]

Rebuilding the read subset (first 1,200 reads >= 50 kb of GIAB flowcell 1):
  curl -s https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/\
AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/\
GM24385_1.fastq.gz | gunzip -c | python -c "
import sys, gzip
out = gzip.open('data/HG002_UL_subset.fastq.gz', 'wt'); kept = 0
it = sys.stdin
while kept < 1200:
    h = it.readline(); s = it.readline(); p = it.readline(); q = it.readline()
    if not q: break
    if len(s) - 1 >= 50_000:
        out.write(h + s + p + q); kept += 1
out.close()"
"""

import json
import sys
import time

import numpy as np
import pandas as pd

from .config import DATA, OUT, Params, PARAMS
from .fracminhash import encode_sequence
from .project import Kit

MIN_LEN = 50_000
CONF_MAPQ = 50


def get_aligner():
    import mappy

    mmi = DATA / "chm13_ont.mmi"
    if mmi.exists():
        return mappy.Aligner(str(mmi))
    return mappy.Aligner(str(DATA / "chm13v2.0.fa.gz"), preset="map-ont",
                         fn_idx_out=str(mmi))


def run(fastq=None, params: Params = PARAMS, max_reads=1500):
    import mappy

    fastq = fastq or str(DATA / "HG002_UL_subset.fastq.gz")
    kit = Kit(params)
    chroms = list(dict.fromkeys(kit.bins["chrom"]))
    chrom_of_bin = np.array([chroms.index(c) for c in kit.bins["chrom"]])
    bin0 = {}
    i = 0
    for c in chroms:
        bin0[c] = i
        i += int((kit.bins["chrom"] == c).sum())
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    sat = (annot["censat_class"] != "").to_numpy()

    print("building/loading minimap2 index (map-ont)…")
    t0 = time.time()
    al = get_aligner()
    print(f"  aligner ready in {time.time()-t0:.0f}s")

    rows = []
    t0 = time.time()
    n_proj = 0.0
    for name, seq, _ in mappy.fastx_read(fastq):
        if len(seq) < MIN_LEN:
            continue
        if len(rows) >= max_reads:
            break
        # ---- projector
        tp = time.time()
        codes = encode_sequence(seq.encode())
        r = kit.project_window(codes)
        n_proj += time.time() - tp
        if not r or not r["ehits"]:
            continue
        top, j1 = r["ehits"][0], r["ejacc"][0]
        # margin: best hit at least 1 Mb / one chromosome away from top.
        # If every stored hit is local, the true distant best is unknown —
        # bound it by the weakest stored hit instead of claiming zero.
        j_alt = float(r["ejacc"][-1])
        for b, j in zip(r["ehits"][1:], r["ejacc"][1:]):
            if chrom_of_bin[b] != chrom_of_bin[top] or \
                    abs(b - top) * params.bin_bp > 1_000_000:
                j_alt = j
                break
        # ---- minimap2
        best = None
        for h in al.map(seq):
            if h.is_primary and (best is None or h.mlen > best.mlen):
                best = h
        if best is not None and best.ctg not in bin0:
            best = None  # e.g. chrM: outside the analysis bin set
        row = {"read": name, "len": len(seq),
               "our_bin": int(top), "our_chrom": chroms[chrom_of_bin[top]],
               "our_j": float(j1), "our_alt_j": float(j_alt),
               "our_cover": float(r["cover"]),
               "our_err": float(1 - r["cover"] ** (1 / params.k))}
        if best is not None:
            mid = (best.r_st + best.r_en) // 2
            mm_bin = bin0[best.ctg] + mid // params.bin_bp
            row.update(mm_chrom=best.ctg, mm_bin=int(mm_bin),
                       mm_mapq=int(best.mapq),
                       mm_ident=float(best.mlen / max(best.blen, 1)),
                       mm_sat=bool(sat[mm_bin]))
        rows.append(row)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} reads…")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "bench_real_reads.parquet")
    print(f"{len(df)} reads in {time.time()-t0:.0f}s "
          f"(projector {1000*n_proj/len(df):.0f} ms/read)")

    # ---- summaries
    m = df.dropna(subset=["mm_bin"]).copy()
    m["mm_bin"] = m["mm_bin"].astype(int)
    conf = m[(m.mm_mapq >= CONF_MAPQ) & (~m.mm_sat.astype(bool))]
    chrom_acc = float((conf.our_chrom == conf.mm_chrom).mean())
    bin_acc = float((abs(conf.our_bin - conf.mm_bin) <= 1).mean())

    sat_reads = m[m.mm_sat.astype(bool)]
    aligned_any = df["mm_bin"].notna()
    conf_ok = conf[conf.our_chrom == conf.mm_chrom]
    conf_bad = conf[conf.our_chrom != conf.mm_chrom]
    margin = conf.our_j / np.maximum(conf.our_alt_j, 1e-4)
    keep = margin >= 2
    kept = conf[keep]
    res_margin = {
        "discordant_margins": sorted(
            round(float(m), 3) for m in
            (conf_bad.our_j / np.maximum(conf_bad.our_alt_j, 1e-4))),
        "correct_margin_median": round(float(
            (conf_ok.our_j / np.maximum(conf_ok.our_alt_j, 1e-4)).median()), 2),
        "abstain_frac_at_2x": round(float((~keep).mean()), 4),
        "kept_chrom_acc": round(float(
            (kept.our_chrom == kept.mm_chrom).mean()), 4),
        "note": "2x threshold chosen post hoc on this set (worst bin-level "
                "discordant margin 1.98); validate on independent flowcells",
    }
    try:
        _fig_real(m, conf_ok, conf_bad)
    except Exception as e:
        print(f"(figure skipped: {e})")

    res = {
        "fastq": fastq, "n_reads": len(df),
        "margin": res_margin,
        "read_len_median": int(df["len"].median()),
        "mm_identity_median": float(m.mm_ident.median()),
        "concordance": {
            "n": len(conf), "chrom": round(chrom_acc, 4),
            "bin_pm1": round(bin_acc, 4)},
        "satellite": {
            "n": len(sat_reads),
            "mm_conf_frac": round(float((sat_reads.mm_mapq >= 20).mean()), 3)
            if len(sat_reads) else None,
            "our_chrom_agree_with_mm_best":
                round(float((sat_reads.our_chrom == sat_reads.mm_chrom).mean()), 3)
                if len(sat_reads) else None},
        "unaligned_by_mm": int((~aligned_any).sum()),
        "error_meter": {
            "pearson_r": round(float(np.corrcoef(
                m.our_err, 1 - m.mm_ident)[0, 1]), 3),
            "spearman": round(float(
                pd.Series(m.our_err).corr(pd.Series(1 - m.mm_ident),
                                          method="spearman")), 3)},
    }
    with open(OUT / "bench_real.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return res


def _fig_real(m, ok, disc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=115)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax = axes[0]
    ax.scatter(1 - m.mm_ident, m.our_err, s=7, alpha=0.35, c="#3987e5",
               linewidths=0)
    A = np.polyfit(1 - m.mm_ident, m.our_err, 1)
    xs = np.linspace(0, float((1 - m.mm_ident).max()), 50)
    ax.plot(xs, A[0] * xs + A[1], color="#d94f26", lw=1.4,
            label=f"fit: {A[0]:.2f}·x + {A[1]:.3f}")
    ax.set_xlabel("minimap2 alignment error (1 − identity)")
    ax.set_ylabel("our estimate: 1 − cover$^{1/21}$")
    r = float(np.corrcoef(m.our_err, 1 - m.mm_ident)[0, 1])
    ax.set_title("read error from vocabulary coverage alone\n"
                 f"{len(m)} real ultralong reads · Pearson r = {r:.3f}",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    bins = np.geomspace(0.5, 400, 36)
    mo = np.maximum(ok.our_j / np.maximum(ok.our_alt_j, 1e-4), 0.5)
    md = np.maximum(disc.our_j / np.maximum(disc.our_alt_j, 1e-4), 0.5)
    ax.hist(mo, bins=bins, color="#3987e5", alpha=0.75,
            label=f"concordant (n={len(ok)})")
    ax.hist(md, bins=bins, color="#d94f26", alpha=0.9,
            label=f"discordant (n={len(disc)})")
    ax.axvline(2, color="#171916", lw=1, ls="--")
    ax.text(2.2, ax.get_ylim()[1] * 0.5,
            "margin ≥ 2×\nabstains on every\ndiscordant read", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("placement margin (top J / best distant J)")
    ax.set_ylabel("reads")
    ax.set_title("wrong placements identify themselves\n"
                 "(minimap2 mapq ≥ 50 reads as truth)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = OUT / "figs" / "real_reads.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"figure: {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
