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
        # margin: best hit at least 1 Mb / one chromosome away from top
        j_alt = 0.0
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
    res = {
        "fastq": fastq, "n_reads": len(df),
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


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
