"""Stage: per-bin tandem periodicity from kept-hash positions.

Annotation-free and alignment-free: within one bin, the same FracMinHash word
recurring every P bases is the signature of a tandem array with unit ~P. For
each bin we take the gaps between successive occurrences of each repeated
hash, find the dominant gap on a log-spaced grid, and report

  period_bp  - median gap inside the dominant band
  strength   - fraction of the bin's gaps that fall in that band (0 if the
               bin has fewer than MIN_GAPS gaps)

Live alpha-HOR arrays should read as multiples of the 171 bp monomer, rDNA
near its ~45 kb unit -- that expectation is a judge, not an input.
"""

import json

import numpy as np
import pandas as pd

from .config import OUT, Params
from .fasta import iter_chrom_codes
from .fracminhash import sketch_codes

MIN_GAPS = 15
GRID = np.geomspace(60, 100_000, 90)


def bin_periods(pos: np.ndarray, hashes: np.ndarray, n_bins: int, bin_bp: int):
    """Dominant within-bin gap between repeats of the same hash, per bin."""
    period = np.zeros(n_bins, np.float32)
    strength = np.zeros(n_bins, np.float32)
    if pos.size < 2:
        return period, strength
    order = np.lexsort((pos, hashes))
    p = pos[order].astype(np.int64)
    h = hashes[order]
    same = h[1:] == h[:-1]
    gaps = (p[1:] - p[:-1])[same]
    gbin = (p[:-1][same] // bin_bp).astype(np.int64)
    ok = (gaps >= GRID[0]) & (gaps <= GRID[-1])
    gaps, gbin = gaps[ok], gbin[ok]
    if gaps.size == 0:
        return period, strength
    o2 = np.argsort(gbin, kind="stable")
    gaps, gbin = gaps[o2], gbin[o2]
    starts = np.concatenate(([0], np.flatnonzero(gbin[1:] != gbin[:-1]) + 1, [gbin.size]))
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s < MIN_GAPS:
            continue
        b = gbin[s]
        g = gaps[s:e]
        hist, _ = np.histogram(g, bins=GRID)
        i = int(hist.argmax())
        band = g[(g >= GRID[i]) & (g < GRID[i + 1])]
        if band.size < MIN_GAPS // 3:
            continue
        period[b] = float(np.median(band))
        strength[b] = float(band.size / g.size)
    return period, strength


def run(params: Params) -> pd.DataFrame:
    all_period, all_strength = [], []
    for chrom, codes in iter_chrom_codes(params):
        pos, hashes = sketch_codes(codes, params.k, params.base_scaled)
        n_bins = (codes.size + params.bin_bp - 1) // params.bin_bp
        period, strength = bin_periods(pos, hashes, n_bins, params.bin_bp)
        all_period.append(period)
        all_strength.append(strength)
        n_per = int((strength > 0.3).sum())
        print(f"  {chrom}: {n_per} bins with periodicity strength > 0.3")
    df = pd.DataFrame({
        "period_bp": np.concatenate(all_period),
        "strength": np.concatenate(all_strength),
    })
    out = OUT / f"periods_{params.bin_bp}.parquet"
    df.to_parquet(out, index=False)

    # judge-side sanity, appended to metrics.json
    annot_path = OUT / f"annot_{params.bin_bp}.parquet"
    if annot_path.exists():
        annot = pd.read_parquet(annot_path)
        live = (annot["cov_asat_hor_live"].to_numpy() >= 0.5) & (
            df["strength"].to_numpy() > 0.3
        )
        rdna = (annot["cov_rDNA"].to_numpy() >= 0.5) & (df["strength"].to_numpy() > 0.3)
        summary = {}
        if live.sum() >= 10:
            med = float(np.median(df["period_bp"][live]))
            summary["hor_live_median_period_bp"] = round(med, 1)
            summary["hor_live_period_monomers"] = round(med / 170.8, 2)
            summary["hor_live_frac_periodic"] = round(
                float(live.sum() / max((annot["cov_asat_hor_live"] >= 0.5).sum(), 1)), 3
            )
        if rdna.sum() >= 5:
            summary["rdna_median_period_bp"] = round(
                float(np.median(df["period_bp"][rdna])), 1
            )
        mpath = OUT / "metrics.json"
        if mpath.exists() and summary:
            m = json.loads(mpath.read_text())
            m["periodicity"] = summary
            mpath.write_text(json.dumps(m, indent=2))
            print("periodicity summary:", json.dumps(summary))
    print(f"periods -> {out}")
    return df
