"""Stage: sketch the genome into per-bin FracMinHash sketches.

Output is a "sketch store": for every fixed-size bin, the sorted unique kept
hashes and their within-bin multiplicities, concatenated CSR-style. Coarser
bins (e.g. 1 Mb from 100 kb) and coarser scaleds are derived from this store
without touching sequence again.
"""

import time

import numpy as np
import pandas as pd

from .config import Params
from .fasta import iter_chrom_codes
from .fracminhash import bin_stats, sketch_codes


def aggregate_bin_hashes(bin_ids: np.ndarray, hashes: np.ndarray, n_bins: int):
    """Unique (bin, hash) pairs with multiplicities.

    Returns (indptr int64 per bin, hashes uint64, counts uint32); hashes are
    sorted within each bin.
    """
    order = np.lexsort((hashes, bin_ids))
    b = bin_ids[order]
    h = hashes[order]
    if b.size == 0:
        return np.zeros(n_bins + 1, np.int64), h, np.zeros(0, np.uint32)
    new = np.empty(b.size, bool)
    new[0] = True
    new[1:] = (b[1:] != b[:-1]) | (h[1:] != h[:-1])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, b.size)).astype(np.uint32)
    b_u = b[starts]
    h_u = h[starts]
    indptr = np.zeros(n_bins + 1, np.int64)
    np.cumsum(np.bincount(b_u, minlength=n_bins), out=indptr[1:])
    return indptr, h_u, counts


def run(params: Params) -> None:
    k, scaled, bin_bp = params.k, params.base_scaled, params.bin_bp
    rows = []
    all_indptr = [np.zeros(1, np.int64)]
    all_hashes = []
    all_counts = []
    t0 = time.time()
    for chrom, codes in iter_chrom_codes(params):
        t1 = time.time()
        pos, hashes = sketch_codes(codes, k, scaled)
        n_bins = (codes.size + bin_bp - 1) // bin_bp
        bin_ids = (pos // bin_bp).astype(np.int64)
        indptr, h_u, c_u = aggregate_bin_hashes(bin_ids, hashes, n_bins)
        acgt, gc = bin_stats(codes, bin_bp)
        base = all_indptr[-1][-1]
        all_indptr.append(indptr[1:] + base)
        all_hashes.append(h_u)
        all_counts.append(c_u)
        sketch_sizes = np.diff(indptr)
        for b in range(n_bins):
            start = b * bin_bp
            rows.append(
                (
                    chrom,
                    start,
                    min(start + bin_bp, codes.size),
                    int(acgt[b]),
                    float(gc[b] / max(acgt[b], 1)),
                    int(sketch_sizes[b]),
                )
            )
        print(
            f"  {chrom}: {codes.size/1e6:.1f} Mb, {hashes.size/1e6:.2f} M kept, "
            f"{n_bins} bins, {time.time()-t1:.1f}s"
        )
        del codes
    indptr = np.concatenate(all_indptr)
    hashes = np.concatenate(all_hashes) if all_hashes else np.zeros(0, np.uint64)
    counts = np.concatenate(all_counts) if all_counts else np.zeros(0, np.uint32)
    bins = pd.DataFrame(
        rows, columns=["chrom", "start", "end", "acgt", "gc", "sketch_size"]
    )
    bins["distinct_est"] = bins["sketch_size"] * scaled
    bins.to_parquet(params.bins_parquet, index=False)
    np.savez_compressed(
        params.sketch_npz,
        indptr=indptr,
        hashes=hashes,
        counts=counts,
        meta=np.array([k, scaled, bin_bp], np.int64),
    )
    print(
        f"sketched {len(bins)} bins, {hashes.size/1e6:.1f} M unique (bin,hash) pairs "
        f"in {time.time()-t0:.0f}s -> {params.sketch_npz.name}"
    )


def load_store(params: Params):
    """Load the sketch store -> (bins DataFrame, indptr, hashes, counts)."""
    z = np.load(params.sketch_npz)
    bins = pd.read_parquet(params.bins_parquet)
    return bins, z["indptr"], z["hashes"], z["counts"]
