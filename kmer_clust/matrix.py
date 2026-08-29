"""Stage: sketch store -> weighted sparse matrix -> randomized SVD.

Design notes:
- Hashes private to a single bin (df < min_df) carry no between-bin signal and
  are the vast majority of euchromatic vocabulary; they are dropped from the
  model, but each bin's private fraction is kept as an interpretive feature.
- Weighting: log1p(within-bin multiplicity) * mild IDF, L2 row-normalized.
  Abundance is kept because low-complexity arrays express identity through
  multiplicity, and IDF is kept mild because ubiquity across bins is already
  informative down-weighting (kmer_dust's representation study).
- The SVD is a randomized subspace iteration on the implicit Gram operator
  X X^T, blocked so nothing larger than an (n_features x block) workspace is ever dense
"""

import time

import numpy as np
import scipy.sparse as sp

from .config import OUT, Params
from .fracminhash import max_hash_for_scaled
from .sketch_run import load_store


def downsample_store(indptr, hashes, counts, scaled: int):
    """Per-bin prefix cut to a coarser scaled (hashes are sorted within bins)."""
    t = np.uint64(max_hash_for_scaled(scaled))
    n = indptr.size - 1
    cuts = np.empty(n + 1, np.int64)
    cuts[0] = 0
    keep_idx = []
    for b in range(n):
        lo, hi = indptr[b], indptr[b + 1]
        c = lo + np.searchsorted(hashes[lo:hi], t, side="right")
        cuts[b + 1] = cuts[b] + (c - lo)
        keep_idx.append(np.arange(lo, c))
    idx = np.concatenate(keep_idx) if keep_idx else np.zeros(0, np.int64)
    return cuts, hashes[idx], counts[idx]


def build_matrix(indptr, hashes, counts, min_df: int = 2):
    """CSR (bins x shared-hash-universe) with weights, plus bin diagnostics.

    Returns (X_weighted_normalized, universe_hashes, df, private_frac).
    """
    universe, inv = np.unique(hashes, return_inverse=True)
    df = np.bincount(inv, minlength=universe.size)
    col_keep = df >= min_df
    n_bins = indptr.size - 1

    # per-bin fraction of distinct vocabulary that is private (dropped)
    row_ids = np.repeat(np.arange(n_bins), np.diff(indptr))
    private = ~col_keep[inv]
    priv_ct = np.bincount(row_ids[private], minlength=n_bins)
    tot_ct = np.maximum(np.diff(indptr), 1)
    private_frac = priv_ct / tot_ct

    new_col = np.cumsum(col_keep) - 1
    keep = col_keep[inv]
    X = sp.csr_matrix(
        (
            counts[keep].astype(np.float32),
            new_col[inv[keep]].astype(np.int32),
            np.concatenate(([0], np.cumsum(np.bincount(row_ids[keep], minlength=n_bins)))),
        ),
        shape=(n_bins, int(col_keep.sum())),
    )
    kept_df = df[col_keep].astype(np.float32)
    idf = np.log1p(n_bins / kept_df).astype(np.float32)
    X.data = np.log1p(X.data)
    X = X @ sp.diags(idf, format="csr", dtype=np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    with np.errstate(divide="ignore"):
        inv_norm = np.where(norms > 0, 1.0 / norms, 0.0).astype(np.float32)
    X = sp.diags(inv_norm, format="csr", dtype=np.float32) @ X
    return X.tocsr(), universe[col_keep], df[col_keep], private_frac


def gram_rsvd(X: sp.csr_matrix, dims: int, n_iter: int = 5, oversample: int = 24,
              seed: int = 42, block: int = 32):
    """Left singular vectors of X via randomized subspace iteration on X X^T.

    Memory never exceeds O(n_features * block) floats. Returns (U*sigma, sigma).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    l = min(dims + oversample, n)
    Xt = X.T.tocsr()

    def op(Q):  # (X X^T) @ Q, blocked over columns of Q
        S = np.empty((n, Q.shape[1]), np.float32)
        for j in range(0, Q.shape[1], block):
            W = Xt @ Q[:, j : j + block]
            S[:, j : j + block] = X @ W
        return S

    Q, _ = np.linalg.qr(rng.standard_normal((n, l)).astype(np.float32))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(op(Q))
    S = op(Q)
    T = Q.T @ S
    T = (T + T.T) / 2
    lam, V = np.linalg.eigh(T)
    order = np.argsort(lam)[::-1][:dims]
    lam = np.maximum(lam[order], 0)
    U = Q @ V[:, order]
    sigma = np.sqrt(lam)
    return (U * sigma[None, :]).astype(np.float32), sigma.astype(np.float32)


def run(params: Params, scaled: int | None = None, tag: str = "") -> dict:
    scaled = scaled or params.embed_scaled
    if scaled < params.base_scaled:
        raise ValueError(
            f"embed scaled={scaled} is denser than the stored base "
            f"scaled={params.base_scaled}; re-sketch with a smaller base_scaled"
        )
    t0 = time.time()
    bins, indptr, hashes, counts = load_store(params)
    if scaled > params.base_scaled:
        indptr, hashes, counts = downsample_store(indptr, hashes, counts, scaled)
    X, universe, df, private_frac = build_matrix(indptr, hashes, counts, params.min_df)
    print(
        f"matrix: {X.shape[0]} bins x {X.shape[1]} shared hashes, "
        f"nnz={X.nnz/1e6:.1f}M (scaled={scaled}, t={time.time()-t0:.0f}s)"
    )
    Z, sigma = gram_rsvd(X, params.svd_dims, seed=params.seed)
    out = params.svd_npz(scaled)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(
        out, Z=Z, sigma=sigma, private_frac=private_frac.astype(np.float32),
        n_shared=np.diff(indptr) - np.round(private_frac * np.diff(indptr)).astype(np.int64),
        meta=np.array([params.k, scaled, params.bin_bp], np.int64),
    )
    print(f"svd: {Z.shape} (sigma[0]={sigma[0]:.1f}) -> {out.name}, t={time.time()-t0:.0f}s")
    return {"X_shape": X.shape, "nnz": X.nnz, "out": out}
