"""FracMinHash sketching, bit-compatible with sourmash.

A k-mer is canonicalized (lexicographic min of forward and reverse complement),
rendered as uppercase ASCII, hashed with MurmurHash3 x64-128 (seed 42, low 64
bits), and kept when hash <= round((2^64-1)/scaled) -- the same rule sourmash
uses, so sketches here are interchangeable with sourmash's and its Jaccard/ANI
theory applies unchanged.

Note: inside numba kernels every operand of uint64 arithmetic must already be
uint64 -- mixing in an int64 silently promotes to float64.
"""

import numpy as np
from numba import njit, prange

MMH_SEED = np.uint64(42)
MAX_HASH = 0xFFFFFFFFFFFFFFFF

# base -> 2-bit code (A0 C1 G2 T3), case-insensitive; anything else is 255
CODE_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    CODE_LUT[_b] = _i
    CODE_LUT[_b + 32] = _i  # lowercase

ASCII_LUT = np.frombuffer(b"ACGT", dtype=np.uint8).copy()

_C1 = np.uint64(0x87C37B91114253D5)
_C2 = np.uint64(0x4CF5AD432745937F)
_F1 = np.uint64(0xFF51AFD7ED558CCD)
_F2 = np.uint64(0xC4CEB9FE1A85EC53)
_U2 = np.uint64(2)
_U3 = np.uint64(3)
_U33 = np.uint64(33)
_U64 = np.uint64(64)


def max_hash_for_scaled(scaled: int) -> int:
    # the sourmash Rust core computes (u64::MAX as f64 / scaled) as u64:
    # float64 division of 2^64, then truncation. Match it bit-for-bit.
    if scaled <= 1:
        return MAX_HASH
    return int(float(2**64) / scaled)


@njit(inline="always")
def _rotl(x, r):
    return (x << r) | (x >> (_U64 - r))


@njit(inline="always")
def _fmix(z):
    z ^= z >> _U33
    z = z * _F1
    z ^= z >> _U33
    z = z * _F2
    z ^= z >> _U33
    return z


@njit(cache=True)
def _murmur128_low64(buf, n, seed):
    """MurmurHash3 x64-128 of buf[:n], returning the first 64 bits."""
    h1 = seed
    h2 = seed
    r31 = np.uint64(31)
    r27 = np.uint64(27)
    r33 = np.uint64(33)
    m5 = np.uint64(5)
    a1 = np.uint64(0x52DCE729)
    a2 = np.uint64(0x38495AB5)
    nblocks = n // 16
    for b in range(nblocks):
        o = b * 16
        k1 = np.uint64(0)
        k2 = np.uint64(0)
        for j in range(8):
            sh = np.uint64(8 * j)
            k1 |= np.uint64(buf[o + j]) << sh
            k2 |= np.uint64(buf[o + 8 + j]) << sh
        k1 = k1 * _C1
        k1 = _rotl(k1, r31)
        k1 = k1 * _C2
        h1 ^= k1
        h1 = _rotl(h1, r27)
        h1 = h1 + h2
        h1 = h1 * m5 + a1
        k2 = k2 * _C2
        k2 = _rotl(k2, r33)
        k2 = k2 * _C1
        h2 ^= k2
        h2 = _rotl(h2, r31)
        h2 = h2 + h1
        h2 = h2 * m5 + a2
    o = nblocks * 16
    tail = n - o
    if tail > 8:
        k2 = np.uint64(0)
        for j in range(tail - 8):
            k2 |= np.uint64(buf[o + 8 + j]) << np.uint64(8 * j)
        k2 = k2 * _C2
        k2 = _rotl(k2, r33)
        k2 = k2 * _C1
        h2 ^= k2
    if tail > 0:
        k1 = np.uint64(0)
        for j in range(min(tail, 8)):
            k1 |= np.uint64(buf[o + j]) << np.uint64(8 * j)
        k1 = k1 * _C1
        k1 = _rotl(k1, r31)
        k1 = k1 * _C2
        h1 ^= k1
    h1 ^= np.uint64(n)
    h2 ^= np.uint64(n)
    h1 = h1 + h2
    h2 = h2 + h1
    h1 = _fmix(h1)
    h2 = _fmix(h2)
    h1 = h1 + h2
    return h1


@njit(cache=True)
def _hash_canonical_code(canon, k, buf):
    for i in range(k):
        sh = np.uint64(2 * (k - i - 1))
        buf[i] = ASCII_LUT[(canon >> sh) & _U3]
    return _murmur128_low64(buf, k, MMH_SEED)


@njit(parallel=True, cache=True)
def _sketch_kernel(
    codes, k, max_hash, seg_starts, seg_ends, out_pos, out_hash, slab, caps, counts
):
    """Emit (position, hash) for canonical k-mers with hash <= max_hash.

    Segments run in parallel; segment s writes out_*[slab[s]:slab[s]+counts[s]].
    A k-mer belongs to the segment containing its start position, so segment
    boundaries lose nothing. counts[s] == caps[s] signals slab overflow.
    """
    n = codes.shape[0]
    mask = np.uint64((1 << (2 * k)) - 1)
    shift_rc = np.uint64(2 * (k - 1))
    for s in prange(seg_starts.shape[0]):
        lo = seg_starts[s]
        hi = seg_ends[s]
        buf = np.empty(64, dtype=np.uint8)
        w = slab[s]
        w_max = slab[s] + caps[s]
        f = np.uint64(0)
        r = np.uint64(0)
        run = 0
        end = min(hi + k - 1, n)
        for i in range(lo, end):
            c = codes[i]
            if c > 3:
                run = 0
                continue
            cc = np.uint64(c)
            f = ((f << _U2) | cc) & mask
            r = (r >> _U2) | ((_U3 - cc) << shift_rc)
            run += 1
            if run >= k:
                start = i - k + 1
                if start >= hi:
                    break
                canon = f if f < r else r
                h = _hash_canonical_code(canon, k, buf)
                if h <= max_hash and w < w_max:
                    out_pos[w] = start
                    out_hash[w] = h
                    w += 1
        counts[s] = w - slab[s]


def sketch_codes(codes: np.ndarray, k: int, scaled: int, n_segments: int = 40):
    """FracMinHash one sequence (2-bit codes, 255=invalid).

    Returns (positions uint32, hashes uint64), position-sorted.
    """
    n = codes.shape[0]
    if n < k:
        return np.empty(0, np.uint32), np.empty(0, np.uint64)
    max_hash = np.uint64(max_hash_for_scaled(scaled))
    n_segments = int(max(1, min(n_segments, n // max(k * 4, 1) + 1)))
    bounds = np.linspace(0, n, n_segments + 1).astype(np.int64)
    seg_starts, seg_ends = bounds[:-1].copy(), bounds[1:].copy()
    # generous per-segment capacity: keep rate is ~1/scaled per position
    caps = ((seg_ends - seg_starts) // scaled) * 3 + 4096
    slab = np.concatenate(([0], np.cumsum(caps)))[:-1]
    total = int(caps.sum())
    out_pos = np.empty(total, np.uint32)
    out_hash = np.empty(total, np.uint64)
    counts = np.zeros(n_segments, np.int64)
    _sketch_kernel(
        codes, k, max_hash, seg_starts, seg_ends, out_pos, out_hash, slab, caps, counts
    )
    if np.any(counts >= caps):  # pragma: no cover - astronomically unlikely
        raise RuntimeError("sketch slab overflow; raise capacity margin")
    keep_pos = np.concatenate(
        [out_pos[slab[s] : slab[s] + counts[s]] for s in range(n_segments)]
    )
    keep_hash = np.concatenate(
        [out_hash[slab[s] : slab[s] + counts[s]] for s in range(n_segments)]
    )
    return keep_pos, keep_hash


@njit(parallel=True, cache=True)
def _bin_stats_kernel(codes, bin_bp, acgt, gc):
    n = codes.shape[0]
    for b in prange(acgt.shape[0]):
        lo = b * bin_bp
        hi = min(lo + bin_bp, n)
        a_ct = 0
        g_ct = 0
        for i in range(lo, hi):
            c = codes[i]
            if c <= 3:
                a_ct += 1
                if c == 1 or c == 2:
                    g_ct += 1
        acgt[b] = a_ct
        gc[b] = g_ct


def bin_stats(codes: np.ndarray, bin_bp: int):
    """Per-bin ACGT and G+C counts."""
    n_bins = (codes.shape[0] + bin_bp - 1) // bin_bp
    acgt = np.zeros(n_bins, np.int64)
    gc = np.zeros(n_bins, np.int64)
    _bin_stats_kernel(codes, bin_bp, acgt, gc)
    return acgt, gc


def encode_sequence(seq: bytes | str) -> np.ndarray:
    """ASCII sequence -> 2-bit codes (255 = non-ACGT)."""
    if isinstance(seq, str):
        seq = seq.encode()
    return CODE_LUT[np.frombuffer(seq, dtype=np.uint8)]


def downsample(hashes: np.ndarray, scaled: int) -> np.ndarray:
    """Subset a sketch to a coarser scaled (FracMinHash containment property)."""
    return hashes[hashes <= np.uint64(max_hash_for_scaled(scaled))]
