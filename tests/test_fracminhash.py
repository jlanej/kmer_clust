"""The sketch engine must be bit-identical to sourmash."""

import mmh3
import numpy as np
import pytest
import sourmash

from kmer_clust.fracminhash import (
    MMH_SEED,
    _murmur128_low64,
    bin_stats,
    downsample,
    encode_sequence,
    max_hash_for_scaled,
    sketch_codes,
)

RNG = np.random.default_rng(7)


def random_seq(n: int) -> str:
    return "".join(RNG.choice(list("ACGT"), size=n))


@pytest.mark.parametrize("length", [1, 5, 8, 15, 16, 17, 21, 31, 32, 47])
def test_murmur_matches_mmh3(length):
    for _ in range(20):
        data = bytes(RNG.integers(0, 256, size=length, dtype=np.uint8))
        buf = np.frombuffer(data, dtype=np.uint8)
        ours = int(_murmur128_low64(buf, length, MMH_SEED))
        ref = mmh3.hash64(data, seed=42, signed=False)[0]
        assert ours == ref


@pytest.mark.parametrize("k,scaled", [(21, 10), (31, 5), (15, 20)])
def test_sketch_matches_sourmash(k, scaled):
    seq = random_seq(20_000)
    mh = sourmash.MinHash(n=0, ksize=k, scaled=scaled)
    mh.add_sequence(seq)
    _, hashes = sketch_codes(encode_sequence(seq), k, scaled)
    assert set(int(h) for h in hashes) == set(mh.hashes)


def test_lowercase_and_invalid_bases_match_sourmash():
    seq = list(random_seq(5_000))
    for i in RNG.choice(len(seq), size=50, replace=False):
        seq[i] = "N"
    for i in RNG.choice(len(seq), size=500, replace=False):
        seq[i] = seq[i].lower()
    seq = "".join(seq)
    mh = sourmash.MinHash(n=0, ksize=21, scaled=5)
    mh.add_sequence(seq, force=True)
    _, hashes = sketch_codes(encode_sequence(seq), 21, 5)
    assert set(int(h) for h in hashes) == set(mh.hashes)


def test_revcomp_invariance():
    seq = random_seq(10_000)
    rc = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    _, h1 = sketch_codes(encode_sequence(seq), 21, 5)
    _, h2 = sketch_codes(encode_sequence(rc), 21, 5)
    assert set(h1.tolist()) == set(h2.tolist())


def test_downsample_is_subset_and_matches_direct():
    seq = random_seq(50_000)
    codes = encode_sequence(seq)
    _, dense = sketch_codes(codes, 21, 5)
    _, direct = sketch_codes(codes, 21, 50)
    assert set(downsample(dense, 50).tolist()) == set(direct.tolist())


def test_positions_are_kmer_starts():
    seq = random_seq(30_000)
    codes = encode_sequence(seq)
    pos, hashes = sketch_codes(codes, 21, 10)
    mh = sourmash.MinHash(n=0, ksize=21, scaled=10)
    for p in pos[:200]:
        kmer = seq[p : p + 21]
        assert mh.seq_to_hashes(kmer)[0] in set(int(h) for h in hashes)


def test_segment_boundaries_lose_nothing():
    seq = random_seq(100_000)
    codes = encode_sequence(seq)
    _, a = sketch_codes(codes, 21, 10, n_segments=1)
    _, b = sketch_codes(codes, 21, 10, n_segments=17)
    assert sorted(a.tolist()) == sorted(b.tolist())


def test_bin_stats():
    codes = encode_sequence("ACGTNNGGCC" * 10)  # 100 bp
    acgt, gc = bin_stats(codes, 50)
    assert acgt.tolist() == [40, 40]
    assert gc.tolist() == [30, 30]


def test_max_hash_rule_matches_sourmash():
    for scaled in (1, 2, 10, 1000, 100_000):
        mh = sourmash.MinHash(n=0, ksize=21, scaled=scaled)
        assert max_hash_for_scaled(scaled) == mh._max_hash
