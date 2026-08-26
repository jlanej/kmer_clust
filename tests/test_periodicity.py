"""Periodicity detector on synthetic tandem arrays."""

import numpy as np

from kmer_clust.fracminhash import encode_sequence, sketch_codes
from kmer_clust.periodicity import bin_periods

RNG = np.random.default_rng(3)


def test_tandem_array_period_recovered():
    unit = "".join(RNG.choice(list("ACGT"), size=2000))
    seq = unit * 40  # 80 kb array, 2 kb unit
    codes = encode_sequence(seq)
    pos, hashes = sketch_codes(codes, 21, 5)
    period, strength = bin_periods(pos, hashes, n_bins=1, bin_bp=100_000)
    assert strength[0] > 0.6
    assert abs(period[0] - 2000) / 2000 < 0.05


def test_random_sequence_is_aperiodic():
    seq = "".join(RNG.choice(list("ACGT"), size=100_000))
    codes = encode_sequence(seq)
    pos, hashes = sketch_codes(codes, 21, 5)
    period, strength = bin_periods(pos, hashes, n_bins=1, bin_bp=100_000)
    assert strength[0] < 0.3


def test_two_bins_independent():
    unit = "".join(RNG.choice(list("ACGT"), size=500))
    array = unit * 100                      # 50 kb of 500 bp tandem
    rand = "".join(RNG.choice(list("ACGT"), size=50_000))
    codes = encode_sequence(array + rand * 2)  # bin0 = array+rand, bin1 = rand
    pos, hashes = sketch_codes(codes, 21, 5)
    period, strength = bin_periods(pos, hashes, n_bins=2, bin_bp=100_000)
    assert strength[0] > 0.4 and abs(period[0] - 500) / 500 < 0.1
    assert strength[1] < 0.3
