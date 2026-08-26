"""Toy-scale checks of the matrix, pairwise, and annotation math."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kmer_clust.annotate import censat_class, coverage
from kmer_clust.matrix import build_matrix, downsample_store, gram_rsvd
from kmer_clust.pairwise import pairwise_distances
from kmer_clust.sketch_run import aggregate_bin_hashes


def test_aggregate_bin_hashes():
    bins = np.array([1, 0, 1, 1, 0])
    hashes = np.array([7, 3, 7, 5, 3], dtype=np.uint64)
    indptr, h, c = aggregate_bin_hashes(bins, hashes, 3)
    assert indptr.tolist() == [0, 1, 3, 3]
    assert h.tolist() == [3, 5, 7]
    assert c.tolist() == [2, 1, 2]


def test_build_matrix_drops_private_columns():
    # bin0: {1,2}, bin1: {2,3}, bin2: {9}
    indptr = np.array([0, 2, 4, 5])
    hashes = np.array([1, 2, 2, 3, 9], dtype=np.uint64)
    counts = np.array([1, 4, 2, 1, 1], dtype=np.uint32)
    X, universe, df, private = build_matrix(indptr, hashes, counts, min_df=2)
    assert universe.tolist() == [2]
    assert df.tolist() == [2]
    assert X.shape == (3, 1)
    np.testing.assert_allclose(private, [0.5, 0.5, 1.0])
    # rows with any shared vocabulary are unit-normalized
    assert np.isclose(X[0, 0], 1.0) and np.isclose(X[1, 0], 1.0)
    assert X[2].nnz == 0


def test_pairwise_distances_exact():
    # A={1,2,3,4}, B={3,4,5,6}, C subset of A: {1,2}
    indptr = np.array([0, 4, 8, 10])
    hashes = np.array([1, 2, 3, 4, 3, 4, 5, 6, 1, 2], dtype=np.uint64)
    jacc, cani, sizes = pairwise_distances(indptr, hashes, k=21)
    assert sizes.tolist() == [4, 4, 2]
    assert np.isclose(jacc[0, 1], 2 / 6)
    assert np.isclose(jacc[0, 2], 2 / 4)
    # C fully contained in A -> max-containment 1 -> cANI 1
    assert np.isclose(cani[0, 2], 1.0)
    assert np.isclose(cani[0, 1], (2 / 4) ** (1 / 21))


def test_downsample_store_prefix_cut():
    from kmer_clust.fracminhash import max_hash_for_scaled

    t50 = np.uint64(max_hash_for_scaled(50))
    indptr = np.array([0, 3, 5])
    hashes = np.array([10, 20, t50 + 5, 15, t50 + 99], dtype=np.uint64)
    counts = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    cuts, h, c = downsample_store(indptr, hashes, counts, 50)
    assert cuts.tolist() == [0, 2, 3]
    assert h.tolist() == [10, 20, 15]
    assert c.tolist() == [1, 2, 4]


def test_gram_rsvd_matches_dense_svd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 300)).astype(np.float32)
    A[:30] += 3.0  # give it structure
    X = sp.csr_matrix(A)
    Z, sigma = gram_rsvd(X, dims=5, n_iter=7, seed=1)
    _, s_ref, _ = np.linalg.svd(A, full_matrices=False)
    np.testing.assert_allclose(sigma, s_ref[:5], rtol=1e-3)
    # row-space geometry preserved: pairwise dots of Z match A A^T top-rank
    G = Z @ Z.T
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    G_ref = (U[:, :5] * S[:5]) @ (U[:, :5] * S[:5]).T
    np.testing.assert_allclose(G, G_ref, atol=1e-2 * S[0] ** 2)


def test_censat_class_rules():
    assert censat_class("hor_1_5(S1C1/5/19H1L)") == "asat_hor_live"
    assert censat_class("hor_9_2(S4C9H2)") == "asat_hor"
    assert censat_class("dhor_22_1(S2C22H2_d)") == "asat_dhor"
    assert censat_class("mon_13_1(mixedAlp)") == "asat_mon"
    assert censat_class("ct_1_1(p_arm)") == "ct"
    assert censat_class("censat_1_1(rnd-6_family-4384)") == "other_sat"
    assert censat_class("hsat1A_13_2") == "hsat1A"
    assert censat_class("rDNA_13_1") == "rDNA"
    assert censat_class("hsat3_X_1(x)") == "hsat3"
    assert censat_class("mystery_1_1(z)") == ""


def test_coverage_spanning_intervals():
    bins = pd.DataFrame(
        {"chrom": ["c1", "c1", "c2"], "start": [0, 100, 0], "end": [100, 200, 100]}
    )
    offsets = {"c1": (0, 200), "c2": (2, 100)}
    cov = coverage(
        bins, offsets, 100,
        chroms=["c1", "c1", "c2", "cX"],
        starts=[10, 90, 0, 0],
        ends=[20, 120, 100, 50],
        groups=["a", "a", "b", "a"],
        group_names=["a", "b"],
    )
    np.testing.assert_allclose(cov[0], [0.2, 0.2, 0.0])  # 10+10 in bin0, 20 in bin1
    np.testing.assert_allclose(cov[1], [0.0, 0.0, 1.0])
