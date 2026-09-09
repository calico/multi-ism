#!/usr/bin/env python3
"""Fast unit tests for Tier 1 helpers in ``mism.core.mism_utils``.

Covers the pure / lightweight functions that need no model, GPU, or genome:
    - coef2score
    - mut_vector_to_1hot
    - make_alt
    - get_mutation_region
    - coo_list2csr

Run with:  pytest tests/test_utils.py
"""

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix

from mism.core.mism_utils import (
    coef2score,
    get_mutation_region,
    make_alt,
    mut_vector_to_1hot,
)
from mism.core.mutation_designer import coo_list2csr

# A, C, G, T one-hot rows for convenient sequence construction
_ONEHOT = np.eye(4, dtype="float32")


def _seq_1hot(bases):
    """Build an (L, 4) one-hot array from a base string like 'ACGT'."""
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.stack([_ONEHOT[idx[b]] for b in bases])


# --------------------------------------------------------------------------- #
# coef2score
# --------------------------------------------------------------------------- #
def test_coef2score_layout():
    # ref = A, C  ->  each position has 3 non-ref coefficients laid out in order
    ref = _seq_1hot("AC")
    coefs = np.array([[1, 2, 3, 4, 5, 6]], dtype="float32")  # (target_num=1, L*3=6)

    scores = coef2score(ref, coefs, mut_len=2, target_num=1)

    assert scores.shape == (2, 4, 1)
    # pos0 ref A: [ref=0, C=1, G=2, T=3]
    assert np.allclose(scores[0, :, 0], [0, 1, 2, 3])
    # pos1 ref C: [A=4, ref=0, G=5, T=6]
    assert np.allclose(scores[1, :, 0], [4, 0, 5, 6])


def test_coef2score_skips_n_position():
    # middle position is 'N' (all-zero ref row): its score stays 0 but the
    # coefficient index must still advance by 3 to keep later positions aligned.
    ref = np.stack([_ONEHOT[0], np.zeros(4, "float32"), _ONEHOT[1]])  # A, N, C
    coefs = np.arange(1, 10, dtype="float32").reshape(1, 9)  # (1, L*3=9)

    scores = coef2score(ref, coefs, mut_len=3, target_num=1)

    assert np.allclose(scores[1, :, 0], 0)  # N position untouched
    # pos0 ref A consumes coefs 1,2,3; N consumes 4,5,6; pos2 ref C -> A=7,G=8,T=9
    assert np.allclose(scores[2, :, 0], [7, 0, 8, 9])


def test_coef2score_normalize():
    ref = _seq_1hot("A")
    coefs = np.array([[3, 6, 9]], dtype="float32")
    scores = coef2score(ref, coefs, mut_len=1, target_num=1, normalize=True)
    # mean across the 4 nucleotides is subtracted at each position
    assert np.allclose(scores.mean(axis=1), 0)


# --------------------------------------------------------------------------- #
# mut_vector_to_1hot
# --------------------------------------------------------------------------- #
def test_mut_vector_to_1hot_applies_alts():
    ref = _seq_1hot("ACG")
    # pos0 -> alt 1 of A ([C,G,T]) = G ; pos1 no mut ; pos2 -> alt 0 of G ([A,C,T]) = A
    mut_vec = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    out = mut_vector_to_1hot(mut_vec.ravel(), ref)
    assert np.array_equal(out, _seq_1hot("GCA"))


def test_mut_vector_to_1hot_flattened_input():
    ref = _seq_1hot("AT")
    # pos0 no mut ; pos1 -> alt 2 of T ([A,C,G]) = G
    out = mut_vector_to_1hot(np.array([0, 0, 0, 0, 0, 1]), ref)
    assert np.array_equal(out, _seq_1hot("AG"))


def test_mut_vector_to_1hot_no_mutation_returns_reference():
    ref = _seq_1hot("ACGT")
    out = mut_vector_to_1hot(np.zeros(12), ref)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize(
    "ref_base, alts",
    [
        ("A", "CGT"),
        ("C", "AGT"),
        ("G", "ACT"),
        ("T", "ACG"),
    ],
)
def test_mut_vector_to_1hot_all_alternatives(ref_base, alts):
    # each of the 3 alt indices must map to the correct non-reference base
    ref = _seq_1hot(ref_base)
    for alt_idx, expected in enumerate(alts):
        mut_vec = np.zeros(3)
        mut_vec[alt_idx] = 1
        out = mut_vector_to_1hot(mut_vec, ref)
        assert np.array_equal(out, _seq_1hot(expected))


@pytest.mark.parametrize(
    "vec",
    [
        np.zeros((2, 3)),  # not 1D
        np.zeros(5),  # wrong length (L*3 = 6)
        np.array([1, 1, 0, 0, 0, 0]),  # >1 mutation at position 0
    ],
)
def test_mut_vector_to_1hot_validation(vec):
    ref = _seq_1hot("AC")
    with pytest.raises(ValueError):
        mut_vector_to_1hot(vec, ref)


# --------------------------------------------------------------------------- #
# make_alt
# --------------------------------------------------------------------------- #
def _ref_1hot_torch(bases):
    """Build a (1, 4, L) torch reference tensor from a base string."""
    return torch.tensor(_seq_1hot(bases).T[None], dtype=torch.float32)


def test_make_alt_dense():
    ref = _ref_1hot_torch("ACGT")
    # mutate within region starting at index 1 (covers C,G,T)
    # local pos0 (C) -> alt 0 of C ([A,G,T]) = A ; local pos2 (T) -> alt 2 ([A,C,G]) = G
    X_mut3 = np.zeros((3, 3), dtype="float32")
    X_mut3[0, 0] = 1
    X_mut3[2, 2] = 1

    alt = make_alt(ref, mut_start=1, X_mut3=X_mut3)
    alt_np = alt[0].numpy().T  # (L, 4)
    assert np.array_equal(alt_np, _seq_1hot("AAGG"))


def test_make_alt_sparse_matches_dense():
    ref = _ref_1hot_torch("ACGT")
    X_dense = np.zeros((3, 3), dtype="float32")
    X_dense[0, 0] = 1
    X_dense[2, 2] = 1
    X_sparse = coo_matrix(X_dense)

    alt_dense = make_alt(ref, 1, X_dense)
    alt_sparse = make_alt(ref, 1, X_sparse)
    assert torch.equal(alt_dense, alt_sparse)


def test_make_alt_does_not_mutate_reference():
    ref = _ref_1hot_torch("ACGT")
    ref_copy = ref.clone()
    X_mut3 = np.zeros((3, 3), dtype="float32")
    X_mut3[0, 0] = 1
    make_alt(ref, 1, X_mut3)
    assert torch.equal(ref, ref_copy)  # input tensor untouched


# --------------------------------------------------------------------------- #
# get_mutation_region
# --------------------------------------------------------------------------- #
def test_get_mutation_region_centered():
    start, end = get_mutation_region(seq_length=100, mut_len=10)
    assert (start, end) == (45, 55)


def test_get_mutation_region_offset():
    start, end = get_mutation_region(seq_length=100, mut_len=10, mut2center=5)
    assert (start, end) == (50, 60)


def test_get_mutation_region_out_of_bounds():
    with pytest.raises(ValueError):
        get_mutation_region(seq_length=10, mut_len=20)


# --------------------------------------------------------------------------- #
# coo_list2csr
# --------------------------------------------------------------------------- #
def test_coo_list2csr_layout():
    # per-sequence (pos, nt) mutation lands in flattened column pos*3 + nt
    mut_len = 4
    # seq0: pos1 nt2, pos3 nt0 ; seq1: pos0 nt1
    dense = np.zeros((2, mut_len, 3), dtype=bool)
    dense[0, 1, 2] = True
    dense[0, 3, 0] = True
    dense[1, 0, 1] = True

    coo_list = [coo_matrix(dense[i]) for i in range(dense.shape[0])]
    X_mut = coo_list2csr(coo_list)

    assert X_mut.shape == (2, mut_len * 3)
    expected = dense.reshape(2, mut_len * 3)
    assert np.array_equal(X_mut.toarray().astype(bool), expected)


def test_coo_list2csr_matches_dense_random():
    rng = np.random.default_rng(0)
    num_seq, mut_len = 25, 60
    dense = np.zeros((num_seq, mut_len, 3), dtype=bool)
    for s in range(num_seq):
        for _ in range(rng.integers(3, 9)):
            dense[s, rng.integers(mut_len), rng.integers(3)] = True

    coo_list = [coo_matrix(dense[s]) for s in range(num_seq)]
    X_mut = coo_list2csr(coo_list)

    assert X_mut.nnz == dense.sum()
    assert np.array_equal(
        X_mut.toarray().astype(bool), dense.reshape(num_seq, mut_len * 3)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
