#!/usr/bin/env python3
"""Correctness tests for MutationDesigner design methods.

Run with:  pytest tests/test_designer/test_designer.py -v

These are fast, CPU-only tests on small windows that assert the structural
guarantees of each design method (coverage, spacing, sequence counts).
The heavier scaling benchmark lives in benchmark_designer.py.
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix, issparse

from mism.core.mutation_designer import (
    MutationDesigner,
    mutation_design_matrix_vanilla,
    mutation_design_matrix_bins,
    mutation_design_matrix_uniform_force,
    mutation_design_random,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _iter_seq_positions(X_mut3):
    """Yield (positions, cols) arrays for each designed sequence.

    Handles both the dense ndarray output (vanilla) and the list-of-coo output
    (bins / star-and-bar / random).
    """
    if isinstance(X_mut3, np.ndarray):
        for i in range(X_mut3.shape[0]):
            rows, cols = np.where(X_mut3[i])
            yield rows, cols
    else:
        for seq in X_mut3:
            coo = seq if hasattr(seq, "row") else seq.tocoo()
            yield coo.row, coo.col


def _mut_coverage(X_mut3, mut_len):
    """Return an (mut_len, 3) int array of how many times each (pos, nt) is used."""
    cov = np.zeros((mut_len, 3), dtype=np.int64)
    for rows, cols in _iter_seq_positions(X_mut3):
        np.add.at(cov, (rows, cols), 1)
    return cov


def _min_within_seq_distance(X_mut3):
    """Smallest gap between two mutated positions within any single sequence."""
    min_d = np.inf
    for rows, _ in _iter_seq_positions(X_mut3):
        if len(rows) > 1:
            d = np.diff(np.sort(rows))
            if d.size:
                min_d = min(min_d, int(d.min()))
    return min_d


# --------------------------------------------------------------------------- #
# vanilla
# --------------------------------------------------------------------------- #
def test_vanilla_shape_and_single_mutation():
    mut_len = 200
    X = mutation_design_matrix_vanilla(mut_len)
    # exactly 3 * L sequences, dense boolean tensor
    assert X.shape == (3 * mut_len, mut_len, 3)
    assert X.dtype == bool
    # each sequence carries exactly one mutation
    per_seq = X.reshape(3 * mut_len, -1).sum(axis=1)
    assert np.all(per_seq == 1)


def test_vanilla_covers_every_position_and_alt_once():
    mut_len = 128
    X = mutation_design_matrix_vanilla(mut_len)
    cov = _mut_coverage(X, mut_len)
    # every (position, alt-nucleotide) appears exactly once
    assert np.all(cov == 1)


# --------------------------------------------------------------------------- #
# bins
# --------------------------------------------------------------------------- #
def test_bins_exact_coverage():
    mut_len, mut_reps, min_dist = 300, 5, 50
    X = mutation_design_matrix_bins(mut_len, mut_reps, min_dist)
    cov = _mut_coverage(X, mut_len)
    # every (pos, nt) is sampled exactly mut_reps times
    assert np.all(cov == mut_reps)


def test_bins_respects_min_distance():
    mut_len, mut_reps, min_dist = 300, 5, 50
    X = mutation_design_matrix_bins(mut_len, mut_reps, min_dist)
    # mutations in one sequence come from alternating (same-parity) bins,
    # so any two must be at least min_dist apart
    assert _min_within_seq_distance(X) >= min_dist


def test_bins_outputs_sparse_sequences():
    X = mutation_design_matrix_bins(200, 3, 50)
    assert isinstance(X, list)
    assert all(issparse(s) for s in X)


# --------------------------------------------------------------------------- #
# star-and-bar (uniform_force)
# --------------------------------------------------------------------------- #
def test_star_and_bar_reaches_target_coverage():
    mut_len, mut_reps, min_dist = 400, 5, 50
    X = mutation_design_matrix_uniform_force(mut_len, mut_reps, min_dist)
    pos_cov = _mut_coverage(X, mut_len).sum(axis=1)
    # forced-coverage guarantees every position hits at least mut_reps * 3
    assert pos_cov.min() >= mut_reps * 3


def test_star_and_bar_respects_min_distance():
    mut_len, mut_reps, min_dist = 400, 5, 50
    X = mutation_design_matrix_uniform_force(mut_len, mut_reps, min_dist)
    assert _min_within_seq_distance(X) >= min_dist


def test_star_and_bar_infeasible_raises():
    # n_mutation forced too large for the window -> ValueError
    with pytest.raises(ValueError):
        mutation_design_matrix_uniform_force(
            100, mut_reps=5, min_dist=50, n_mutation=50
        )


# --------------------------------------------------------------------------- #
# random
# --------------------------------------------------------------------------- #
def test_random_sequence_count_and_rate():
    mut_len, mut_rate, num_seq = 1000, 0.01, 500
    np.random.seed(0)
    X = mutation_design_random(mut_len, mut_rate, num_seq)
    assert len(X) == num_seq
    per_seq = np.array([s.nnz for s in X])
    # mean mutations per sequence should be near mut_rate * mut_len (=10)
    assert abs(per_seq.mean() - mut_rate * mut_len) < 3.0


# --------------------------------------------------------------------------- #
# public API dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["vanilla", "bins", "star-and-bar", "random"])
def test_designer_api_dispatch(method):
    mut_len = 300
    params = {"mut_reps": 3, "mut_distance_min": 50, "mut_rate": 0.02, "num_seq": 100}
    designer = MutationDesigner()
    designer.design_initial_batch(mut_len, method, params)

    X_mut3 = designer.get_mutation_matrix()
    X_mut = designer.get_regression_matrix()

    assert designer.get_mutation_count() > 0
    # flattened regression matrix has 3 * mut_len columns
    assert X_mut.shape[1] == 3 * mut_len
    assert X_mut.shape[0] == designer.get_mutation_count()
    # mutation matrix count matches
    n = X_mut3.shape[0] if isinstance(X_mut3, np.ndarray) else len(X_mut3)
    assert n == designer.get_mutation_count()


def test_designer_invalid_method_raises():
    designer = MutationDesigner()
    with pytest.raises(ValueError):
        designer.design_initial_batch(100, "does-not-exist", {})
