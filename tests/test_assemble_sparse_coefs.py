#!/usr/bin/env python3
"""Tests for ``assemble_sparse_coefs`` in ``mism.core.mism_utils``.

The helper replaces the old dense ``coefs_full`` (T, p) construction in
``scripts/design/ism_regressor.py``. These tests pin it to the exact legacy
behavior: build the dense array, cast to float16, ``.to_sparse()``.

Run with:  pytest tests/test_assemble_sparse_coefs.py
"""

import numpy as np
import torch

from mism.core.mism_utils import assemble_sparse_coefs


def _legacy_sparse(
    coefs_subset, overlapping_coef_indices, p, ism_idx=None, ism_values=None
):
    """Reference dense implementation: scatter, cast to float16, ``to_sparse()``."""
    T = coefs_subset.shape[0]
    coefs_full = np.zeros((T, p), dtype=np.float16)
    coefs_full[:, overlapping_coef_indices] = coefs_subset
    if ism_idx is not None:
        coefs_full[:, ism_idx] = ism_values
    dense = torch.tensor(coefs_full, dtype=torch.float16)
    return dense.to_sparse()


def _assert_same(a, b):
    """Compare two sparse tensors by their densified float16 contents."""
    assert a.shape == b.shape
    assert torch.equal(a.coalesce().to_dense(), b.coalesce().to_dense())


def _random_sparse_coefs(T, n_overlap, density, rng):
    """(T, n_overlap) float64 matrix with a fraction ``density`` nonzero."""
    m = rng.standard_normal((T, n_overlap))
    mask = rng.random((T, n_overlap)) > density
    m[mask] = 0.0
    return m


def test_matches_legacy_no_ism():
    rng = np.random.default_rng(0)
    T, p, n_overlap = 6, 40, 34
    overlapping = rng.choice(p, size=n_overlap, replace=False)
    overlapping.sort()
    coefs_subset = _random_sparse_coefs(T, n_overlap, density=0.3, rng=rng)

    got = assemble_sparse_coefs(coefs_subset, overlapping, p)
    exp = _legacy_sparse(coefs_subset, overlapping, p)
    _assert_same(got, exp)


def test_matches_legacy_with_ism_overwrite():
    rng = np.random.default_rng(1)
    T, p, n_overlap = 5, 50, 45
    overlapping = rng.choice(p, size=n_overlap, replace=False)
    overlapping.sort()
    coefs_subset = _random_sparse_coefs(T, n_overlap, density=0.4, rng=rng)

    # ISM-select columns are a subset of the overlapping columns (as in the
    # pipeline) so the overwrite genuinely replaces existing Lasso entries.
    ism_idx = np.sort(rng.choice(overlapping, size=6, replace=False))
    ism_values = rng.standard_normal((T, ism_idx.size)).astype(np.float32)

    got = assemble_sparse_coefs(
        coefs_subset, overlapping, p, ism_idx=ism_idx, ism_values=ism_values
    )
    exp = _legacy_sparse(
        coefs_subset, overlapping, p, ism_idx=ism_idx, ism_values=ism_values
    )
    _assert_same(got, exp)


def test_ism_zero_values_are_dropped():
    # An ISM column whose value is 0 must remove the underlying Lasso entry,
    # matching the dense overwrite followed by to_sparse().
    T, p = 3, 10
    overlapping = np.arange(p)
    coefs_subset = np.zeros((T, p), dtype=np.float64)
    coefs_subset[:, 4] = 2.0  # nonzero Lasso entry at global col 4
    ism_idx = np.array([4])
    ism_values = np.zeros((T, 1), dtype=np.float32)  # overwrite col 4 with 0

    got = assemble_sparse_coefs(
        coefs_subset, overlapping, p, ism_idx=ism_idx, ism_values=ism_values
    )
    exp = _legacy_sparse(
        coefs_subset, overlapping, p, ism_idx=ism_idx, ism_values=ism_values
    )
    _assert_same(got, exp)
    assert got.coalesce().values().numel() == 0


def test_all_zero_coefs():
    T, p = 4, 12
    overlapping = np.arange(8)
    coefs_subset = np.zeros((T, overlapping.size), dtype=np.float64)

    got = assemble_sparse_coefs(coefs_subset, overlapping, p)
    exp = _legacy_sparse(coefs_subset, overlapping, p)
    _assert_same(got, exp)
    assert got.shape == (T, p)


def test_dtype_and_shape():
    rng = np.random.default_rng(2)
    T, p, n_overlap = 7, 30, 25
    overlapping = np.sort(rng.choice(p, size=n_overlap, replace=False))
    coefs_subset = _random_sparse_coefs(T, n_overlap, density=0.5, rng=rng)

    got = assemble_sparse_coefs(coefs_subset, overlapping, p)
    assert got.dtype == torch.float16
    assert got.shape == (T, p)
    assert got.is_sparse
