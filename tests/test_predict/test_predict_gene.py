#!/usr/bin/env python
# Copyright 2025 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""
test_predict_gene.py

Tests for the three core inference helpers in mism.core.mism_utils:
  - create_seqnn_model
  - predict_gene            (used when model.ensemble_rc=True)
  - predict_gene_with_rc    (used when model.ensemble_rc=False, forward/reverse)

Input wiring mirrors tests/example_active_ism.sh and scripts/inference/gene_pred.py.

Requires a CUDA GPU and the `hydra` conda env (where baskerville_torch imports).
Run directly:
    python tests/test_predict/test_predict_gene.py
or via pytest:
    pytest tests/test_predict/test_predict_gene.py -v -s
Data-file locations can be overridden with the env vars listed in DATA_PATHS below.
"""
import os

import numpy as np
import pandas as pd
import pysam
import pytest
import torch

from baskerville_torch import dataset
from baskerville_torch import dna
from baskerville_torch import gene as bgene
from mism.core.mism_utils import (
    create_seqnn_model,
    predict_gene,
    predict_gene_with_rc,
    set_bounds,
)

################################################################################
# data paths (override via environment variables)
################################################################################
_DATA = os.environ.get("MISM_DATA", "/path/to/mism_data")

DATA_PATHS = {
    "params_file": os.environ.get(
        "MISM_PARAMS", f"{_DATA}/model_flashzoi/pretrained_f0c0_params.json"
    ),
    "model_file": os.environ.get(
        "MISM_MODEL", f"{_DATA}/model_flashzoi/f0c0/train/model_best.pth"
    ),
    "targets_file": os.environ.get(
        "MISM_TARGETS", f"{_DATA}/model_flashzoi/targets_human.txt"
    ),
    "fasta_file": os.environ.get("MISM_FASTA", "/path/to/genome.fa"),
    "gtf_file": os.environ.get("MISM_GTF", f"{_DATA}/gtf/GATA1.gtf"),
}

MIX_DTYPE = os.environ.get("MISM_MIX_DTYPE", "bfloat16")

# golden reference output (GATA1). Regenerate by setting MISM_REGEN_GOLDEN=1.
GOLDEN_PATH = os.environ.get(
    "MISM_GOLDEN", os.path.join(os.path.dirname(__file__), "golden_gata1.npz")
)
REGEN_GOLDEN = os.environ.get("MISM_REGEN_GOLDEN", "0") == "1"
# bfloat16 + GPU kernel nondeterminism tolerance
GOLDEN_RTOL = float(os.environ.get("MISM_GOLDEN_RTOL", "1e-2"))
GOLDEN_ATOL = float(os.environ.get("MISM_GOLDEN_ATOL", "1e-2"))


################################################################################
# helpers
################################################################################
def _require_files():
    """Skip (pytest) or raise (direct) if any required data file is missing."""
    missing = [name for name, path in DATA_PATHS.items() if not os.path.exists(path)]
    if missing:
        detail = "\n".join(f"  {n}: {DATA_PATHS[n]}" for n in missing)
        raise FileNotFoundError(f"Missing required data files:\n{detail}")


def _build_gene_inputs(seqnn_model, targets_df):
    """Build (seq_1hot_tensor, gene_strand_mask, gene_slice) for the first gene
    in the GTF, following scripts/inference/gene_pred.py."""
    model_stride = seqnn_model.output_stride()
    model_one_side_crop_bp = seqnn_model.output_crop_bp()
    target_length = seqnn_model.output_length()
    seq_len = seqnn_model.seq_length

    transcriptome = bgene.Transcriptome(DATA_PATHS["gtf_file"])
    assert len(transcriptome.genes) >= 1, "GTF contains no genes"
    gene_id, gene = next(iter(transcriptome.genes.items()))

    set_bounds(gene, seq_len)

    if gene.strand == "+":
        gene_strand_mask = (targets_df.strand != "-").to_numpy()
    else:
        gene_strand_mask = (targets_df.strand != "+").to_numpy()

    fasta_open = pysam.Fastafile(DATA_PATHS["fasta_file"])
    ref_dna = fasta_open.fetch(gene.chrom, gene.seq_start, gene.seq_end)
    fasta_open.close()
    if len(ref_dna) < seq_len:
        ref_dna += "N" * (seq_len - len(ref_dna))

    ref_1hot = dna.dna_1hot(ref_dna)
    ref_1hot = np.expand_dims(ref_1hot.T, axis=0)
    ref_1hot_tensor = torch.tensor(
        ref_1hot, device=seqnn_model.device, dtype=torch.float32
    )

    seq_out_start = gene.seq_start + model_one_side_crop_bp
    seq_out_len = model_stride * target_length
    gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride)

    return gene_id, ref_1hot_tensor, gene_strand_mask, gene_slice


def _compute_gene_scores(model_ensemble, model_single, targets_df):
    """Compute deterministic GATA1 gene scores for the golden reference.

    Returns dict of 1-D arrays: 'ensemble', 'forward', 'reverse'.
    """
    gene_id, seq_1hot, gene_strand_mask, gene_slice = _build_gene_inputs(
        model_ensemble, targets_df
    )
    with torch.inference_mode(), torch.amp.autocast(
        device_type=model_ensemble.device, dtype=model_ensemble.mix_dtype
    ):
        ensemble = predict_gene(
            model_ensemble, seq_1hot, targets_df, gene_strand_mask, gene_slice
        )

    _, seq_1hot_s, gene_strand_mask_s, gene_slice_s = _build_gene_inputs(
        model_single, targets_df
    )
    with torch.inference_mode(), torch.amp.autocast(
        device_type=model_single.device, dtype=model_single.mix_dtype
    ):
        forward = predict_gene_with_rc(
            model_single,
            seq_1hot_s,
            "forward",
            targets_df,
            gene_strand_mask_s,
            gene_slice_s,
        )
        reverse = predict_gene_with_rc(
            model_single,
            seq_1hot_s,
            "reverse",
            targets_df,
            gene_strand_mask_s,
            gene_slice_s,
        )
    return {"ensemble": ensemble, "forward": forward, "reverse": reverse}


def _compare_to_golden(scores):
    """Assert computed scores match the saved golden .npz within tolerance."""
    ref = np.load(GOLDEN_PATH)
    for key in ("ensemble", "forward", "reverse"):
        assert key in ref, f"golden file missing '{key}'"
        cur = scores[key]
        exp = ref[key]
        assert cur.shape == exp.shape, f"{key}: shape {cur.shape} != golden {exp.shape}"
        np.testing.assert_allclose(
            cur,
            exp,
            rtol=GOLDEN_RTOL,
            atol=GOLDEN_ATOL,
            err_msg=f"{key} scores drifted from golden reference",
        )


################################################################################
# fixtures
################################################################################
@pytest.fixture(scope="module")
def targets_df():
    _require_files()
    df = pd.read_csv(DATA_PATHS["targets_file"], index_col=0, sep="\t")
    # adds the derived 'strand' column in place (from strand_pair/identifier)
    dataset.targets_prep_strand(df)
    return df


@pytest.fixture(scope="module")
def model_ensemble():
    """Model with ensemble_rc=True (forward+reverse averaged internally)."""
    _require_files()
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for SeqNN inference")
    return create_seqnn_model(
        DATA_PATHS["params_file"],
        DATA_PATHS["model_file"],
        DATA_PATHS["targets_file"],
        mix_dtype=MIX_DTYPE,
        ensemble_rc=True,
    )


@pytest.fixture(scope="module")
def model_single():
    """Model with ensemble_rc=False (single-direction predictions)."""
    _require_files()
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for SeqNN inference")
    return create_seqnn_model(
        DATA_PATHS["params_file"],
        DATA_PATHS["model_file"],
        DATA_PATHS["targets_file"],
        mix_dtype=MIX_DTYPE,
        ensemble_rc=False,
    )


################################################################################
# tests
################################################################################
def test_create_seqnn_model(model_ensemble):
    m = model_ensemble
    # ensemble flag propagated
    assert m.ensemble_rc is True
    # mix_dtype resolved to a torch dtype
    assert (
        m.mix_dtype
        == {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[MIX_DTYPE]
    )
    # model in eval mode
    assert m.model.training is False
    # geometry accessors return sane values
    assert m.seq_length > 0
    assert m.output_stride() > 0
    assert m.output_length() > 0
    assert m.output_crop_bp() >= 0
    # strand_pair present for RC handling
    assert m.strand_pair is not None


def test_predict_gene(model_ensemble, targets_df):
    _, seq_1hot, gene_strand_mask, gene_slice = _build_gene_inputs(
        model_ensemble, targets_df
    )
    with torch.inference_mode(), torch.amp.autocast(
        device_type=model_ensemble.device, dtype=model_ensemble.mix_dtype
    ):
        preds = predict_gene(
            model_ensemble, seq_1hot, targets_df, gene_strand_mask, gene_slice
        )
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert preds.shape[0] == int(gene_strand_mask.sum())
    assert np.all(np.isfinite(preds))
    # log2 gene scores are non-negative (log2(sum(preds>=0)+1))
    assert np.all(preds >= 0.0)


def test_predict_gene_with_rc(model_single, targets_df):
    _, seq_1hot, gene_strand_mask, gene_slice = _build_gene_inputs(
        model_single, targets_df
    )
    with torch.inference_mode(), torch.amp.autocast(
        device_type=model_single.device, dtype=model_single.mix_dtype
    ):
        preds_fwd = predict_gene_with_rc(
            model_single, seq_1hot, "forward", targets_df, gene_strand_mask, gene_slice
        )
        preds_rev = predict_gene_with_rc(
            model_single, seq_1hot, "reverse", targets_df, gene_strand_mask, gene_slice
        )

    for name, preds in (("forward", preds_fwd), ("reverse", preds_rev)):
        assert isinstance(preds, np.ndarray), name
        assert preds.ndim == 1, name
        assert preds.shape[0] == int(gene_strand_mask.sum()), name
        assert np.all(np.isfinite(preds)), name
        assert np.all(preds >= 0.0), name

    # forward and reverse predictions of the same gene should be highly correlated
    if preds_fwd.shape[0] > 2:
        corr = np.corrcoef(preds_fwd, preds_rev)[0, 1]
        assert corr > 0.9, f"forward/reverse correlation too low: {corr:.3f}"

    # invalid rc_mode must raise
    with pytest.raises(ValueError):
        predict_gene_with_rc(
            model_single, seq_1hot, "sideways", targets_df, gene_strand_mask, gene_slice
        )


def test_golden_gata1(model_ensemble, model_single, targets_df):
    """Deterministic golden-output regression test for GATA1."""
    scores = _compute_gene_scores(model_ensemble, model_single, targets_df)
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(
            f"golden file not found: {GOLDEN_PATH}. "
            f"Regenerate with MISM_REGEN_GOLDEN=1."
        )
    _compare_to_golden(scores)


################################################################################
# direct execution
################################################################################
def _run_all():
    _require_files()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for SeqNN inference")

    targets = pd.read_csv(DATA_PATHS["targets_file"], index_col=0, sep="\t")
    # adds the derived 'strand' column in place (from strand_pair/identifier)
    dataset.targets_prep_strand(targets)

    print("[1/3] create_seqnn_model (ensemble_rc=True) ...")
    m_ens = create_seqnn_model(
        DATA_PATHS["params_file"],
        DATA_PATHS["model_file"],
        DATA_PATHS["targets_file"],
        mix_dtype=MIX_DTYPE,
        ensemble_rc=True,
    )
    test_create_seqnn_model(m_ens)
    print("      OK")

    print("[2/3] predict_gene (ensemble_rc=True) ...")
    test_predict_gene(m_ens, targets)
    print("      OK")

    print("[3/3] predict_gene_with_rc (ensemble_rc=False, forward/reverse) ...")
    m_sin = create_seqnn_model(
        DATA_PATHS["params_file"],
        DATA_PATHS["model_file"],
        DATA_PATHS["targets_file"],
        mix_dtype=MIX_DTYPE,
        ensemble_rc=False,
    )
    test_predict_gene_with_rc(m_sin, targets)
    print("      OK")

    print("[golden] GATA1 deterministic output ...")
    scores = _compute_gene_scores(m_ens, m_sin, targets)
    if REGEN_GOLDEN or not os.path.exists(GOLDEN_PATH):
        np.savez(GOLDEN_PATH, **scores)
        print(f"      saved golden reference -> {GOLDEN_PATH}")
    else:
        _compare_to_golden(scores)
        print("      OK (matches golden)")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_all()
