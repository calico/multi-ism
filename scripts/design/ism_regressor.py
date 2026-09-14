#! /usr/bin/env python3

import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import time
from mism.core.elasticnet_cpu import ElasticNetMulti, load_sparse_matrix_from_hdf5
from mism.core.mism_utils import (
    load_ism_scores,
    assemble_sparse_coefs,
    print_cpu_memory_peak,
)
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import os

# Start timing
start_time = time.time()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="ElasticNet regression with multiprocessing"
)
parser.add_argument(
    "--mut_XY_files",
    type=str,
    required=True,
    help="Comma-separated paths to matrices_logSUM.h5 files",
)
parser.add_argument(
    "--target_strand_file",
    type=str,
    required=True,
    help="Path to the targets_strand.txt file",
)
parser.add_argument(
    "--target_subset_file",
    type=str,
    default=None,
    help="Path to the targets_subset.txt file. If None, use all targets.",
)
parser.add_argument(
    "--coef_annot_file", type=str, required=True, help="annotation of coefs in X."
)
parser.add_argument(
    "--ism_select_sed_files",
    type=str,
    default=None,
    help="Path to ISM select SED files for adjustment",
)
parser.add_argument(
    "--ism_select_annot_files",
    type=str,
    default=None,
    help="Path to ISM select annotation files",
)
parser.add_argument(
    "--prev_coefs",
    type=str,
    default=None,
    help="Path to previous coefficients file for warm start",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=1e-4,
    help="ElasticNet alpha parameter [Default: %(default)s]",
)
parser.add_argument(
    "--njobs",
    type=int,
    default=100,
    help="Number of parallel jobs [Default: %(default)s]",
)
parser.add_argument(
    "--outdir",
    type=str,
    default="lasso_output",
    help="Output directory [Default: %(default)s]",
)

args = parser.parse_args()


##########
# inputs #
##########
mut_XY_files = [f.strip() for f in args.mut_XY_files.split(",")]
njobs = args.njobs
alpha = args.alpha

print(f"Loading mutation data from {len(mut_XY_files)} files:")
for i, file_path in enumerate(mut_XY_files):
    print(f"  {i+1}. {file_path}")

# load coefficient annotations (already filtered if bed file was used in ism_design.py)
coef_annot = pd.read_csv(args.coef_annot_file, sep="\t")
coef_annot.index = coef_annot.iloc[:, 1:5].astype(str).agg("_".join, axis=1)
print(f"Total coefficients: {len(coef_annot)}")

# filter to only bed-overlapping coefficients
overlapping_coef_indices = coef_annot[coef_annot["bed_ovlp"] == 1]["coef_idx"].values
print(f"Coefficients used in regression: {len(overlapping_coef_indices)}")

# subset targets
targets = pd.read_csv(args.target_strand_file, sep="\t", index_col=0)
if args.target_subset_file is not None:
    target_rna = pd.read_csv(args.target_subset_file, sep="\t", index_col=0)
    select_targets = np.where(targets["identifier"].isin(target_rna["identifier"]))[
        0
    ]  # targets index
else:
    # Use all targets
    select_targets = np.arange(targets.shape[0])

# Load and concatenate data from multiple files
X_list = []
Y_list = []
rc_list = [] if args.ism_select_sed_files is not None else None
ref_mut_1hot = None

for i, mut_XY_file in enumerate(mut_XY_files):
    print(f"Processing file {i+1}/{len(mut_XY_files)}: {mut_XY_file}")
    f = h5py.File(mut_XY_file, "r")

    # Handle both dense matrix and sparse matrix group cases
    if "X_mut" in f and hasattr(f["X_mut"], "keys"):
        X_curr = load_sparse_matrix_from_hdf5(f, "X_mut").astype(np.float32)
    else:
        X_curr = f["X_mut"][:].copy()
        X_curr = csr_matrix(X_curr).astype(np.float32)

    Y_curr = f["y_mut_diff"][:].astype(np.float32)

    # Only read rc_vector if needed for ISM select adjustment
    if args.ism_select_sed_files is not None:
        rc_curr = f["rc_vector"][:]
        rc_list.append(rc_curr)

    # Store ref_mut_1hot from the first file (assuming it's the same across all files)
    if ref_mut_1hot is None:
        ref_mut_1hot = f["ref_mut_1hot"][:]

    X_list.append(X_curr)
    Y_list.append(Y_curr)
    f.close()

# Concatenate all X and Y matrices vertically (more examples, same targets)
X_full = vstack(X_list).astype(np.float32)
Y = np.concatenate(Y_list, axis=0).astype(np.float32)
rc_vector = np.concatenate(rc_list, axis=0) if rc_list is not None else None

# Apply ISM select adjustment if specified
if args.ism_select_sed_files is not None:
    print("Applying ISM select variant adjustment...")

    # Parse comma-separated file lists
    ism_select_sed_file_list = [f.strip() for f in args.ism_select_sed_files.split(",")]
    ism_select_annot_file_list = [
        f.strip() for f in args.ism_select_annot_files.split(",")
    ]

    # Verify same number of files
    if len(ism_select_sed_file_list) != len(ism_select_annot_file_list):
        raise ValueError(
            f"Number of sed files ({len(ism_select_sed_file_list)}) must match number of annot files ({len(ism_select_annot_file_list)})"
        )

    print(f"Loading {len(ism_select_sed_file_list)} ISM select file pair(s)")

    # Load and concatenate all sed files
    ism_select_list = []
    for sed_file in ism_select_sed_file_list:
        f = h5py.File(sed_file, "r")
        ism_select_list.append(f["y_mut_diff"][:])
        f.close()
    ism_select = np.concatenate(ism_select_list, axis=0)

    # Load and concatenate all annot files
    ism_select_annot_list = []
    for annot_file in ism_select_annot_file_list:
        ism_select_annot_list.append(pd.read_csv(annot_file, sep="\t", index_col=0))
    ism_select_annot = pd.concat(ism_select_annot_list, axis=0)

    print(f"Total ISM select variants: {len(ism_select_annot)}")

    # find index of ism_select_annot variants inside X_full.
    ism_idx = coef_annot.loc[ism_select_annot.index, "coef_idx"].values

    # calculate total contribution from selected variants
    X_ism = X_full[:, ism_idx]
    N, T = X_ism.shape[0], ism_select.shape[1]
    select_variant_contrib = np.zeros((N, T), dtype=np.float32)
    select_variant_contrib[rc_vector == 0] = X_ism[rc_vector == 0] @ ism_select[:, :, 0]
    select_variant_contrib[rc_vector == 1] = X_ism[rc_vector == 1] @ ism_select[:, :, 1]

    # adjust Y
    Y = Y - select_variant_contrib

# subset Y
Y = Y[:, select_targets]

# subset X to only overlapping coefficients
X = X_full[:, overlapping_coef_indices]
# celer needs CSC; convert once here so forked workers share it copy-on-write
# instead of each worker re-converting CSR->CSC into a private copy (OOM).
X = X.tocsc()
print(f"Number of targets: {Y.shape[1]}")
print(f"X_full shape: {X_full.shape}")
print(f"X_filtered shape: {X.shape}")
print(f"Y shape: {Y.shape}")

#########
# lasso #
#########
# Load previous coefficients for warm start if provided
coefs_input = None
if args.prev_coefs is not None:
    print(f"Loading previous coefficients from {args.prev_coefs}")
    prev_coefs = load_ism_scores(args.prev_coefs)
    coefs_input = np.array(prev_coefs.todense(), dtype=np.float32)
    print(f"Previous coefs shape (variants, targets): {coefs_input.shape}")
    # Filter to only the overlapping coefficients we're using in this regression
    coefs_input = coefs_input[overlapping_coef_indices, :]
    print(f"Filtered coefs shape (overlapping variants only): {coefs_input.shape}")

print(f"Running L1 regression...")
coefs_subset = ElasticNetMulti(
    X,
    Y,
    njobs=args.njobs,
    alpha=args.alpha,
    coefs_input=coefs_input,
    return_sparse=True,
)

# expand to full (T, p) sparse coefs directly, without a dense (T, p) intermediate
if args.ism_select_sed_files is not None:
    ism_values = ism_select[:, select_targets, :].mean(axis=2).transpose()
else:
    ism_idx = None
    ism_values = None

sparse = assemble_sparse_coefs(
    coefs_subset,
    overlapping_coef_indices,
    X_full.shape[1],
    ism_idx=ism_idx,
    ism_values=ism_values,
)

os.makedirs(args.outdir, exist_ok=True)
torch.save(sparse, f"{args.outdir}/coefs.pt")  # save estimated coefs
np.save(f"{args.outdir}/ref_mut_1hot.npy", ref_mut_1hot)  # save ref_mut_1hot
coef_annot.to_csv(
    f"{args.outdir}/coefs_annot.csv", index=False
)  # save coefs annotation
targets.iloc[select_targets, :].to_csv(
    f"{args.outdir}/targets_subset.csv", index=False
)  # save targets subset

# End timing and print total runtime
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime:.2f} seconds")
print_cpu_memory_peak()
