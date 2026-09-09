#!/usr/bin/env python3

import argparse
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from tqdm import tqdm


def coef2score(ref_mut_1hot, coefs, mut_len, target_num, normalize=False):
    # Initialize scores
    # coefs shape: (target_num, mut_len)
    seq_scores = np.zeros((mut_len, 4, target_num), dtype="float32")

    # for each mutated position
    ci = 0
    for mi in range(mut_len):
        # for each nucleotide
        for ni in range(4):
            if ref_mut_1hot[mi, ni]:
                seq_scores[mi, ni, :] = 0
            else:
                seq_scores[mi, ni, :] = coefs[:, ci]
                ci += 1

    # normalize positions
    if normalize:
        seq_scores -= seq_scores.mean(axis=1, keepdims=True)

    return seq_scores


def feature_mean(X, Y):
    X_csc = X.tocsc()
    n_samples, n_features = X_csc.shape
    n_targets = Y.shape[1]

    feature_sums = (
        X_csc.T @ Y
    )  # Shape: (n_features, n_targets), Sum of Y when each feature is active
    Y_total_sum = Y.sum(axis=0)  # Total sum across all samples

    feature_counts = np.array(X_csc.sum(axis=0)).flatten()
    feature_effects = np.zeros((n_features, n_targets))

    for j in tqdm(range(n_features), desc="Computing feature effects"):
        n_active = feature_counts[j]
        n_inactive = n_samples - n_active

        mean_active = feature_sums[j] / n_active
        mean_inactive = (Y_total_sum - feature_sums[j]) / n_inactive
        feature_effects[j] = mean_active - mean_inactive

    print(f"Computed {n_features} feature effects")

    return feature_effects


# Parse command line arguments
parser = argparse.ArgumentParser(description="coefficient by feature mean.")
parser.add_argument(
    "--input_folder",
    default="/path/to/input_folder",
    type=str,
    help="Input folder containing the data files",
)

args = parser.parse_args()

##########
# inputs #
##########
input_folder = args.input_folder
target_strand_file = "%s/ism_out/targets_strand.txt" % input_folder
ref_mut_1hot_file = "%s/ism_out/ref_mut_1hot.npy" % input_folder

# Find the matrices file using glob pattern
matrices_pattern = f"{input_folder}/ism_out/matrices_*_logSUM.h5"
matrices_files = glob.glob(matrices_pattern)
if (len(matrices_files) == 0) or (len(matrices_files) > 1):
    raise FileNotFoundError(f"Expected 1 matrices file, found {len(matrices_files)}")
mut_XY_file = matrices_files[0]

targets = pd.read_csv(target_strand_file, sep="\t", index_col=0)
ref_mut_1hot = np.load(ref_mut_1hot_file)
f = h5py.File(mut_XY_file, "r")
X = f["X_mut"][:]
Y = f["y_mut_diff"][:]
X = csr_matrix(X.astype(np.float32))
Y = np.array(Y, dtype=np.float32)
f.close()

coefs = feature_mean(X, Y)
coefs = coefs.transpose()

scores_m = coef2score(
    ref_mut_1hot, coefs, ref_mut_1hot.shape[0], Y.shape[1], normalize=True
)

# Convert to float16 tensor
scores_tensor = torch.tensor(scores_m, dtype=torch.float16)
torch.save(scores_tensor, "ism_scores.pt")
