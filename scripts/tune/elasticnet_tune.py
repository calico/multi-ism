#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
from celer import ElasticNetCV
from sklearn.model_selection import PredefinedSplit

# Start timing
start_time = time.time()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Tune ElasticNet parameters")
parser.add_argument("--target_id", type=str, required=True, help="Target identifier")
parser.add_argument(
    "--l1_ratio", type=float, required=True, help="L1 ratio for ElasticNet"
)
parser.add_argument(
    "--fold_file", type=str, required=True, help="Path to fold IDs file"
)
parser.add_argument(
    "--out_file", type=str, default="mse.csv", help="Output CSV file path"
)
parser.add_argument(
    "--n_jobs", type=int, default=16, help="Number of jobs to run in parallel"
)

args = parser.parse_args()

target_id = args.target_id
l1_ratio = args.l1_ratio
fold_file = args.fold_file
out_file = args.out_file
n_jobs = args.n_jobs

alphas = np.logspace(-7, 2, num=10)

##########
# inputs #
##########
input_folder = "/path/to/input_folder"
target_strand_file = "%s/ism_out/targets_strand.txt" % input_folder
mut_XY_file = "%s/ism_out/matrices_ENSG00000102145.15_logSUM.h5" % input_folder

targets = pd.read_csv(target_strand_file, sep="\t", index_col=0)
f = h5py.File(mut_XY_file, "r")
X = f["X_mut"][:].copy()
Y = f["y_mut_diff"][:].copy()
X = csr_matrix(X.astype(np.float64))
Y = np.array(Y, dtype=np.float64, copy=True)
f.close()

idx = np.where(targets["identifier"] == target_id)[0][0]
y = Y[:, idx]
fold_ids = np.load(fold_file)

print("Fitting ElasticNet...")
print(f"l1_ratio: {l1_ratio}")
print(f"alphas: {alphas}")
print(f"n_jobs: {n_jobs}")

ps = PredefinedSplit(fold_ids)
clr = ElasticNetCV(
    l1_ratio=l1_ratio,
    alphas=alphas,
    cv=ps,
    fit_intercept=False,
    max_iter=500,
    n_jobs=n_jobs,
)
clr.fit(X, y)
mse = pd.DataFrame(clr.mse_path_, index=clr.alphas_.astype(str))
mse.to_csv(out_file)

print(f"Time taken: {time.time() - start_time:.2f} seconds")
