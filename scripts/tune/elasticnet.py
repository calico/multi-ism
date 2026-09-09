#! /usr/bin/env python3

import argparse
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
import time
from mism.core.elasticnet_cpu import ElasticNetMulti
from mism.core.mism_utils import coef2score

# Start timing
start_time = time.time()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="ElasticNet regression with multiprocessing"
)
parser.add_argument(
    "--input_folder",
    default="/path/to/input_folder",
    type=str,
    help="Input folder containing the data files",
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

njobs = args.njobs
alpha = args.alpha

print(f"Loading data from: {input_folder}")

targets = pd.read_csv(target_strand_file, sep="\t", index_col=0)
ref_mut_1hot = np.load(ref_mut_1hot_file)
f = h5py.File(mut_XY_file, "r")
X = f["X_mut"][:].copy()
Y = f["y_mut_diff"][:].copy()
X = csr_matrix(X.astype(np.float64))
Y = np.array(Y, dtype=np.float64, copy=True)
f.close()

#########
# lasso #
#########
coefs = ElasticNetMulti(X, Y, njobs=args.njobs, alpha=args.alpha)
scores_m = coef2score(ref_mut_1hot, coefs, ref_mut_1hot.shape[0], Y.shape[1])

########
# save #
########
dense = torch.tensor(scores_m, dtype=torch.float16)
sparse = dense.to_sparse()
torch.save(sparse, "ism_scores_raw.pt")

# End timing and print total runtime
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime:.2f} seconds")
