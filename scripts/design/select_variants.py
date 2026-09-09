#! /usr/bin/env python3

import argparse
import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
import os
from mism.core.mism_utils import *


def select_variants(ism_m, coef_annot, n_pos):
    # this function selects top n_pos positions based on coefficient magnitudes from ISM scores.
    ism_m.data = np.abs(ism_m.data)  # take absolute value of coefs
    coef_annot["coef"] = np.array(
        ism_m.max(axis=1).todense()
    ).flatten()  # max abs coef across targets
    df_subset = coef_annot.loc[
        coef_annot["bed_ovlp"] == 1, :
    ]  # only consider bed-overlapping variants
    # sort by coef, drop duplicates by pos, take top n_pos
    df_subset = (
        df_subset.sort_values("coef", ascending=False)
        .drop_duplicates(subset="pos", keep="first")
        .iloc[:n_pos, :]
    )

    return df_subset


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Select top variants based on ISM scores"
    )
    parser.add_argument(
        "--ism_scores", type=str, required=True, help="Path to ISM scores (.pt file)"
    )
    parser.add_argument(
        "--ism_annot",
        type=str,
        required=True,
        help="Path to ISM annotation (.csv file)",
    )
    parser.add_argument(
        "--n_pos",
        type=int,
        default=100,
        help="Number of top positions to select [Default: %(default)s]",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="select_variants",
        help="Output directory [Default: %(default)s]",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist

    # inputs
    coef_annot = pd.read_csv(args.ism_annot, index_col=0)
    coef_annot.index = coef_annot.iloc[:, :4].astype(str).agg("_".join, axis=1)
    ism_m = load_ism_scores(args.ism_scores)  # csr_matrix (seqlen*3, n_targets)
    os.makedirs(args.outdir, exist_ok=True)

    # Select variants
    df_subset = select_variants(ism_m, coef_annot, args.n_pos)
    variants = coef_annot.loc[coef_annot["pos"].isin(df_subset["pos"]), :]
    variants.to_csv(f"{args.outdir}/variants.csv")

    # Save outputs
    bed = df_subset[["chr", "pos"]].copy()
    bed.sort_values(["chr", "pos"], inplace=True)
    bed["start"] = bed["pos"] - 1  # BED start (0-based)
    bed["end"] = bed["pos"]  # BED end (exclusive)
    bed[["chr", "start", "end"]].to_csv(
        f"{args.outdir}/variants.bed", sep="\t", header=False, index=False
    )
