#!/usr/bin/env python3
'''
"""
Compare standard ISM vs alternative ISM scores for K562 RNA targets.
Evaluates correlation between the two methods across a 10kb window.
"""

'''
import h5py
import numpy as np
import pandas as pd
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
import baskerville_torch.gene as bgene
import matplotlib.patches as patches
from mism.core.mism_utils import *


def offset_crop(arr, offset, target_len):
    seq_length = arr.shape[0]
    seq_center = seq_length // 2
    start = seq_center + offset - (target_len // 2)
    end = start + target_len

    if start < 0 or end > seq_length:
        raise ValueError(f"Crop window [{start}:{end}] exceeds bounds [0:{seq_length}]")

    return arr[start:end]


def estimate_r(x, y):

    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])

    # Center
    x_mean = x - x.mean(axis=0)
    y_mean = y - y.mean(axis=0)

    # Compute numerator and denominator
    numerator = np.sum(x_mean * y_mean, axis=0)
    x_norm = np.sqrt(np.sum(x_mean**2, axis=0))
    y_norm = np.sqrt(np.sum(y_mean**2, axis=0))
    denominator = x_norm * y_norm

    # Avoid divide-by-zero
    pearson_r = np.zeros_like(numerator)
    mask = denominator != 0
    pearson_r[mask] = numerator[mask] / denominator[mask]
    pearson_r[~mask] = np.nan

    return pearson_r


def compute_nonzero_by_distance(ism_m, bin_size=2000):
    """
    Compute non-zero fraction by distance bins for ISM sparse matrix.

    Args:
        ism_m: Sparse matrix (300k, n_targets) where 300k = seq_len * 3 nucleotides
        bin_size: Size of each bin in base pairs (default: 2000)

    Returns:
        pd.DataFrame with 'distance' and 'nonzero' columns
    """
    n_nucleotides = 3
    seq_len = ism_m.shape[0] // n_nucleotides
    n_targets = ism_m.shape[1]

    sparse_matrix = ism_m.tocoo()
    row_coords = sparse_matrix.row  # feature indices (3*seqlen)
    seq_positions = row_coords // n_nucleotides

    n_bins = seq_len // bin_size
    bin_indices = seq_positions // bin_size
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    entries_per_bin = (bin_size * n_nucleotides) * n_targets
    chunks_nonzero = bin_counts / entries_per_bin
    center_bin = n_bins // 2
    distance = (np.arange(n_bins) - center_bin) * (bin_size / 1000)  # Convert to kb

    return pd.DataFrame({"distance": distance, "nonzero": chunks_nonzero})


def variant2ref(variant_m):
    """
    Converts variant effect scores (3 alternatives per position) to
    reference nucleotide scores by taking the negative average:
    ref = -sum(3 alts) / 4
    """
    # variant_m shape: (seqlen * 3, n_targets)
    seq_len = variant_m.shape[0] // 3
    n_targets = variant_m.shape[1]
    ref_score = -variant_m.reshape(seq_len, 3, n_targets).sum(axis=1) / 4
    return ref_score


def main():
    parser = argparse.ArgumentParser(description="Evaluation m-ism with vanilla-ism")
    parser.add_argument(
        "--ism_scores", required=True, help="Path to ISM scores .pt file"
    )
    parser.add_argument("--ism_annot", required=True, help="Path to ISM annot file")
    parser.add_argument("--ism_target", required=True, help="Path to ISM target file")
    parser.add_argument("--outdir", default="eval", help="Path to output directory")
    parser.add_argument(
        "--gtf",
        default="/path/to/gencode_basic_protein.gtf",
        help="GTF file [Default: %(default)s]",
    )
    parser.add_argument(
        "--celltype", default="K562", help="Cell type [Default: %(default)s]"
    )
    args = parser.parse_args()

    transcriptome = bgene.Transcriptome(args.gtf)

    os.makedirs(args.outdir, exist_ok=True)

    ###################
    # load ism scores #
    ###################
    targets = pd.read_csv(args.ism_target)
    coef_annot = pd.read_csv(args.ism_annot, index_col=0)
    coef_annot.index = coef_annot.iloc[:, :4].astype(str).agg("_".join, axis=1)

    ism_m = load_ism_scores(args.ism_scores)  # csr_matrix (seqlen*3, n_targets)

    chr = coef_annot["chr"].iloc[0]
    win_start = coef_annot["pos"].min()
    win_end = coef_annot["pos"].max()
    transcriptome_filter = filter_transcriptome_by_range(
        transcriptome, chr, win_start, win_end
    )

    ############################
    # non-zero by distance-bin #
    ############################
    print("plot m-ism non-zeros by 2kb bins, raw scores (not mean-centered)")

    toplot = compute_nonzero_by_distance(ism_m, bin_size=2000)

    f, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=toplot, x="distance", y="nonzero", ax=ax)
    ax.set_xlabel("distance-to-center (kb)")
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::5])
    ax.set_xticklabels(toplot["distance"][::5])
    f.tight_layout()
    f.savefig("%s/nonzero_by_distance.pdf" % args.outdir)

    ###############
    # plot tracks #
    ###############

    # Find K562 RNA targets
    select = targets["description"].str.contains(args.celltype).values

    ism_mean = np.array(ism_m[:, select].mean(axis=1))
    ism_mean = variant2ref(ism_mean)[:, 0]
    x_ism = np.arange(coef_annot["pos"].min(), coef_annot["pos"].max() + 1)

    print("plot m-ism tracks corresponding to %s" % args.celltype)

    f, axs = plt.subplots(figsize=(12, 3), nrows=2, sharex=True)

    # M-ISM track
    sns.lineplot(x=x_ism, y=ism_mean, ax=axs[0])
    axs[0].set_title(f"LASSO-ISM")
    axs[0].set_ylabel("ISM Score")
    axs[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plot_gene_track(transcriptome_filter, chr, win_start, win_end, ax=axs[1])

    axs[1].set_xlabel("bp")

    f.tight_layout()
    f.savefig("%s/ism_tracks_comparison.pdf" % args.outdir)


if __name__ == "__main__":
    main()
