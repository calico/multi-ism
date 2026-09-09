#!/usr/bin/env python3
"""
Generate sequence logos comparing standard ISM vs alternative ISM scores 
for K562 RNA targets around GATA1 binding sites.
"""

import h5py
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import argparse
import logomaker
import matplotlib.patches as patches
import seaborn as sns


def offset_crop(arr, offset, target_len):
    seq_length = arr.shape[0]
    seq_center = seq_length // 2
    start = seq_center + offset - (target_len // 2)
    end = start + target_len

    if start < 0 or end > seq_length:
        raise ValueError(f"Crop window [{start}:{end}] exceeds bounds [0:{seq_length}]")

    return arr[start:end]


def get_mutation_region(seq_length, mut_len, mut2center=0):
    """Calculate the start and end positions of mutation region.

    Args:
        seq_length (int): Total length of the sequence
        mut_len (int): Length of mutation window
        mut2center (int): Offset from sequence center.
            Negative values shift mutation region left, positive values shift right.
    Returns:
        tuple: (mut_start, mut_end) positions
    """
    seq_center = seq_length // 2
    mut_start = seq_center - (mut_len // 2) + mut2center
    mut_end = mut_start + mut_len

    # Check if mutation region stays within sequence bounds
    if mut_start < 0 or mut_end > seq_length:
        raise ValueError(
            f"Invalid mutation region: [{mut_start}, {mut_end}] exceeds sequence bounds [0, {seq_length}]. "
        )

    return mut_start, mut_end


def plot_logo(m, ymin, ymax, ax, title, pwm_pos=None, pwm_len=None):
    nn_logo = logomaker.Logo(m, ax=ax, baseline_width=0)
    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=["left"], visible=True, bounds=[ymin, ymax])
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)

    # label the pwm
    if pwm_pos is not None:
        rect = patches.Rectangle(
            (pwm_pos - 1.5, ymin + 0.01),
            pwm_len,
            ymax - (ymin + 0.01),
            linewidth=1,
            edgecolor="b",
            facecolor="none",
            linestyle="dashed",
        )
        ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation m-ism with vanilla-ism logo"
    )
    parser.add_argument(
        "--ref_path",
        default="/path/to/design/ism_out/ref_mut_1hot.npy",
        help="Path to reference mut 1hot file, (100k, 4)",
    )
    parser.add_argument(
        "--ism_scores",
        required=True,
        help="Path to ISM scores .pt file, (100k, 4, n_targets)",
    )
    parser.add_argument(
        "--gene_id", default="ENSG00000102145.15", help="Gene ID [Default: %(default)s]"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=-200,
        help="Offset from gene center for logo plot [Default: %(default)s]",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Window size for logo plot [Default: %(default)s]",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Whether input ISM scores are already normalized [Default: %(default)s]",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Minimum y-axis value for logo plot [Default: %(default)s]",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Maximum y-axis value for logo plot [Default: %(default)s]",
    )

    args = parser.parse_args()

    offset = args.offset
    window = args.window
    ymin = args.ymin
    ymax = args.ymax
    gene_id = args.gene_id

    # Determine offset_type based on offset value
    #'Vanilla ISM is done on 3 separate windows, 10k around 0, +40k, -40k from gene center
    if -5000 <= offset <= 5000:
        offset_type = "offset_0"
    elif -45000 <= offset <= -35000:
        offset_type = "offset_-40k"
    elif 35000 <= offset <= 45000:
        offset_type = "offset_+40k"
    else:
        raise ValueError(
            f"Invalid offset {offset}. Must be in [-5k,5k], [-45k,-35k], or [35k,45k]"
        )

    print(f"Auto-detected offset_type: {offset_type} (offset={offset})")

    # Set paths based on auto-detected offset_type
    gata_ism_path = f"/path/to/vanilla_ism/ism_10k_{offset_type}/{gene_id}/ism_out"
    outdir = f"eval_{offset_type}"

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    ###############
    # vanilla ism #
    ###############
    targets = pd.read_csv(
        "%s/targets_strand.txt" % gata_ism_path, sep="\t", index_col=0
    )
    f = h5py.File("%s/scores.h5" % gata_ism_path, "r")
    score_vanilla = f["logSUM"][0, :, :, :]
    f.close()

    ref_mut_1hot = np.load(args.ref_path)

    ###################
    # load ism scores #
    ###################
    ism_tensor = torch.load(args.ism_scores)

    if ism_tensor.is_sparse:
        dense = ism_tensor.to_dense()
        scores_raw = dense.cpu().numpy()
    else:
        scores_raw = ism_tensor.cpu().numpy()

    # mean-centered scores
    scores_raw = scores_raw.astype(np.float32)
    scores = scores_raw - scores_raw.mean(axis=1, keepdims=True)

    ###############
    # compare 10k #
    ###############
    offset_map = {"offset_0": 0, "offset_+40k": 40000, "offset_-40k": -40000}
    offset_center = offset_map[offset_type]

    compare_window = min(scores.shape[0], score_vanilla.shape[0])
    m0 = offset_crop(
        score_vanilla, offset=0, target_len=compare_window
    )  # vanilla-ism, 10k, no crop
    m1 = offset_crop(
        scores, offset=offset_center, target_len=compare_window
    )  # m-ism, 100k, crop
    m1_raw = offset_crop(
        scores_raw, offset=offset_center, target_len=compare_window
    )  # m-ism raw, 100k, crop
    m_ref = offset_crop(
        ref_mut_1hot, offset=offset_center, target_len=compare_window
    )  # ref_1hot, 100k, crop

    ###############
    # plot logo #
    ###############
    k562_idx = np.where(
        targets["description"].str.contains("K562")
        & targets["assay"].isin(["rna", "rna3"])
    )[0]

    start, end = get_mutation_region(m0.shape[0], window, offset - offset_center)
    dist2center_start = offset - window // 2
    dist2center_end = dist2center_start + window

    # vanilla ism
    toplot0 = pd.DataFrame(
        m0[:, :, k562_idx].mean(axis=2) * m_ref, columns=["A", "C", "G", "T"]
    )
    toplot0 = toplot0.iloc[start:end, :]
    toplot0.index = np.arange(dist2center_start, dist2center_end)

    # m-ism
    toplot1 = pd.DataFrame(
        m1[:, :, k562_idx].mean(axis=2) * m_ref, columns=["A", "C", "G", "T"]
    )
    toplot1 = toplot1.iloc[start:end, :]
    toplot1.index = np.arange(dist2center_start, dist2center_end)

    # m-ism raw scores
    toplot1_raw = pd.DataFrame(
        m1_raw[:, :, k562_idx].mean(axis=2), columns=["A", "C", "G", "T"]
    )
    toplot1_raw = toplot1_raw.iloc[start:end, :]
    toplot1_raw.index = np.arange(dist2center_start, dist2center_end)

    ###############
    # plot params #
    ###############

    toplot0_min = float(ymin if ymin is not None else toplot0.values.min())
    toplot1_min = float(ymin if ymin is not None else toplot1.values.min())
    toplot0_max = float(ymax if ymax is not None else toplot0.values.max())
    toplot1_max = float(ymax if ymax is not None else toplot1.values.max())

    fig_w = int(window / 10)
    fig_h = 6

    f, axs = plt.subplots(figsize=(fig_w, fig_h), nrows=3)

    plot_logo(toplot0, toplot0_min, toplot0_max, axs[0], "Standard-ISM (mean-centered)")
    plot_logo(toplot1, toplot1_min, toplot1_max, axs[1], "M-ISM (mean-centered)")
    sns.heatmap(toplot1_raw.T, center=0, cmap="coolwarm", cbar=False, ax=axs[2])

    if args.normalized:
        axs[2].set_title("M-ISM Scores (mean-normalized)")
    else:
        axs[2].set_title("M-ISM Raw Scores (not normalized)")

    f.tight_layout()
    f.savefig("%s/seqlogo_%d_%d.pdf" % (outdir, dist2center_start, dist2center_end))


if __name__ == "__main__":
    main()
