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
import argparse
import gc
import os

import h5py
import numpy as np
import pandas as pd
import pysam
import torch
import pybedtools
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm

from baskerville_torch import dataset
from baskerville_torch import dna
from baskerville_torch import gene as bgene
from mism.core.mism_utils import *
from mism.core.elasticnet_cpu import save_sparse_matrix_to_hdf5
from mism.core.mutation_designer import MutationDesigner

"""
ism_design.py

Perform multi-mutation in silico saturation mutagenesis of a single gene sequence,
defined by a GTF file.
"""


def coef_overlap_bed(coef_annot_df, bed_df):
    """Find which coefficients overlap with bed intervals using pybedtools."""
    # coef_data to bed format (0-based)
    coef_df = coef_annot_df.copy().reset_index(drop=False)
    coef_df["pos_0based"] = coef_df["pos"] - 1
    coef_bed_data = pd.DataFrame(
        {
            "chr": coef_df["chr"],
            "start": coef_df["pos_0based"],
            "end": coef_df["pos_0based"] + 1,
            "name": coef_df["index"],
        }
    )

    # overlap
    coef_bed = pybedtools.BedTool.from_dataframe(coef_bed_data)
    bed_intervals = pybedtools.BedTool.from_dataframe(bed_df)
    overlaps = coef_bed.intersect(bed_intervals, wa=True)

    # output coef indices
    overlaps_df = overlaps.to_dataframe()
    overlapping_indices = overlaps_df["name"].astype(int).values
    return np.unique(overlapping_indices)


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(description="ISM single gene sequence.")

    #################################################################
    # model inference parameters
    #################################################################
    parser.add_argument(
        "--rc",
        default="forward",
        choices=["ensemble", "random", "forward"],
        help="RC mode: 'ensemble' for ensemble, 'random' for random RC assignment, 'forward' for forward only [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc_seed",
        type=int,
        default=42,
        help="Random seed for RC binary vector when using --rc random [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "-m",
        "--mix_dtype",
        dest="mix_dtype",
        default="bfloat16",
        help="Mixed precision dtype",
    )
    parser.add_argument(
        "-f",
        "--fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %(default)s]",
    )

    #################################################################
    # mutation design parameters
    #################################################################
    parser.add_argument(
        "--design",
        dest="mut_design",
        default="bins",
        choices=["vanilla", "bins", "random", "star-and-bar"],
        help="Mutation design matrix method [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        dest="mut_len",
        default=1000,
        type=int,
        help="Length of center sequence to mutate [Default: %(default)s]",
    )
    parser.add_argument(
        "-d",
        dest="mut_distance_min",
        default=50,
        type=int,
        help="Minimum allowed distance between mutations [Default: %(default)s]",
    )
    parser.add_argument(
        "-n",
        dest="mut_reps",
        default=50,
        type=int,
        help="Number of replicates per mutation [Default: %(default)s]",
    )
    parser.add_argument(
        "--mut2center",
        type=int,
        default=0,
        help="Offset from sequence center for mutation region. Negative values shift left, positive right. [Default: %(default)s]",
    )
    parser.add_argument(
        "--mut_rate",
        type=float,
        default=0.01,
        help="Mutation rate for random design (fraction of sequence length) [Default: %(default)s]",
    )
    parser.add_argument(
        "--num_seq",
        type=int,
        default=1000,
        help="Number of sequences to generate for random design [Default: %(default)s]",
    )
    parser.add_argument(
        "--bed_files",
        type=str,
        default=None,
        help="bed file(s) indicating regions of interest. Can be comma-separated for multiple files. If None, use all coefficients.",
    )
    parser.add_argument(
        "--bed_exclude",
        action="store_true",
        default=False,
        help="If set, exclude regions in --bed_files instead of including them [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        default="ism_out",
        help="Output directory [Default: %(default)s]",
    )

    #################################################################
    # prepare
    #################################################################
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file.")
    parser.add_argument("genes_gtf_file", help="GTF file with single gene annotation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # read targets
    if args.targets_file is None:
        raise ValueError("targets_file is required")
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    targets_strand_df = dataset.targets_prep_strand(targets_df)
    targets_strand_df.to_csv(f"{args.out_dir}/targets_strand.txt", sep="\t")

    # create seqnn model
    # For 'ensemble' mode, use ensemble RC; for 'random' and 'forward', disable ensemble RC
    ensemble_rc = args.rc == "ensemble"
    seqnn_model = create_seqnn_model(
        args.params_file,
        args.model_file,
        args.targets_file,
        args.mix_dtype,
        ensemble_rc,
    )
    model_stride = seqnn_model.output_stride()
    model_one_side_crop_bp = seqnn_model.output_crop_bp()
    target_length = seqnn_model.output_length()
    seq_len = seqnn_model.seq_length

    # determine mutation region
    mut_start, mut_end = get_mutation_region(seq_len, args.mut_len, args.mut2center)

    # gene sequences - assume single gene
    transcriptome = bgene.Transcriptome(args.genes_gtf_file)
    if len(transcriptome.genes) != 1:
        raise ValueError(f"Expect a single gene in GTF file.")

    gene_id, gene = list(transcriptome.genes.items())[0]
    set_bounds(gene, seq_len)

    print(f"Processing: {gene_id}")
    gene_name = gene.kv.get("gene_name", "NA")
    with open(f"{args.out_dir}/genes.tsv", "w") as f:
        f.write("gene_id\tgene_name\tchrom\tstart\tend\tstrand\tmut_start\tmut_end\n")
        f.write(
            f"{gene_id}\t{gene_name}\t{gene.chrom}\t{gene.seq_start}\t{gene.seq_end}\t{gene.strand}\t{mut_start}\t{mut_end}\n"
        )

    #################################################################
    # design mutation matrix
    #################################################################
    design_params = {
        "mut_reps": args.mut_reps,
        "mut_distance_min": args.mut_distance_min,
        "mut_rate": args.mut_rate,
        "num_seq": args.num_seq,
    }

    # design initial batch
    designer = MutationDesigner()
    designer.design_initial_batch(args.mut_len, args.mut_design, design_params)
    X_mut3 = designer.get_mutation_matrix()  # list of coo_matrix: (mut_len, 3)
    X_mut = designer.get_regression_matrix()  # csr_matrix: (num_mutseqs, 3 * mut_len)

    # Generate RC vector for all modes (0=forward, 1=reverse, 2=ensemble)
    num_mutseqs = designer.num_mutseqs

    if args.rc == "random":
        # Set random seed for reproducibility
        np.random.seed(args.rc_seed)

        # rc vector: 0 = forward, 1 = reverse
        half = num_mutseqs // 2
        rc_vector = np.concatenate(
            [np.zeros(half, dtype=int), np.ones(num_mutseqs - half, dtype=int)]
        )
        np.random.shuffle(rc_vector)
        print(f"forward sequences: {np.sum(rc_vector == 0)}")
        print(f"reverse sequences: {np.sum(rc_vector == 1)}")
    elif args.rc == "forward":
        rc_vector = np.zeros(num_mutseqs, dtype=int)  # All forward
    elif args.rc == "ensemble":
        rc_vector = np.full(num_mutseqs, 2, dtype=int)  # All ensemble

    #################################################################
    # prep gene sequence
    #################################################################
    seq_out_start = gene.seq_start + model_one_side_crop_bp
    seq_out_len = model_stride * target_length
    gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride)
    if gene.strand == "+":
        gene_strand_mask = (targets_df.strand != "-").to_numpy()
    else:
        gene_strand_mask = (targets_df.strand != "+").to_numpy()

    fasta_open = pysam.Fastafile(args.fasta)
    ref_dna = fasta_open.fetch(gene.chrom, gene.seq_start, gene.seq_end)
    fasta_open.close()

    if len(ref_dna) < seq_len:
        ref_dna += "N" * (seq_len - len(ref_dna))
    ref_1hot = dna.dna_1hot(ref_dna)

    # mutation region 1-hot
    ref_mut_1hot = ref_1hot[mut_start:mut_end]

    #################################################################
    # create X_mut annotation dataframe
    #################################################################
    x_mut_annotation = create_x_mut_annotation(
        gene, mut_start, args.mut_len, ref_mut_1hot
    )

    # Apply bed file filtering if provided
    if args.bed_files is not None:
        bed_files = [f.strip() for f in args.bed_files.split(",")]
        mode_str = "exclude" if args.bed_exclude else "include"
        print(f"Processing {len(bed_files)} bed file(s) in {mode_str} mode")

        # Union overlapping indices from all bed files
        all_overlapping_indices = set()
        for bed_file in bed_files:
            bed_df = pd.read_csv(
                bed_file,
                sep="\t",
                header=None,
                names=["chr", "start", "end"],
                usecols=[0, 1, 2],
            )
            overlapping_indices = coef_overlap_bed(x_mut_annotation, bed_df)
            all_overlapping_indices.update(overlapping_indices)
            print(f"  {bed_file}: {len(overlapping_indices)} overlapping coefficients")

        if args.bed_exclude:  # mark overlapping positions as 0 (to be excluded)
            x_mut_annotation["bed_ovlp"] = (
                ~x_mut_annotation["coef_idx"].isin(all_overlapping_indices)
            ).astype(int)
        else:  # mark overlapping positions as 1 (to be included)
            x_mut_annotation["bed_ovlp"] = (
                x_mut_annotation["coef_idx"].isin(all_overlapping_indices).astype(int)
            )

        print(f"Total coefficients: {len(x_mut_annotation)}")
        print(f"Coefficients to estimate: {x_mut_annotation['bed_ovlp'].sum()}")
    else:
        x_mut_annotation["bed_ovlp"] = 1  # All coefficients are included
        print(f"Total coefficients: {len(x_mut_annotation)}")

    # mask non-ACGT positions (N) in mutation region by zeroing bed_ovlp
    ref_row_sums = ref_mut_1hot.sum(axis=1)
    invalid_positions = np.where(ref_row_sums != 1)[0]
    if invalid_positions.size > 0:
        invalid_coef_indices = (invalid_positions[:, None] * 3 + np.arange(3)).ravel()
        x_mut_annotation.loc[
            x_mut_annotation["coef_idx"].isin(invalid_coef_indices), "bed_ovlp"
        ] = 0
        print(
            f"Masked {invalid_positions.size} non-ACGT positions in mutation region "
            f"({len(invalid_coef_indices)} coefficients)."
        )

    x_mut_annotation.to_csv(
        f"{args.out_dir}/x_mut_annotation.tsv", sep="\t", index=False
    )

    #################################################################
    # zero out non-overlapping positions if any bed_ovlp == 0
    #################################################################
    # Get indices of coefficients to zero out (non-overlapping)
    zero_coef_indices = x_mut_annotation[x_mut_annotation["bed_ovlp"] == 0][
        "coef_idx"
    ].values

    # Zero X_mut
    if len(zero_coef_indices) > 0:
        X_mut_lil = X_mut.tolil()
        X_mut_lil[:, zero_coef_indices] = 0
        X_mut = X_mut_lil.tocsr()

        # Filter X_mut3
        zero_positions = set(zero_coef_indices // 3)
        for i, mut_matrix in enumerate(X_mut3):
            if mut_matrix.nnz > 0 and zero_positions:
                keep_mask = ~np.isin(mut_matrix.row, list(zero_positions))
                if not keep_mask.all():
                    X_mut3[i] = coo_matrix(
                        (
                            mut_matrix.data[keep_mask],
                            (mut_matrix.row[keep_mask], mut_matrix.col[keep_mask]),
                        ),
                        shape=mut_matrix.shape,
                    )

        print(f"Zeroed out {len(zero_coef_indices)} non-overlapping coefficients")

    #################################################################
    # forward pass
    #################################################################
    mut_scores = {"logSUM": []}

    ref_1hot = np.expand_dims(ref_1hot.T, axis=0)
    ref_1hot = torch.tensor(ref_1hot, device=seqnn_model.device, dtype=torch.float32)

    with torch.inference_mode(), torch.amp.autocast(
        device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
    ):
        for si in tqdm(range(designer.num_mutseqs)):
            # make alt sequence
            alt_1hot = make_alt(ref_1hot, mut_start, X_mut3[si])

            # predict based on RC mode
            if args.rc == "random":
                rc_mode = "reverse" if rc_vector[si] == 1 else "forward"
                alt_score = predict_gene_with_rc(
                    seqnn_model,
                    alt_1hot,
                    rc_mode,
                    targets_df,
                    gene_strand_mask,
                    gene_slice,
                )
            else:  # ensemble and forward modes
                # Use predict_gene since model is already configured with correct ensemble_rc setting
                alt_score = predict_gene(
                    seqnn_model, alt_1hot, targets_df, gene_strand_mask, gene_slice
                )

            mut_scores["logSUM"].append(alt_score)
            torch.cuda.empty_cache()

            if (si + 1) % 100 == 0:
                gc.collect()

    gc.collect()

    with torch.inference_mode(), torch.amp.autocast(
        device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
    ):
        if args.rc == "random":
            # For random mode, calculate both forward and reverse reference scores
            ref_score_forward = predict_gene_with_rc(
                seqnn_model,
                ref_1hot,
                "forward",
                targets_df,
                gene_strand_mask,
                gene_slice,
            )
            ref_score_reverse = predict_gene_with_rc(
                seqnn_model,
                ref_1hot,
                "reverse",
                targets_df,
                gene_strand_mask,
                gene_slice,
            )
        else:  # ensemble and forward modes
            # Use predict_gene since model is already configured with correct ensemble_rc setting
            ref_score = predict_gene(
                seqnn_model, ref_1hot, targets_df, gene_strand_mask, gene_slice
            )

    y_mut = np.array(mut_scores["logSUM"])

    # Calculate y_mut_diff based on RC mode
    if args.rc == "random":
        # For random mode, subtract appropriate reference based on RC vector
        y_mut_diff = np.zeros_like(y_mut)
        for si in range(designer.num_mutseqs):
            if rc_vector[si] == 0:  # forward
                y_mut_diff[si] = y_mut[si] - ref_score_forward
            else:  # reverse
                y_mut_diff[si] = y_mut[si] - ref_score_reverse
    else:
        # For ensemble and forward modes, use single reference
        y_mut_diff = y_mut - ref_score[np.newaxis, :]  # Broadcasting for each target
    regression_file = f"{args.out_dir}/matrices_logSUM.h5"
    print(f"Saving regression matrices to {regression_file}")

    with h5py.File(regression_file, "w") as reg_h5:
        # Save sparse matrix using save_sparse_matrix_to_hdf5
        save_sparse_matrix_to_hdf5(reg_h5, "X_mut", X_mut)
        reg_h5.create_dataset("y_mut_diff", data=y_mut_diff, compression="gzip")
        reg_h5.create_dataset("ref_mut_1hot", data=ref_mut_1hot, compression="gzip")
        reg_h5.create_dataset("rc_vector", data=rc_vector, compression="gzip")
        if args.rc == "random":
            reg_h5.create_dataset(
                "ref_score_forward", data=ref_score_forward, compression="gzip"
            )
            reg_h5.create_dataset(
                "ref_score_reverse", data=ref_score_reverse, compression="gzip"
            )
        else:
            reg_h5.create_dataset(
                "ref_score_forward", data=ref_score, compression="gzip"
            )
            reg_h5.create_dataset(
                "ref_score_reverse", data=ref_score, compression="gzip"
            )

    print_cpu_memory_peak()
    print_gpu_memory_peak()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
