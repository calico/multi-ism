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
from tqdm import tqdm

from baskerville_torch import dataset
from baskerville_torch import dna
from baskerville_torch import gene as bgene
from mism.core.mism_utils import *
from mism.core.mutation_designer import MutationDesigner

"""
ism_vanilla_select.py

Evaluate variants from a CSV file on a single gene sequence,
defined by a GTF file.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate variants on single gene sequence."
    )

    #################################################################
    # model inference parameters
    #################################################################
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
    # variant file parameters
    #################################################################
    parser.add_argument(
        "-o",
        dest="out_dir",
        default="ism_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--variants_csv",
        required=True,
        help="CSV file with columns: chr, pos, ref, alt",
    )
    parser.add_argument(
        "--k_background",
        type=int,
        default=4,
        help="Number of background sequences to generate [Default: %(default)s]",
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
    num_targets_strand = targets_strand_df.shape[0]
    targets_strand_df.to_csv(f"{args.out_dir}/targets_strand.txt", sep="\t")

    # read variants (with index column if present)
    variants_df = pd.read_csv(args.variants_csv, index_col=0)

    # create seqnn model (always compute forward and reverse separately)
    seqnn_model = create_seqnn_model(
        args.params_file,
        args.model_file,
        args.targets_file,
        args.mix_dtype,
        ensemble_rc=False,
    )
    model_stride = seqnn_model.output_stride()
    model_one_side_crop_bp = seqnn_model.output_crop_bp()
    target_length = seqnn_model.output_length()
    seq_len = seqnn_model.seq_length

    # gene sequences - assume single gene
    transcriptome = bgene.Transcriptome(args.genes_gtf_file)
    if len(transcriptome.genes) != 1:
        raise ValueError(f"Expect a single gene in GTF file.")
    gene_id, gene = list(transcriptome.genes.items())[0]
    set_bounds(gene, seq_len)

    print(f"Processing: {gene_id}")
    gene_name = gene.kv.get("gene_name", "NA")

    with open(f"{args.out_dir}/genes.tsv", "w") as f:
        f.write("gene_id\tgene_name\tchrom\tstart\tend\tstrand\n")
        f.write(
            f"{gene_id}\t{gene_name}\t{gene.chrom}\t{gene.seq_start}\t{gene.seq_end}\t{gene.strand}\n"
        )

    #################################################################
    # prep gene sequence
    #################################################################
    # Account for output cropping - model output coordinates
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

    #################################################################
    # filter variants to gene region
    #################################################################
    # Filter variants that fall within the gene sequence bounds
    variants_in_gene = variants_df[
        (variants_df["chr"] == gene.chrom)
        & (variants_df["pos"] >= gene.seq_start)
        & (variants_df["pos"] < gene.seq_end)
    ].copy()

    if len(variants_in_gene) == 0:
        raise ValueError(
            f"No variants found within gene region {gene.chrom}:{gene.seq_start}-{gene.seq_end}"
        )

    print(f"Found {len(variants_in_gene)} variants within gene region")

    # Convert genomic positions to sequence-relative positions
    variants_in_gene["seq_pos"] = variants_in_gene["pos"] - gene.seq_start - 1

    #################################################################
    # design 100 background sequences
    #################################################################
    print(f"\nDesigning 100 background sequences...")

    # Design background mutations for full sequence using bins strategy
    bg_designer = MutationDesigner()
    bg_designer.design_initial_batch(
        seq_len, "bins", {"mut_reps": 5, "mut_distance_min": 50}
    )
    X_mut_all = (
        bg_designer.get_regression_matrix()
    )  # csr_matrix: (num_mutseqs, 3 * seq_len)

    print(f"Total designed sequences: {X_mut_all.shape[0]}")

    # Use all available sequences (up to 100)
    selected_indices = np.random.choice(X_mut_all.shape[0], 100, replace=False)
    X_mut_selected = X_mut_all[selected_indices]
    print(f"Selected 100 background sequences")

    # Convert all background sequences to DNA strings (no exclusion zones)
    ref_1hot_full = ref_1hot.copy()  # (seq_len, 4)
    background_dna_list = []
    background_nt_indices_list = []  # Store nucleotide indices (0=A, 1=C, 2=G, 3=T)

    for idx in range(100):
        # Get mutation vector for this background
        bg_mut_vector = X_mut_selected[idx].toarray().flatten()  # (3 * seq_len,)

        # Convert to 1-hot and DNA string
        bg_1hot = mut_vector_to_1hot(bg_mut_vector, ref_1hot_full)
        bg_dna = dna.hot1_dna(bg_1hot)
        background_dna_list.append(bg_dna)
        background_nt_indices_list.append(np.argmax(bg_1hot, axis=1).astype(np.uint8))

    # Stack to create [num_backgrounds, seq_len] array
    background_nt_indices = np.stack(background_nt_indices_list, axis=0)
    print(
        f"Background nucleotide indices shape: {background_nt_indices.shape} ({background_nt_indices.nbytes / 1024**2:.1f} MB)"
    )

    #################################################################
    # forward and reverse passes for all backgrounds
    #################################################################
    print(f"\nGetting predictions for all 100 background sequences...")
    background_scores_fwd = []
    background_scores_rev = []

    for bg_idx in range(100):
        bg_dna = background_dna_list[bg_idx]

        # Convert to 1hot and torch tensor
        bg_1hot = dna.dna_1hot(bg_dna)
        bg_1hot_tensor = np.expand_dims(bg_1hot.T, axis=0)
        bg_1hot_tensor = torch.tensor(
            bg_1hot_tensor, device=seqnn_model.device, dtype=torch.float32
        )

        # Get reference predictions for this background (forward and reverse)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
        ):
            bg_ref_score_fwd = predict_gene_with_rc(
                seqnn_model,
                bg_1hot_tensor,
                "forward",
                targets_df,
                gene_strand_mask,
                gene_slice,
            )
            bg_ref_score_rev = predict_gene_with_rc(
                seqnn_model,
                bg_1hot_tensor,
                "reverse",
                targets_df,
                gene_strand_mask,
                gene_slice,
            )

        background_scores_fwd.append(bg_ref_score_fwd)
        background_scores_rev.append(bg_ref_score_rev)

        if (bg_idx + 1) % 10 == 0:
            print(f"  Processed {bg_idx + 1}/100 backgrounds")
            torch.cuda.empty_cache()
            gc.collect()

    background_scores_fwd = np.array(background_scores_fwd)  # shape: [100, n_targets]
    background_scores_rev = np.array(background_scores_rev)  # shape: [100, n_targets]
    print(f"Background scores shape: {background_scores_fwd.shape}")

    #################################################################
    # evaluate variants
    #################################################################
    all_y_mut_diff_fwd = []  # Will store y_mut_diff_fwd for each variant
    all_y_mut_diff_rev = []  # Will store y_mut_diff_rev for each variant
    processed_indices = []  # Track which variants were successfully processed

    print(f"\nEvaluating variants...")
    for variant_num, (idx, variant) in enumerate(
        tqdm(variants_in_gene.iterrows(), total=len(variants_in_gene), desc="Variants")
    ):
        seq_pos = int(variant["seq_pos"])
        ref_allele = str(variant["ref"]).upper()
        alt_allele = str(variant["alt"]).upper()

        # Verify reference allele matches
        ref_base = ref_dna[seq_pos].upper()
        if ref_base != ref_allele:
            print(
                f"Warning: Reference mismatch at position {variant['pos']}: expected {ref_allele}, found {ref_base}. Skipping."
            )
            continue

        # Find backgrounds where the nucleotide at seq_pos matches the REF allele
        ref_nt_idx = np.argmax(
            dna.dna_1hot(ref_allele)[0]
        )  # Get nucleotide index (0-3)
        matching_bg_indices = np.flatnonzero(
            background_nt_indices[:, seq_pos] == ref_nt_idx
        )

        # Sample k_background from matching backgrounds (or use all if fewer available)
        if len(matching_bg_indices) < args.k_background:
            print(
                f"Warning: Only {len(matching_bg_indices)} backgrounds have REF allele at position {variant['pos']}, using all of them."
            )
            sampled_bg_indices = matching_bg_indices
        else:
            sampled_bg_indices = np.random.choice(
                matching_bg_indices, args.k_background, replace=False
            )

        # Evaluate variant on sampled backgrounds
        variant_diff_fwd = []
        variant_diff_rev = []

        with torch.inference_mode(), torch.amp.autocast(
            device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
        ):
            for bg_idx in sampled_bg_indices:
                bg_dna = background_dna_list[bg_idx]

                # Create alt DNA sequence by replacing the nucleotide in the background sequence
                alt_dna = bg_dna[:seq_pos] + alt_allele + bg_dna[seq_pos + 1 :]

                # Convert alt DNA to one-hot and create tensor
                alt_1hot = dna.dna_1hot(alt_dna)
                alt_1hot_tensor = np.expand_dims(alt_1hot.T, axis=0)
                alt_1hot_tensor = torch.tensor(
                    alt_1hot_tensor, device=seqnn_model.device, dtype=torch.float32
                )

                # Predict forward and reverse separately
                alt_score_fwd = predict_gene_with_rc(
                    seqnn_model,
                    alt_1hot_tensor,
                    "forward",
                    targets_df,
                    gene_strand_mask,
                    gene_slice,
                )
                alt_score_rev = predict_gene_with_rc(
                    seqnn_model,
                    alt_1hot_tensor,
                    "reverse",
                    targets_df,
                    gene_strand_mask,
                    gene_slice,
                )

                # Calculate difference from background reference
                diff_fwd = alt_score_fwd - background_scores_fwd[bg_idx]
                diff_rev = alt_score_rev - background_scores_rev[bg_idx]

                variant_diff_fwd.append(diff_fwd)
                variant_diff_rev.append(diff_rev)

        # Average across sampled backgrounds for this variant
        avg_diff_fwd = np.mean(variant_diff_fwd, axis=0)  # shape: [n_targets]
        avg_diff_rev = np.mean(variant_diff_rev, axis=0)  # shape: [n_targets]

        all_y_mut_diff_fwd.append(avg_diff_fwd)
        all_y_mut_diff_rev.append(avg_diff_rev)
        processed_indices.append(idx)

        torch.cuda.empty_cache()

        if (variant_num + 1) % 100 == 0:
            gc.collect()

    gc.collect()

    #################################################################
    # create output matrix as 3D tensor [variants, targets, direction]
    #################################################################
    # Filter to only variants that were successfully processed
    variants_processed = variants_in_gene.loc[processed_indices]
    print(f"\nSuccessfully processed {len(variants_processed)} variants")
    if len(variants_processed) < len(variants_in_gene):
        print(f"Skipped {len(variants_in_gene) - len(variants_processed)} variants")

    # Save the processed variant annotation
    variants_processed.to_csv(f"{args.out_dir}/variants_annotation.tsv", sep="\t")

    # Convert lists to arrays
    y_mut_diff_fwd = np.array(all_y_mut_diff_fwd)  # shape: [n_variants, n_targets]
    y_mut_diff_rev = np.array(all_y_mut_diff_rev)  # shape: [n_variants, n_targets]

    # Stack to create 3D tensor: direction 0=forward, 1=reverse
    y_mut_diff = np.stack(
        [y_mut_diff_fwd, y_mut_diff_rev], axis=2
    )  # shape: [n_variants, n_targets, 2]

    # Save to HDF5 file
    matrices_file = f"{args.out_dir}/matrices_logSED.h5"
    print(f"\nSaving 3D tensor to {matrices_file} with shape {y_mut_diff.shape}")
    print(
        f"  Dimensions: [variants={y_mut_diff.shape[0]}, targets={y_mut_diff.shape[1]}, direction={y_mut_diff.shape[2]}]"
    )
    print(f"  Direction 0=forward, 1=reverse")
    print(f"  Each variant averaged over {args.k_background} sampled backgrounds.")
    with h5py.File(matrices_file, "w") as h5f:
        h5f.create_dataset("y_mut_diff", data=y_mut_diff, compression="gzip")
        h5f.create_dataset("k_background", data=args.k_background)
    print_cpu_memory_peak()
    print_gpu_memory_peak()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
