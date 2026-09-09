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
ism_vanilla.py

Perform multi-mutation in silico saturation mutagenesis of a single gene sequence,
defined by a GTF file.
"""


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
        default="ensemble",
        choices=["forward", "reverse", "ensemble"],
        help="RC mode: forward, reverse, or ensemble [Default: %(default)s]",
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
        "-l",
        dest="mut_len",
        default=1000,
        type=int,
        help="Length of center sequence to mutate [Default: %(default)s]",
    )
    parser.add_argument(
        "--mut2center",
        type=int,
        default=0,
        help="Offset from sequence center for mutation region. Negative values shift left, positive right. [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        default="ism_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--k_background",
        type=int,
        default=10,
        help="Number of background sequences to generate [Default: %(default)s]",
    )
    parser.add_argument(
        "--mutagenesis_in_ism_region",
        action="store_true",
        default=False,
        help="Allow background mutations in ISM region (skip exclusion filtering) [Default: %(default)s]",
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

    # create seqnn model
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

    # mutation region 1-hot
    ref_mut_1hot = ref_1hot[mut_start:mut_end]

    # Create mutation annotation table
    mutation_annotation = create_x_mut_annotation(
        gene, mut_start, args.mut_len, ref_mut_1hot
    )
    mutation_annotation.to_csv(
        f"{args.out_dir}/mutation_annotation.tsv", sep="\t", index=False
    )

    #################################################################
    # design k background sequences
    #################################################################
    print(f"\nDesigning {args.k_background} background sequences...")

    # Design background mutations on full sequence, excluding ISM region
    # We'll exclude mutations in (mut_start-50, mut_end+50) window
    exclude_start = max(0, mut_start - 50)
    exclude_end = min(seq_len, mut_end + 50)

    # Design mutations for entire sequence using bins strategy
    bg_designer = MutationDesigner()
    bg_designer.design_initial_batch(
        seq_len, "bins", {"mut_reps": 10, "mut_distance_min": 50}
    )

    X_mut_all = (
        bg_designer.get_regression_matrix()
    )  # csr_matrix: (num_mutseqs, 3 * seq_len)
    print(f"\nTotal designed sequences: {X_mut_all.shape[0]}")

    # Sample k background sequences
    selected_indices = np.random.choice(
        X_mut_all.shape[0], args.k_background, replace=False
    )
    X_mut_selected = X_mut_all[selected_indices]
    print(f"Selected {args.k_background} background sequences")

    # Conditionally filter mutations in ISM region
    if args.mutagenesis_in_ism_region:
        print(f"\n MUTAGENESIS IN ISM REGION ENABLED:")
        print(f"   Skipping exclusion zone filtering")
        print(
            f"   Background mutations will be present in ISM region [{mut_start}, {mut_end})"
        )
        X_mut_selected_filtered = X_mut_selected
    else:
        # Zero out mutations in exclude region (mut_start-50, mut_end+50)
        # Create mask for columns to exclude: positions in exclude region, all 3 alternative nucleotides
        exclude_cols = []
        for pos in range(exclude_start, exclude_end):
            exclude_cols.extend([pos * 3, pos * 3 + 1, pos * 3 + 2])

        print(f"\n📊 MUTATION EXCLUSION ZONE:")
        print(f"   ISM region: [{mut_start}, {mut_end}) = {mut_end - mut_start} bp")
        print(
            f"   Exclusion zone: [{exclude_start}, {exclude_end}) = {exclude_end - exclude_start} bp"
        )
        print(
            f"   Excluded columns: {len(exclude_cols)} (should be {(exclude_end - exclude_start) * 3})"
        )

        # Count mutations before filtering
        mutations_before = []
        for idx in range(args.k_background):
            mut_vec = X_mut_selected[idx].toarray().flatten()
            num_muts = np.sum(mut_vec != 0)
            mutations_before.append(num_muts)
        print(
            f"   Mutations per sequence before filtering: mean={np.mean(mutations_before):.1f}, std={np.std(mutations_before):.1f}"
        )

        # Zero out excluded columns in selected sequences (use LIL format for efficient modification)
        X_mut_selected_lil = X_mut_selected.tolil()
        X_mut_selected_lil[:, exclude_cols] = 0
        X_mut_selected_filtered = X_mut_selected_lil.tocsr()

        # Count mutations after filtering
        mutations_after = []
        mutations_in_ism = []
        for idx in range(args.k_background):
            mut_vec = X_mut_selected_filtered[idx].toarray().flatten()
            num_muts = np.sum(mut_vec != 0)
            mutations_after.append(num_muts)

            # Count mutations in ISM region specifically
            ism_cols = []
            for pos in range(mut_start, mut_end):
                ism_cols.extend([pos * 3, pos * 3 + 1, pos * 3 + 2])
            ism_muts = np.sum(mut_vec[ism_cols] != 0)
            mutations_in_ism.append(ism_muts)

        print(
            f"   Mutations per sequence after filtering: mean={np.mean(mutations_after):.1f}, std={np.std(mutations_after):.1f}"
        )
        print(
            f"   Mutations in ISM region after filtering: mean={np.mean(mutations_in_ism):.1f}, max={np.max(mutations_in_ism)}"
        )

    # Convert selected background sequences to 1hot format
    ref_1hot_full = ref_1hot.copy()  # (seq_len, 4)
    background_1hot_list = []

    print(f"\n🧬 CONVERTING BACKGROUND SEQUENCES TO 1HOT:")
    for idx in range(args.k_background):
        # Get mutation vector for this background (with exclude region zeroed out)
        bg_mut_vector = (
            X_mut_selected_filtered[idx].toarray().flatten()
        )  # (3 * seq_len,)

        # Convert to 1hot sequence
        bg_1hot = mut_vector_to_1hot(bg_mut_vector, ref_1hot_full)
        background_1hot_list.append(bg_1hot)

        # Count actual differences from reference
        diff_positions = np.where(np.any(bg_1hot != ref_1hot_full, axis=1))[0]
        ism_diff = np.sum((diff_positions >= mut_start) & (diff_positions < mut_end))
        outside_diff = np.sum(
            (diff_positions < mut_start) | (diff_positions >= mut_end)
        )

        if idx < 3:  # Show details for first 3
            print(
                f"   Background {idx}: {len(diff_positions)} differences ({ism_diff} in ISM, {outside_diff} outside)"
            )

    # Check if all backgrounds are identical (bug check)
    all_identical = True
    for idx in range(1, args.k_background):
        if not np.array_equal(background_1hot_list[0], background_1hot_list[idx]):
            all_identical = False
            break

    if all_identical:
        print(f"   ⚠️  ERROR: All background sequences are IDENTICAL!")
    else:
        print(f"   ✓ Background sequences differ from each other")

    #################################################################
    # design vanilla ISM for mutation region
    #################################################################
    # Design vanilla ISM for the mutation region only
    ism_designer = MutationDesigner()
    ism_designer.design_initial_batch(args.mut_len, "vanilla")
    X_mut3_ism = (
        ism_designer.get_mutation_matrix()
    )  # dense array: (mut_len * 3, mut_len, 3)

    print(f"\nVanilla ISM design: {len(X_mut3_ism)} sequences")

    #################################################################
    # forward pass for k backgrounds
    #################################################################
    all_y_mut_diff = []  # Will store k y_mut_diff matrices

    for bg_idx in range(args.k_background):
        print(f"\nProcessing background {bg_idx+1}/{args.k_background}...")

        bg_1hot = background_1hot_list[bg_idx]

        # Convert to torch tensor
        bg_1hot_t = np.expand_dims(bg_1hot.T, axis=0)
        bg_1hot_t = torch.tensor(
            bg_1hot_t, device=seqnn_model.device, dtype=torch.float32
        )

        # Get reference prediction for this background
        with torch.inference_mode(), torch.amp.autocast(
            device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
        ):
            if args.rc == "ensemble":
                bg_ref_score = predict_gene(
                    seqnn_model, bg_1hot_t, targets_df, gene_strand_mask, gene_slice
                )
            else:
                bg_ref_score = predict_gene_with_rc(
                    seqnn_model,
                    bg_1hot_t,
                    args.rc,
                    targets_df,
                    gene_strand_mask,
                    gene_slice,
                )

        # Perform ISM on this background
        mut_scores = []
        skipped_count = 0

        with torch.inference_mode(), torch.amp.autocast(
            device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
        ):
            for si in tqdm(
                range(ism_designer.num_mutseqs), desc=f"ISM on background {bg_idx+1}"
            ):
                # Check if mutation position matches reference in this background
                # Only compute ISM if the position is not already mutated in background
                coo_mut = X_mut3_ism[si]  # Dense array: (mut_len, 3)

                # Find mutation position (vanilla ISM: only one position per sequence)
                mut_pos = np.where(np.any(coo_mut != 0, axis=1))[0][0]
                abs_pos = mut_start + mut_pos  # absolute position in full sequence

                # Check if this position matches reference in the background
                position_matches_ref = np.array_equal(
                    bg_1hot[abs_pos], ref_1hot[abs_pos]
                )

                if position_matches_ref:
                    # Position matches reference, safe to apply ISM mutation
                    alt_1hot = make_alt(bg_1hot_t, mut_start, X_mut3_ism[si])
                    if args.rc == "ensemble":
                        alt_score = predict_gene(
                            seqnn_model,
                            alt_1hot,
                            targets_df,
                            gene_strand_mask,
                            gene_slice,
                        )
                    else:
                        alt_score = predict_gene_with_rc(
                            seqnn_model,
                            alt_1hot,
                            args.rc,
                            targets_df,
                            gene_strand_mask,
                            gene_slice,
                        )
                    mut_scores.append(alt_score)
                else:
                    # Position already mutated in background, use NaN
                    mut_scores.append(np.full_like(bg_ref_score, np.nan))
                    skipped_count += 1

                torch.cuda.empty_cache()

                if (si + 1) % 100 == 0:
                    gc.collect()

        if skipped_count > 0:
            print(
                f"   Skipped {skipped_count}/{ism_designer.num_mutseqs} ISM variants (position already mutated in background)"
            )

        gc.collect()

        # Calculate y_mut_diff for this background
        y_mut = np.array(mut_scores)
        y_mut_diff = y_mut - bg_ref_score[np.newaxis, :]
        all_y_mut_diff.append(y_mut_diff)

    #################################################################
    # save results
    #################################################################
    # Stack all y_mut_diff matrices along new axis: (k, num_mutations, num_targets)
    all_y_mut_diff = np.stack(all_y_mut_diff, axis=0)

    # Stack all background full sequences: (k, seq_len, 4)
    all_background_1hot = np.stack(background_1hot_list, axis=0)

    # Save to HDF5 file
    matrices_file = f"{args.out_dir}/matrices_logSED_mutate_background.h5"
    print(f"\nSaving {args.k_background} background results to {matrices_file}")
    with h5py.File(matrices_file, "w") as h5f:
        h5f.create_dataset("y_mut_diff", data=all_y_mut_diff, compression="gzip")
        h5f.create_dataset(
            "background_1hot", data=all_background_1hot, compression="gzip"
        )
        h5f.create_dataset("k_background", data=args.k_background)

    print_cpu_memory_peak()
    print_gpu_memory_peak()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
