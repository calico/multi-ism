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
    # design matrix
    #################################################################
    # design matrix for vanilla ISM
    designer = MutationDesigner()
    designer.design_initial_batch(args.mut_len, "vanilla")
    X_mut3 = designer.get_mutation_matrix()  # shape: (num_mutseqs, mut_len, 3)

    #################################################################
    # forward pass
    #################################################################
    mut_scores = []

    ref_1hot = np.expand_dims(ref_1hot.T, axis=0)
    ref_1hot = torch.tensor(ref_1hot, device=seqnn_model.device, dtype=torch.float32)

    with torch.inference_mode(), torch.amp.autocast(
        device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
    ):
        for si in tqdm(range(designer.num_mutseqs)):
            # make alt sequence
            # predict
            alt_1hot = make_alt(ref_1hot, mut_start, X_mut3[si])
            if args.rc == "ensemble":
                alt_score = predict_gene(
                    seqnn_model, alt_1hot, targets_df, gene_strand_mask, gene_slice
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
            torch.cuda.empty_cache()

            if (si + 1) % 100 == 0:
                gc.collect()

    gc.collect()

    #################################################################
    # create output matrix
    #################################################################
    y_mut = np.array(mut_scores)

    # Get reference prediction for logSED
    with torch.inference_mode(), torch.amp.autocast(
        device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
    ):
        if args.rc == "ensemble":
            ref_score = predict_gene(
                seqnn_model, ref_1hot, targets_df, gene_strand_mask, gene_slice
            )
        else:
            ref_score = predict_gene_with_rc(
                seqnn_model, ref_1hot, args.rc, targets_df, gene_strand_mask, gene_slice
            )

    y_mut_diff = y_mut - ref_score[np.newaxis, :]  # Broadcasting for each target

    # Save to HDF5 file similar to ism_design.py
    matrices_file = f"{args.out_dir}/matrices_logSED.h5"
    print(f"Saving matrices to {matrices_file}")
    with h5py.File(matrices_file, "w") as h5f:
        h5f.create_dataset("y_mut_diff", data=y_mut_diff, compression="gzip")
        h5f.create_dataset("ref_mut_1hot", data=ref_mut_1hot, compression="gzip")

    print_cpu_memory_peak()
    print_gpu_memory_peak()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
