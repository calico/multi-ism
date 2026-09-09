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
import os

import numpy as np
import pandas as pd
import pysam
import torch
from tqdm import tqdm

from baskerville_torch import dataset
from baskerville_torch import dna
from baskerville_torch import gene as bgene
from mism.core.mism_utils import create_seqnn_model, predict_gene, set_bounds

"""
gene_pred.py

Predict log2 gene expression values for all genes in a GTF file using a SeqNN model.
Outputs a dataframe with gene_id, gene_name, and predicted log2 expression for each target.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Predict gene expression for all genes in a GTF file."
    )

    #################################################################
    # model inference parameters
    #################################################################
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        required=True,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "-m",
        "--mix_dtype",
        dest="mix_dtype",
        default="bfloat16",
        help="Mixed precision dtype [Default: %(default)s]",
    )
    parser.add_argument(
        "-f",
        "--fasta",
        default=None,
        required=True,
        help="Genome FASTA for sequences",
    )

    #################################################################
    # output parameters
    #################################################################
    parser.add_argument(
        "-o",
        dest="out_dir",
        default="gene_pred_out",
        help="Output directory [Default: %(default)s]",
    )

    #################################################################
    # required arguments
    #################################################################
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file")
    parser.add_argument("genes_gtf_file", help="GTF file with gene annotations")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # read targets
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    targets_strand_df = dataset.targets_prep_strand(targets_df)
    num_targets_strand = targets_strand_df.shape[0]

    # Save targets info
    targets_strand_df.to_csv(f"{args.out_dir}/targets_strand.txt", sep="\t")

    # create seqnn model with ensemble_rc=True for averaging forward and reverse
    seqnn_model = create_seqnn_model(
        args.params_file,
        args.model_file,
        args.targets_file,
        args.mix_dtype,
        ensemble_rc=True,
    )
    model_stride = seqnn_model.output_stride()
    model_one_side_crop_bp = seqnn_model.output_crop_bp()
    target_length = seqnn_model.output_length()
    seq_len = seqnn_model.seq_length

    # load transcriptome
    print(f"Loading genes from {args.genes_gtf_file}")
    transcriptome = bgene.Transcriptome(args.genes_gtf_file)
    num_genes = len(transcriptome.genes)
    print(f"Found {num_genes} genes")

    # open FASTA file
    fasta_open = pysam.Fastafile(args.fasta)

    # prepare results storage
    gene_info_list = []
    gene_predictions_list = []

    #################################################################
    # process each gene
    #################################################################
    print("Predicting gene expression...")
    with torch.inference_mode(), torch.amp.autocast(
        device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
    ):
        for gene_id, gene in tqdm(transcriptome.genes.items(), total=num_genes):
            # set sequence boundaries
            set_bounds(gene, seq_len)

            # determine gene strand mask
            if gene.strand == "+":
                gene_strand_mask = (targets_df.strand != "-").to_numpy()
            else:
                gene_strand_mask = (targets_df.strand != "+").to_numpy()

            # extract sequence
            try:
                ref_dna = fasta_open.fetch(gene.chrom, gene.seq_start, gene.seq_end)
            except Exception as e:
                print(f"Warning: Could not fetch sequence for {gene_id}: {e}")
                continue

            # pad if necessary
            if len(ref_dna) < seq_len:
                ref_dna += "N" * (seq_len - len(ref_dna))

            # convert to 1-hot encoding
            ref_1hot = dna.dna_1hot(ref_dna)
            ref_1hot = np.expand_dims(ref_1hot.T, axis=0)
            ref_1hot_tensor = torch.tensor(
                ref_1hot, device=seqnn_model.device, dtype=torch.float32
            )

            # determine gene slice for output
            seq_out_start = gene.seq_start + model_one_side_crop_bp
            seq_out_len = model_stride * target_length
            gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride)

            # predict gene expression
            gene_preds = predict_gene(
                seqnn_model, ref_1hot_tensor, targets_df, gene_strand_mask, gene_slice
            )

            # store gene info
            gene_name = gene.kv.get("gene_name", "NA")
            gene_info_list.append(
                {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "chrom": gene.chrom,
                    "start": gene.seq_start,
                    "end": gene.seq_end,
                    "strand": gene.strand,
                }
            )

            # store predictions
            gene_predictions_list.append(gene_preds)

            # cleanup
            del ref_1hot_tensor
            torch.cuda.empty_cache()

    # close FASTA file
    fasta_open.close()

    #################################################################
    # save results
    #################################################################
    # Save gene metadata
    gene_info_df = pd.DataFrame(gene_info_list)
    gene_info_file = f"{args.out_dir}/gene_info.tsv"
    gene_info_df.to_csv(gene_info_file, sep="\t", index=False)
    print(f"\nSaved gene info to {gene_info_file}")
    print(f"Shape: {gene_info_df.shape[0]} genes x {gene_info_df.shape[1]} columns")

    # Save gene x targets predictions matrix
    predictions_matrix = np.array(gene_predictions_list)
    predictions_df = pd.DataFrame(
        predictions_matrix,
        index=gene_info_df["gene_id"],
        columns=targets_strand_df.index,
    )
    predictions_file = f"{args.out_dir}/gene_predictions.tsv"
    predictions_df.to_csv(predictions_file, sep="\t")
    print(f"Saved predictions matrix to {predictions_file}")
    print(f"Shape: {predictions_df.shape[0]} genes x {predictions_df.shape[1]} targets")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
