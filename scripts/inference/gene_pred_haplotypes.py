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
gene_pred_haplotypes.py

Predict gene expression for each haplotype of each sample in a phased VCF file.
Takes a single gene GTF and a phased VCF with SNPs.
Outputs two matrices (samples x targets) for haplotype 1 and haplotype 2.
"""


################################################################################
# helper functions
################################################################################
def parse_vcf_haplotypes(vcf_file, gene_chrom, gene_start, gene_end):
    """
    Parse phased VCF file and extract haplotype matrices.

    Returns:
        variants_df: DataFrame with variant info (CHROM, POS, REF, ALT)
        sample_names: List of sample IDs
        m_hap1: variants x samples matrix for haplotype 1
        m_hap2: variants x samples matrix for haplotype 2
    """
    vcf = pysam.VariantFile(vcf_file)

    # Get sample names
    sample_names = list(vcf.header.samples)

    variants_info = []
    hap1_data = []
    hap2_data = []

    print(f"Parsing VCF file: {vcf_file}")
    print(f"Gene region: {gene_chrom}:{gene_start}-{gene_end}")

    for record in vcf.fetch(gene_chrom, gene_start, gene_end):
        # Store variant info
        variants_info.append(
            {
                "CHROM": record.chrom,
                "POS": record.pos,
                "REF": record.ref,
                "ALT": (
                    ",".join([str(alt) for alt in record.alts]) if record.alts else "."
                ),
            }
        )

        # Extract haplotypes for all samples
        hap1_sample = []
        hap2_sample = []

        for sample in sample_names:
            gt = record.samples[sample]["GT"]

            # Handle phased genotypes (tuple like (0, 1) for "0|1")
            if gt is None or None in gt:
                # Missing genotype - treat as reference
                h1, h2 = 0, 0
            else:
                h1, h2 = gt[0], gt[1]

            # Replace missing (None) with 0 (reference)
            h1 = 0 if h1 is None else h1
            h2 = 0 if h2 is None else h2

            hap1_sample.append(h1)
            hap2_sample.append(h2)

        hap1_data.append(hap1_sample)
        hap2_data.append(hap2_sample)

    vcf.close()

    # Create DataFrames
    variants_df = pd.DataFrame(variants_info)
    m_hap1 = np.array(hap1_data, dtype=int)  # variants x samples
    m_hap2 = np.array(hap2_data, dtype=int)  # variants x samples

    print(f"Loaded {len(variants_df)} variants for {len(sample_names)} samples")

    return variants_df, sample_names, m_hap1, m_hap2


def apply_variants_to_sequence(ref_seq, variants_df, haplotype_vector, gene_start):
    """
    Apply variants from a single haplotype to the reference sequence.

    Args:
        ref_seq: Reference DNA sequence string
        variants_df: DataFrame with variant info (CHROM, POS, REF, ALT)
        haplotype_vector: Array of alleles (0=ref, 1=alt)
        gene_start: Start position of the sequence (0-based)

    Returns:
        Modified sequence string
    """
    # Convert to list for easier manipulation
    seq_list = list(ref_seq)

    # Extract only variants with alt allele (haplotype_vector == 1)
    alt_indices = np.where(haplotype_vector == 1)[0]
    alt_variants = variants_df.iloc[alt_indices]

    for _, variant in alt_variants.iterrows():
        pos = variant["POS"]
        ref_allele = variant["REF"]
        alt_allele = variant["ALT"]

        # Verify SNP (single base substitution)
        if len(ref_allele) != 1 or len(alt_allele) != 1:
            raise ValueError(
                f"Non-SNP variant at position {pos}: {ref_allele}->{alt_allele}. Only bi-allelic SNPs supported."
            )

        # Calculate position in sequence (0-based relative to seq_start)
        seq_pos = pos - 1 - gene_start

        # Verify we're within bounds
        if seq_pos < 0 or seq_pos >= len(seq_list):
            continue

        # Verify reference allele matches reference sequence
        ref_base = seq_list[seq_pos].upper()
        if ref_base != ref_allele.upper():
            print(
                f"Warning: Reference allele mismatch at position {pos}: VCF has {ref_allele}, sequence has {ref_base}"
            )

        # Replace the base
        seq_list[seq_pos] = alt_allele

    return "".join(seq_list)


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Predict gene expression for haplotypes in a phased VCF file."
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
        default="gene_pred_haplotypes_out",
        help="Output directory [Default: %(default)s]",
    )

    #################################################################
    # required arguments
    #################################################################
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file")
    parser.add_argument("genes_gtf_file", help="GTF file with single gene annotation")
    parser.add_argument("vcf_file", help="Phased VCF file with SNPs")

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

    # load transcriptome (expecting single gene)
    print(f"Loading gene from {args.genes_gtf_file}")
    transcriptome = bgene.Transcriptome(args.genes_gtf_file)
    num_genes = len(transcriptome.genes)

    if num_genes != 1:
        print(
            f"Warning: GTF file contains {num_genes} genes, expected 1. Using first gene."
        )

    gene_id = list(transcriptome.genes.keys())[0]
    gene = transcriptome.genes[gene_id]
    gene_name = gene.kv.get("gene_name", "NA")

    print(f"Processing gene: {gene_id} ({gene_name})")

    # set sequence boundaries
    set_bounds(gene, seq_len)

    # determine gene strand mask
    if gene.strand == "+":
        gene_strand_mask = (targets_df.strand != "-").to_numpy()
    else:
        gene_strand_mask = (targets_df.strand != "+").to_numpy()

    # parse VCF file
    variants_df, sample_names, m_hap1, m_hap2 = parse_vcf_haplotypes(
        args.vcf_file, gene.chrom, gene.seq_start, gene.seq_end
    )

    num_samples = len(sample_names)
    num_variants = len(variants_df)

    print(f"\nProcessing {num_samples} samples with {num_variants} variants")

    # Save variant info
    variants_df.to_csv(f"{args.out_dir}/variants_info.tsv", sep="\t", index=False)

    # open FASTA file
    fasta_open = pysam.Fastafile(args.fasta)

    # extract reference sequence
    print(
        f"\nExtracting reference sequence: {gene.chrom}:{gene.seq_start}-{gene.seq_end}"
    )
    try:
        ref_dna = fasta_open.fetch(gene.chrom, gene.seq_start, gene.seq_end)
    except Exception as e:
        print(f"Error: Could not fetch sequence: {e}")
        return

    # pad if necessary
    if len(ref_dna) < seq_len:
        ref_dna += "N" * (seq_len - len(ref_dna))

    print(f"Reference sequence length: {len(ref_dna)}")

    # determine gene slice for output
    seq_out_start = gene.seq_start + model_one_side_crop_bp
    seq_out_len = model_stride * target_length
    gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride)

    # prepare results storage
    predictions_hap1 = []
    predictions_hap2 = []

    #################################################################
    # process each sample
    #################################################################
    print("\nPredicting gene expression for each sample's haplotypes...")

    with torch.inference_mode(), torch.amp.autocast(
        device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
    ):
        for sample_idx, sample_id in enumerate(tqdm(sample_names)):
            # Process haplotype 1
            hap1_vector = m_hap1[:, sample_idx]
            hap1_seq = apply_variants_to_sequence(
                ref_dna, variants_df, hap1_vector, gene.seq_start
            )

            # Convert to 1-hot
            hap1_1hot = dna.dna_1hot(hap1_seq)
            hap1_1hot = np.expand_dims(hap1_1hot.T, axis=0)
            hap1_tensor = torch.tensor(
                hap1_1hot, device=seqnn_model.device, dtype=torch.float32
            )

            # Predict
            hap1_preds = predict_gene(
                seqnn_model, hap1_tensor, targets_df, gene_strand_mask, gene_slice
            )
            predictions_hap1.append(hap1_preds)

            # Process haplotype 2
            hap2_vector = m_hap2[:, sample_idx]
            hap2_seq = apply_variants_to_sequence(
                ref_dna, variants_df, hap2_vector, gene.seq_start
            )

            # Convert to 1-hot
            hap2_1hot = dna.dna_1hot(hap2_seq)
            hap2_1hot = np.expand_dims(hap2_1hot.T, axis=0)
            hap2_tensor = torch.tensor(
                hap2_1hot, device=seqnn_model.device, dtype=torch.float32
            )

            # Predict
            hap2_preds = predict_gene(
                seqnn_model, hap2_tensor, targets_df, gene_strand_mask, gene_slice
            )
            predictions_hap2.append(hap2_preds)

            # Cleanup
            del hap1_tensor, hap2_tensor
            torch.cuda.empty_cache()

    # close FASTA file
    fasta_open.close()

    #################################################################
    # save results
    #################################################################
    # Convert to arrays
    predictions_hap1_matrix = np.array(predictions_hap1)  # samples x targets
    predictions_hap2_matrix = np.array(predictions_hap2)  # samples x targets

    # Save haplotype 1 predictions
    hap1_df = pd.DataFrame(
        predictions_hap1_matrix, index=sample_names, columns=targets_strand_df.index
    )
    hap1_file = f"{args.out_dir}/haplotype1_predictions.tsv"
    hap1_df.to_csv(hap1_file, sep="\t")
    print(f"\nSaved haplotype 1 predictions to {hap1_file}")
    print(f"Shape: {hap1_df.shape[0]} samples x {hap1_df.shape[1]} targets")

    # Save haplotype 2 predictions
    hap2_df = pd.DataFrame(
        predictions_hap2_matrix, index=sample_names, columns=targets_strand_df.index
    )
    hap2_file = f"{args.out_dir}/haplotype2_predictions.tsv"
    hap2_df.to_csv(hap2_file, sep="\t")
    print(f"Saved haplotype 2 predictions to {hap2_file}")
    print(f"Shape: {hap2_df.shape[0]} samples x {hap2_df.shape[1]} targets")

    print("\nDone!")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
