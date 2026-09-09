import json
import numpy as np
import pandas as pd
import sys
import resource
import psutil
import torch
from scipy.sparse import csr_matrix, coo_matrix, issparse

from baskerville_torch import dataset
from baskerville_torch import seqnn
from baskerville_torch import dna
import baskerville_torch.gene as bgene
from matplotlib import patches


def load_ism_scores(ism_scores_pt_file):
    # ism_scores.pt is a sparse tensor saved by torch.save.
    # read and convert to scipy csr_matrix: (seqlen*3, n_targets)
    ism_tensor = torch.load(ism_scores_pt_file)
    indices = ism_tensor.indices().cpu().numpy()
    values = ism_tensor.values().cpu().numpy().astype(np.float32)
    shape = ism_tensor.shape
    scipy_coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)
    ism_m = scipy_coo.T.tocsr()
    return ism_m


def coef2score(ref_mut_1hot, coefs, mut_len, target_num, normalize=False):
    """Convert regression coefficients to position-specific nucleotide scores.

    Args:
        ref_mut_1hot: Reference sequence one-hot encoded, shape (mut_len, 4)
        coefs: Regression coefficients, shape (target_num, mut_len*3)
        mut_len: Length of mutation region
        target_num: Number of targets
        normalize: If True, normalize scores by subtracting mean across nucleotides at each position

    Returns:
        np.ndarray: Position-specific scores, shape (mut_len, 4, target_num)
    """
    # Initialize scores
    # coefs shape: (target_num, mut_len)
    seq_scores = np.zeros((mut_len, 4, target_num), dtype="float32")

    # for each mutated position
    ci = 0
    for mi in range(mut_len):
        # Positions with no reference base (e.g. genomic 'N') still carry 3
        # coefficients in the tensor like any other position. Leave their score
        # at 0 but advance the coefficient index by 3 to stay aligned.
        if not ref_mut_1hot[mi].any():
            ci += 3
            continue
        # for each nucleotide
        for ni in range(4):
            if ref_mut_1hot[mi, ni]:
                seq_scores[mi, ni, :] = 0
            else:
                seq_scores[mi, ni, :] = coefs[:, ci]
                ci += 1

    # normalize positions
    if normalize:
        seq_scores -= seq_scores.mean(axis=1, keepdims=True)

    return seq_scores


def mut_vector_to_1hot(mut_vector, ref_mut_1hot):
    """Convert a flattened mutation vector (L×3) to one-hot encoded sequence (L×4).

    The mutation vector encodes 3 alternative nucleotides for each position,
    excluding the reference nucleotide. This function reconstructs the full
    one-hot encoded sequence by applying mutations to the reference.

    Args:
        mut_vector: 1D numpy array of length (L*3) that will be reshaped to (L, 3)
        ref_mut_1hot: Reference sequence one-hot encoded, shape (L, 4)
                     Each row is [A, C, G, T] with one position = 1

    Returns:
        np.ndarray: One-hot encoded mutated sequence, shape (L, 4)
    """
    L = ref_mut_1hot.shape[0]

    # Validate input
    if mut_vector.ndim != 1:
        raise ValueError(
            f"mut_vector must be 1D array, got {mut_vector.ndim}D with shape {mut_vector.shape}"
        )
    if len(mut_vector) != L * 3:
        raise ValueError(f"mut_vector length {len(mut_vector)} must equal L*3 = {L*3}")
    if ref_mut_1hot.shape[1] != 4:
        raise ValueError(
            f"ref_mut_1hot must have 4 columns (A,C,G,T), got shape {ref_mut_1hot.shape}"
        )

    # Reshape to (L, 3)
    mut_vector = mut_vector.reshape(L, 3)

    # Validate that each position has at most 1 mutation
    muts_per_position = (mut_vector != 0).sum(axis=1)
    if (muts_per_position > 1).any():
        raise ValueError(
            f"Positions with >1 mutation: {np.where(muts_per_position > 1)[0]}"
        )

    result_1hot = ref_mut_1hot.copy()

    has_mut = mut_vector.sum(axis=1) > 0  # shape: (L,)
    if has_mut.any():
        # Get indices of mutated positions and their alternative indices
        mut_positions = np.where(has_mut)[0]
        alt_indices = np.argmax(
            mut_vector[mut_positions], axis=1
        )  # which of 3 alternatives

        # Get reference nucleotides at mutated positions
        ref_nts = np.argmax(ref_mut_1hot[mut_positions], axis=1)

        alt_lookup = np.array(
            [
                [1, 2, 3],  # A -> [C, G, T]
                [0, 2, 3],  # C -> [A, G, T]
                [0, 1, 3],  # G -> [A, C, T]
                [0, 1, 2],  # T -> [A, C, G]
            ]
        )

        target_nts = alt_lookup[ref_nts, alt_indices]
        result_1hot[mut_positions, :] = 0
        result_1hot[mut_positions, target_nts] = 1

    return result_1hot


def predict_gene(model, seq_1hot, targets_df, gene_strand_mask, gene_slice):
    # when model.ensemble_rc=True, this predicts mean of forward and reverse predictions
    preds = model(seq_1hot).coverage[0]
    preds = dataset.untransform_preds(preds, targets_df)
    preds_gene = preds[gene_strand_mask][:, gene_slice]
    preds_score = torch.log2(preds_gene.sum(axis=1) + 1).cpu().numpy()
    del preds, preds_gene
    return preds_score


def predict_gene_with_rc(
    model, seq_1hot, rc_mode, targets_df, gene_strand_mask, gene_slice
):
    # used when model.ensemble_rc=False
    # this predicts forward or reverse separately
    if rc_mode == "forward":
        preds = model(seq_1hot).coverage[0]

    elif rc_mode == "reverse":
        seq_rc = dna.torch_rc(seq_1hot)
        preds_rc = model(seq_rc).coverage[0]
        preds = torch.flip(preds_rc, [1])  # Flip predictions along length axis
        if model.strand_pair is not None:
            preds = preds[model.strand_pair, :]
    else:
        raise ValueError(f"Invalid rc_mode: {rc_mode}")

    # sum across bins and log2 transform
    preds = dataset.untransform_preds(preds, targets_df)
    preds_gene = preds[gene_strand_mask][:, gene_slice]
    preds_score = torch.log2(preds_gene.sum(axis=1) + 1).cpu().numpy()
    del preds, preds_gene
    return preds_score


def create_seqnn_model(
    params_file, model_file, targets_file, mix_dtype="bfloat16", ensemble_rc=False
):
    """Create and initialize a SeqNN model object."""
    # Read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # Read targets
    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")

    # Prep strand
    targets_strand_df = dataset.targets_prep_strand(targets_df)

    # Set strand pairs (using new indexing)
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = [orig_new_index[ti] for ti in targets_df.strand_pair]
    params_model["strand_pair"] = np.array(targets_strand_pair)
    params_model["verbose"] = False  # disable model summary

    # Create model
    seqnn_model = seqnn.SeqNN(params_model, output_slice=targets_df.index)
    seqnn_model.restore(model_file)
    seqnn_model.ensemble_rc = ensemble_rc
    seqnn_model.model.eval()

    seqnn_model.mix_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[mix_dtype]

    return seqnn_model


def get_mutation_region(seq_length, mut_len, mut2center=0):
    # Calculate the start and end positions of mutation region.
    seq_center = seq_length // 2
    mut_start = seq_center - (mut_len // 2) + mut2center
    mut_end = mut_start + mut_len
    if mut_start < 0 or mut_end > seq_length:
        raise ValueError(
            f"Invalid mutation region: [{mut_start}, {mut_end}] exceeds sequence bounds [0, {seq_length}]. "
        )
    return mut_start, mut_end


def set_bounds(gene, seq_len):
    # Set surrounding sequence boundaries for a gene.
    half_len = seq_len // 2
    midp = gene.midpoint()
    seq_start = max(midp - half_len, 0)
    seq_end = seq_start + seq_len
    gene.seq_start = seq_start
    gene.seq_end = seq_end

    return seq_start, seq_end


def make_alt(ref_1hot, mut_start, X_mut3):
    """Make alternative 1-hot sequence, using given mutations.

    Args:
        ref_1hot: Reference sequence one-hot encoded, shape (1, 4, seq_len)
        mut_start: Starting position of mutation region (int)
        X_mut3: Mutation matrix, shape (mut_len, 3) - dense or sparse
    """

    nts = [0, 1, 2, 3]  # A, C, G, T

    # copy reference
    alt_1hot = ref_1hot.clone()

    # find mutations - handle sparse matrices
    if issparse(X_mut3):
        # Convert to COO format if not already
        X_mut3_coo = X_mut3.tocoo()
        mut_pos = X_mut3_coo.row
        mut_nt = X_mut3_coo.col
    else:
        # Original dense matrix handling
        mut_pos, mut_nt = np.where(X_mut3)

    for mi in range(len(mut_pos)):
        # update index based on mut_start
        msi = mut_start + mut_pos[mi]

        # Check bounds
        seq_len = ref_1hot.shape[2]
        if msi >= seq_len:
            raise ValueError(
                f"Mutation position out of bounds: mut_pos[{mi}]={mut_pos[mi]}, "
                f"mut_start={mut_start}, msi={msi}, seq_len={seq_len}. "
                f"X_mut3 shape={X_mut3.shape}, nnz={X_mut3.nnz if issparse(X_mut3) else 'N/A'}"
            )

        # determine nucleotide index based on reference
        nts_mut = [nts[ni] for ni in range(4) if alt_1hot[0, ni, msi] != 1]
        if len(nts_mut) == 4:
            raise NotImplementedError(
                f"Requesting padded nucleotide mutation at position {msi}. "
                f"mut_pos[{mi}]={mut_pos[mi]}, mut_start={mut_start}. "
                f"ref_1hot at this position: {alt_1hot[0, :, msi].tolist()}"
            )
        ni = nts_mut[mut_nt[mi]]

        # zero and set
        alt_1hot[0, :, msi] = 0
        alt_1hot[0, ni, msi] = 1

    return alt_1hot


def print_cpu_memory_peak():
    """Print peak CPU memory usage in GB."""
    peak_cpu_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS returns bytes
        peak_cpu_gb = peak_cpu_kb / (1024 * 1024 * 1024)  # bytes to GB
    else:  # Linux returns kilobytes
        peak_cpu_gb = peak_cpu_kb / (1024 * 1024)  # KB to GB
    print(f"Peak CPU memory: {peak_cpu_gb:.2f} GB")


def print_gpu_memory_peak():
    """Print peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        peak_gpu_gb = torch.cuda.max_memory_allocated() / (
            1024 * 1024 * 1024
        )  # Convert to GB
        print(f"Peak GPU memory: {peak_gpu_gb:.2f} GB")


def filter_transcriptome_by_range(transcriptome, chrom, start, end):
    """Filter a transcriptome to only include genes that overlap with a genomic range.

    Args:
        transcriptome (Transcriptome): The input transcriptome object
        chrom (str): Chromosome name
        start (int): Start position (0-based, inclusive)
        end (int): End position (0-based, exclusive)

    Returns:
        Transcriptome: A new transcriptome object containing only overlapping genes
    """

    filtered_transcriptome = bgene.Transcriptome.__new__(bgene.Transcriptome)
    filtered_transcriptome.genes = {}
    for gene_id, gene in transcriptome.genes.items():
        if gene.chrom == chrom:
            gene_start, gene_end = gene.span()

            if gene_end > start and gene_start < end:
                new_gene = bgene.Gene(gene.chrom, gene.strand, gene.kv, gene.name)
                for exon in gene.exons:
                    new_gene.exons[exon.begin : exon.end] = exon.data

                filtered_transcriptome.genes[gene_id] = new_gene

    return filtered_transcriptome


def plot_gene_track(transcriptome, chrom, start, end, ax):
    """
    Plot gene track for genes in the specified region with strand coloring.
    Overlapping genes are placed on separate rows.
    Labels are centered at the TSS, dodge vertically, and connect to the TSS.
    """

    # Collect genes in region
    genes_in_region = []
    for gene_id, gene in transcriptome.genes.items():
        if gene.chrom == chrom:
            gene_start, gene_end = gene.span()
            if gene_end > start and gene_start < end:
                genes_in_region.append((gene_id, gene))

    if not genes_in_region:
        ax.text(
            0.5,
            0.5,
            f"No genes found in {chrom}:{start}-{end}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{chrom} Position (bp)")
        ax.set_title("Gene Track")
        return

    exon_height = 0.15
    row_intervals = []
    placed_bboxes = []  # store label bounding boxes in data coords

    def assign_row(gene_start, gene_end):
        """Place gene in first row without overlap."""
        for i, intervals in enumerate(row_intervals):
            if all(gene_end < s or gene_start > e for (s, e) in intervals):
                intervals.append((gene_start, gene_end))
                return i
        row_intervals.append([(gene_start, gene_end)])
        return len(row_intervals) - 1

    def dodge_label(text_obj, step=0.15):
        """Shift label upward until no overlap, return final y position."""
        renderer = ax.figure.canvas.get_renderer()
        bbox = text_obj.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())

        while any(bbox_data.overlaps(prev) for prev in placed_bboxes):
            # move up
            x, y = text_obj.get_position()
            text_obj.set_y(y + step)
            bbox = text_obj.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(ax.transData.inverted())

        placed_bboxes.append(bbox_data)
        return text_obj.get_position()[1]

    # Plot genes
    for gene_id, gene in genes_in_region:
        gene_start, gene_end = gene.span()
        row = assign_row(gene_start, gene_end)
        y_center = 0.2 + row * 0.4

        # Strand color
        color = "#ef8a62" if gene.strand == "+" else "#67a9cf"

        # Backbone
        ax.plot(
            [gene_start, gene_end],
            [y_center, y_center],
            color=color,
            linewidth=2,
            alpha=0.7,
        )

        # Exons
        for exon in gene.get_exons():
            exon_start, exon_end = exon.begin, exon.end
            rect = patches.Rectangle(
                (exon_start, y_center - exon_height / 2),
                exon_end - exon_start,
                exon_height,
                facecolor=color,
                edgecolor=color,
            )
            ax.add_patch(rect)

        # TSS position (5' end: gene_start for + strand, gene_end for - strand)
        tss = gene_start if gene.strand == "+" else gene_end
        gene_name = gene.name if gene.name else gene_id

        # Clip both gene ends to visible plotting range
        visible_gene_start = max(start, gene_start)
        visible_gene_end = min(end, gene_end)

        # Position label at TSS if visible, otherwise at the visible edge
        if start <= tss <= end:
            label_pos = tss
        elif gene.strand == "+":
            # TSS at 5' (start), if outside use visible start
            label_pos = visible_gene_start
        else:
            # TSS at 3' (end), if outside use visible end
            label_pos = visible_gene_end

        # Place text centered on label position
        base_y = y_center + exon_height / 2 + 0.1
        text = ax.text(
            label_pos,
            base_y,
            gene_name,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        # Force draw once so bbox is available
        ax.figure.canvas.draw()
        final_y = dodge_label(text)

        # Connector line: from gene backbone to label position
        ax.plot(
            [label_pos, label_pos],
            [y_center, final_y],
            color="gray",
            linestyle="dashed",
            linewidth=0.8,
            alpha=0.7,
            zorder=0,
        )

    # Format
    ax.set_xlim(start, end)
    ax.set_xlabel(f"{chrom} Position (bp)")
    ax.set_ylabel("Genes")
    ax.grid(False)
    ax.set_yticks([])
    ax.ticklabel_format(style="plain", axis="x")
    ax.relim()
    ax.autoscale_view()


def create_x_mut_annotation(gene, mut_start, mut_len, ref_mut_1hot):
    """
    Create annotation dataframe for X_mut matrix columns.

    Args:
        gene: Gene object with chromosome and genomic coordinates
        mut_start: Start position of mutation region relative to sequence
        mut_len: Length of mutation region
        ref_mut_1hot: Reference sequence one-hot encoded for mutation region, shape (mut_len, 4)

    Returns:
        pd.DataFrame with columns: coef_idx, chr, pos, ref_nucleotide, alt_nucleotide
    """
    nt_map = np.array(["A", "C", "G", "T"])

    # Get reference nucleotides for all positions at once
    ref_nt_indices = np.argmax(ref_mut_1hot, axis=1)  # shape: (mut_len,)
    ref_nucleotides = nt_map[ref_nt_indices]  # shape: (mut_len,)

    # Create arrays for all positions
    total_rows = mut_len * 3
    coef_idx = np.arange(total_rows)  # 0, 1, 2, ..., total_rows-1
    pos_indices = np.repeat(np.arange(mut_len), 3)  # 0,0,0, 1,1,1, 2,2,2, ...
    genomic_positions = gene.seq_start + mut_start + pos_indices + 1

    # Create chromosome array
    chr_array = np.full(total_rows, gene.chrom, dtype=object)

    # Create reference nucleotide array (repeat each ref_nt 3 times)
    ref_nt_array = np.repeat(ref_nucleotides, 3)

    # Create alternative nucleotide array
    alt_nt_array = np.empty(total_rows, dtype=object)
    for i in range(mut_len):
        ref_nt = ref_nucleotides[i]
        alt_nts = nt_map[nt_map != ref_nt]  # 3 alternative nucleotides
        alt_nt_array[i * 3 : (i + 1) * 3] = alt_nts

    # Create DataFrame directly from arrays
    return pd.DataFrame(
        {
            "coef_idx": coef_idx,
            "chr": chr_array,
            "pos": genomic_positions,
            "ref": ref_nt_array,
            "alt": alt_nt_array,
        }
    )
