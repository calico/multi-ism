import numpy as np
import time
import random
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix


def coo_list2csr(X_mut3):
    # Convert X_mut3 (list of coo matrices) to X_mut (csr matrix)
    rows = []
    cols = []
    data = []
    num_mutseqs = len(X_mut3)
    mut_len, _ = X_mut3[0].shape

    for i, coo in enumerate(X_mut3):
        coo_rows = coo.row
        coo_cols = coo.col
        coo_data = coo.data

        # Adjust column indices to account for flattening
        adjusted_cols = coo_cols + (coo_rows * 3)

        # Add row offset for this sequence
        rows.extend([i] * len(coo_data))
        cols.extend(adjusted_cols)
        data.extend(coo_data)

    # Create single csr matrix
    X_mut = csr_matrix((data, (rows, cols)), shape=(num_mutseqs, 3 * mut_len))
    return X_mut


################################################################################
# class
################################################################################
class MutationDesigner:
    def __init__(self):
        """Initialize mutation designer with no parameters."""
        self.mut_len = None
        self.design_params = None
        self.X_mut3 = None
        self.X_mut = None
        self.num_mutseqs = 0

    def design_initial_batch(
        self, mut_len, design_method="vanilla", design_params=None
    ):
        """
        Design initial mutation matrix.

        Args:
            mut_len: Length of mutation region
            design_method: Method to use
                - "vanilla": Single mutations
                - "bins": Fixed-bin alternating strategy
                - "star-and-bar": Sequential uniform sampling with forced position coverage
                - "random": Random mutagenesis with Poisson-distributed mutation counts
            design_params: Dict with design parameters
                Common params: mut_reps, mut_distance_min
                Random method: mut_rate (default: 0.01), num_seq (default: 1000)

        Returns:
            X_mut3: 3D mutation matrix (num_mutseqs, mut_len, 3)
        """
        # Store parameters
        self.mut_len = mut_len
        self.design_params = design_params or {}

        if design_method == "vanilla":
            self.X_mut3 = mutation_design_matrix_vanilla(mut_len)
        elif design_method == "bins":
            mut_reps = self.design_params.get("mut_reps", 50)
            mut_distance_min = self.design_params.get("mut_distance_min", 50)
            self.X_mut3 = mutation_design_matrix_bins(
                mut_len, mut_reps, mut_distance_min
            )
        elif design_method == "star-and-bar":
            mut_reps = self.design_params.get("mut_reps", 50)
            min_dist = self.design_params.get("mut_distance_min", 50)
            self.X_mut3 = mutation_design_matrix_uniform_force(
                mut_len, mut_reps, min_dist
            )
        elif design_method == "random":
            mut_rate = self.design_params.get("mut_rate", 0.01)
            num_seq = self.design_params.get("num_seq", 1000)
            self.X_mut3 = mutation_design_random(mut_len, mut_rate, num_seq)
        else:
            raise ValueError(f"Invalid mutation design: {design_method}")

        self.num_mutseqs = len(self.X_mut3)

    def get_mutation_matrix(self):
        # return mutation matrix (num_mutseqs, mut_len, 3).
        if self.X_mut3 is None:
            raise ValueError("Must call design_initial_batch first")
        return self.X_mut3

    def get_regression_matrix(self):
        # return flattened mutation matrix (num_mutseqs, 3 * mut_len).
        if self.X_mut3 is None:
            raise ValueError("Must call design_initial_batch first")
        if isinstance(self.X_mut3, list):
            # when X_mut3 is list of coo matrices.
            self.X_mut = coo_list2csr(self.X_mut3)
        else:
            # when X_mut3 is dense.
            self.X_mut = self.X_mut3.reshape((self.num_mutseqs, 3 * self.mut_len))
        return self.X_mut

    def get_mutation_count(self):
        """Get the number of mutation sequences designed."""
        return self.num_mutseqs

    def design_adaptive_batch(self, regression_results, batch_size):
        """Placeholder"""
        pass

    def update_strategy(self, feedback):
        """Placeholder"""
        pass


################################################################################
# functions
################################################################################
def mutation_design_matrix_vanilla(mut_len):
    """Create a mutation design matrix for vanilla in-silico mutagenesis.
    For each position, creates 3 sequences, each with one mutation to a different nucleotide.

    Args:
        mut_len (int): Length of sequence to mutate

    Returns:
        np.ndarray: Boolean array of shape (n_sequences, mut_len, 3) where n_sequences = mut_len * 3
    """
    # For each position, we'll have 3 mutations (to each other nucleotide)
    n_sequences = mut_len * 3

    # Initialize the mutation matrix
    X_mut3 = np.zeros((n_sequences, mut_len, 3), dtype=bool)

    # For each position
    for pos in range(mut_len):
        # For each alternative nucleotide
        for alt in range(3):
            # Calculate sequence index
            seq_idx = pos * 3 + alt
            # Set the mutation
            X_mut3[seq_idx, pos, alt] = True

    return X_mut3


def mutation_design_matrix_bins(mut_len, mut_reps, mut_distance_min=100):
    """Design mutation sequences using a fixed-bin alternating strategy.

    Design intuition:
    This approach uses a structured, deterministic binning strategy:
    1. Divide the sequence into fixed-size bins of size mut_distance_min
    2. Generate sequences by alternating between even and odd bins:
       - First sequence: sample one mutation from each even bin without replacement
       - Second sequence: sample one mutation from each odd bin without replacement
    3. if a bin is exhaust of mutations, do nothing.
    4. continue until all mutations are sampled.
    5. Total number of sequences = 'mut_distance_min'*'mut_reps'*3*2

    Args:
        mut_len (int): Length of sequence to mutate
        mut_reps (int): Number of replicates needed for each mutation
        mut_distance_min (int): Size of each bin (minimum distance between mutations)
    """

    # 1) make a dict of dict: bin -> mutation (pos,nt) -> count.

    t0 = time.time()

    bin_starts = list(range(0, mut_len, mut_distance_min))
    bin_ranges = [
        (start, min(start + mut_distance_min, mut_len)) for start in bin_starts
    ]
    num_bins = len(bin_ranges)
    bin_dict = defaultdict(dict)
    for b, (start, end) in enumerate(bin_ranges):
        for pos in range(start, end):
            for nt in range(3):
                bin_dict[b][(pos, nt)] = 0

    sequences = []
    i = 0

    # Loop as long as there is any non-empty bin
    while any(bin_dict[b] for b in range(num_bins)):
        sequence = np.zeros((mut_len, 3), dtype=bool)
        selected_bins = [b for b in range(num_bins) if b % 2 == i % 2]

        for b in selected_bins:
            if not bin_dict[b]:
                continue
            available = list(bin_dict[b].keys())
            mut = random.choice(available)
            pos, nt = mut
            sequence[pos, nt] = True
            bin_dict[b][mut] += 1

            if bin_dict[b][mut] >= mut_reps:
                del bin_dict[b][mut]

        if sequence.any():
            sequences.append(coo_matrix(sequence))

        i += 1

    X_mut3 = sequences

    print(f"Design time: {time.time() - t0:.3f} seconds")
    print_mutation_design_stats(X_mut3, method_name="design_bins")

    return X_mut3


def mutation_design_random(mut_len, mut_rate, num_seq=1000):
    """Design mutation sequences using random mutagenesis with Poisson distribution.

    Design intuition:
    This approach uses stochastic random mutagenesis:
    1. For each sequence, draw k from Poisson(mut_rate * mut_len)
    2. Randomly select k unique positions to mutate
    3. For each position, randomly assign one of 3 nucleotides
    4. No guaranteed position coverage - purely random sampling

    Args:
        mut_len (int): Length of sequence to mutate
        mut_rate (float): Mutation rate (fraction of sequence length)
        num_seq (int): Number of sequences to generate (default: 1000)

    Returns:
        list: List of mutation sequences (coo_matrix)
    """
    t0 = time.time()

    avg_num_mut = mut_rate * mut_len
    sequences = []

    for _ in range(num_seq):
        # Draw number of mutations from Poisson distribution
        k = np.random.poisson(avg_num_mut)
        k = min(k, mut_len)  # Can't have more mutations than positions

        if k > 0:
            # Randomly select k positions without replacement
            positions = np.random.choice(mut_len, size=k, replace=False)

            # For each position, randomly select one of 3 nucleotides
            nucleotides = np.random.randint(0, 3, size=k)

            # Create sparse sequence
            current_seq = np.zeros((mut_len, 3), dtype=bool)
            current_seq[positions, nucleotides] = True

            sequences.append(coo_matrix(current_seq))
        else:
            # If k=0, create empty sequence
            sequences.append(coo_matrix((mut_len, 3), dtype=bool))

    X_mut3 = sequences

    print(f"Design time: {time.time() - t0:.3f} seconds")
    print_mutation_design_stats(X_mut3, method_name="design_random")

    return X_mut3


################################################################################
# star-and-bars design
################################################################################


def sample_mutation_positions_forced(
    mut_len, min_dist, n_mutation, force_position=None
):
    """Sample mutation positions with guaranteed minimum distance.

    Simple algorithm:
    1. Generate n_mutation positions using stars-and-bars
    2. If force_position provided:
       - Insert force_position into positions
       - Remove any positions within min_dist from force_position

    Args:
        mut_len (int): Total sequence length
        min_dist (int): Minimum distance between mutations
        n_mutation (int): Number of mutations to place
        force_position (int, optional): Position to force a mutation at

    Returns:
        np.ndarray: Sorted array of mutation positions
    """
    # Generate positions using stars-and-bars
    min_space = n_mutation + (n_mutation - 1) * min_dist
    if min_space > mut_len:
        raise ValueError(
            f"Cannot fit {n_mutation} mutations with min_dist={min_dist} in length {mut_len}"
        )

    remaining_space = mut_len - min_space
    if remaining_space == 0:
        positions = np.arange(n_mutation) * (min_dist + 1)
    else:
        spacing_offsets = np.arange(n_mutation) * (min_dist + 1)
        dividers = np.sort(np.random.randint(0, remaining_space + 1, size=n_mutation))
        positions = dividers + spacing_offsets

    # Handle forced position
    if force_position is not None:
        if force_position < 0 or force_position >= mut_len:
            raise ValueError(
                f"force_position {force_position} out of bounds [0, {mut_len})"
            )

        # Remove positions within min_dist from force_position
        distances = np.abs(positions - force_position)
        valid_mask = distances > min_dist
        positions = positions[valid_mask]

        # Insert force_position and sort
        positions = np.sort(np.append(positions, force_position))

    return positions


def sample_until_position_coverage(
    mut_len, min_dist, n_mutation, target_coverage, position_counts=None
):
    """
    Keep sampling mutation positions until each position appears at least target_coverage times.

    Returns:
        list: List of position arrays
    """
    # Track how many times each position appears
    if position_counts is None:
        position_counts = np.zeros(mut_len, dtype=int)

    sequences = []

    # Track current minimum coverage
    current_min_coverage = 0

    while current_min_coverage < target_coverage:

        # always sample position with minimum coverage
        min_cov = position_counts.min()
        min_positions = np.where(position_counts == min_cov)[0]
        force_pos = np.random.choice(min_positions)

        positions = sample_mutation_positions_forced(
            mut_len, min_dist, n_mutation, force_position=force_pos
        )

        # Update counts (ensure positions are integers)
        positions = positions.astype(int)
        position_counts[positions] += 1
        current_min_coverage = position_counts.min()
        sequences.append(positions)

    return sequences, position_counts


def mutation_design_matrix_uniform_force(
    mut_len, mut_reps, min_dist=50, n_mutation=None
):
    """Design mutation sequences using sequential uniform sampling with forced position coverage.

    Algorithm:
    - Always forces one position with lowest coverage per sequence
    - Continues until all positions reach target coverage

    Args:
        mut_len (int): Length of sequence to mutate
        mut_reps (int): Number of replicates needed for each mutation
        min_dist (int): Minimum distance between mutations
        n_mutation (int): Number of mutations per sequence

    Returns:
        list: List of mutation sequences (coo_matrix)
    """
    t0 = time.time()

    if min_dist <= 0:
        raise ValueError(f"min_dist must be positive, got {min_dist}")

    if n_mutation is None:
        n_mutation = int(0.75 * mut_len // min_dist)

    if n_mutation <= 0:
        raise ValueError(
            f"n_mutation must be positive, got {n_mutation}. Check mut_len and min_dist values."
        )

    if n_mutation > int(mut_len // min_dist):
        raise ValueError(
            f"n_mutation {n_mutation} is greater than the maximum number of mutations that can be placed in length {mut_len} with min_dist {min_dist}"
        )

    sequence_positions, position_counts = sample_until_position_coverage(
        mut_len, min_dist, n_mutation, target_coverage=mut_reps * 3
    )

    # Create 2D array for fully vectorized lookup: nu_array[pos, visit_num]
    max_count = position_counts.max()
    nu_array = np.zeros((mut_len, max_count), dtype=int)
    for pos in range(mut_len):
        if position_counts[pos] > 0:
            nu_array[pos, : position_counts[pos]] = np.random.permutation(
                position_counts[pos]
            )

    # Track visit counter for each position
    visit_counter = np.zeros(mut_len, dtype=int)
    sequences = []

    for positions in sequence_positions:
        # Fully vectorized nucleotide assignment
        visit_itr = visit_counter[positions]
        nut_values = nu_array[positions, visit_itr] % 3

        # Create sparse sequence
        current_seq = np.zeros((mut_len, 3), dtype=bool)
        current_seq[positions, nut_values] = True

        # Update visit counter
        visit_counter[positions] += 1

        sequences.append(coo_matrix(current_seq))

    X_mut3 = sequences

    print(f"Design time: {time.time() - t0:.3f} seconds")
    print_mutation_design_stats(X_mut3, method_name="design_uniform_force")

    return X_mut3


def print_mutation_design_stats(X_mut3, method_name=""):
    """Print comprehensive statistics for a mutation design matrix.

    Args:
        X_mut3 (list): List of scipy.sparse.csr_matrix objects, each representing a mutation sequence
        method_name (str): Name of the design method for the header
    """
    print(f"\nMutation Design Statistics for {method_name}")
    print("=" * 50)

    # Basic dimensions
    n_seqs = len(X_mut3)
    print(f"Total sequences generated: {n_seqs}")

    # Mutations per sequence
    muts_per_seq = np.array([seq.nnz for seq in X_mut3])
    print(f"\nMutations per sequence:")
    print(f"  Min: {muts_per_seq.min()}")
    print(f"  Max: {muts_per_seq.max()}")
    print(f"  Mean: {muts_per_seq.mean():.2f}")

    # Distance between mutations
    min_dist = float("inf")
    max_dist = 0
    all_distances = []

    for seq in X_mut3:
        mut_positions = seq.row
        if len(mut_positions) > 1:
            distances = mut_positions[1:] - mut_positions[:-1]
            if len(distances) > 0:
                min_dist = min(min_dist, distances.min())
                max_dist = max(max_dist, distances.max())
                all_distances.extend(distances)

    if min_dist != float("inf"):
        print(f"\nDistance between mutations:")
        print(f"  Min: {min_dist}")
        print(f"  Max: {max_dist}")
        print(f"  Mean: {np.mean(all_distances):.2f}")

    # Mutation coverage
    mut_counts = defaultdict(int)
    for seq in X_mut3:
        for r, c in zip(seq.row, seq.col):
            mut_counts[(r, c)] += 1
    mut_counts = np.array(list(mut_counts.values()))

    print(f"\nMutation repeats:")
    print(f"  Min: {mut_counts.min()}")
    print(f"  Max: {mut_counts.max()}")
    print(f"  Mean: {mut_counts.mean():.2f}")

    print("=" * 50)
