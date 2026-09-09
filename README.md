# Multi-ISM: Multi-mutation In Silico Saturation Mutagenesis

A Python package for performing multi-mutation in silico saturation mutagenesis of gene sequences using deep learning models with active learning.

## Overview

Multi-ISM provides tools for designing and analyzing mutations in genomic sequences using deep learning models, particularly the Baskerville framework. It implements iterative active learning to efficiently explore mutation space by:
1. Designing mutations and running inference
2. Estimating effect coefficients via elastic net regression
3. Selecting top variants for the next iteration

## Installation

### Prerequisites

- Python ≥ 3.8
- SLURM cluster with GPU nodes (nvidia_geforce_rtx_4090)
- Conda environment manager

### Required Dependencies

```bash
# Core packages
pip install numpy pandas scipy scikit-learn h5py torch
pip install pybedtools pysam

# Baskerville (for deep learning models)
# Follow installation instructions from the Baskerville repository

# SLURM job submission
pip install slurmrunner
```

### Installation from Source

```bash
# Clone the repository
git clone <repository-url>
cd mism_dev

# Install in development mode
pip install -e .
```

## Repository Structure

```
mism_dev/
├── mism/                           # Core Python package
│   ├── __init__.py
│   └── core/                       # Core modules
│       ├── __init__.py
│       ├── mutation_designer.py   # Mutation design logic
│       ├── elasticnet_cpu.py      # Elastic net regression
│       └── mism_utils.py          # Utility functions
│
├── scripts/                        # Pipeline scripts
│   ├── workflow/                   # Top-level orchestration
│   │   └── run_active_hybrid.py    # Per-gene: GPU+CPU SLURM queues
│   │
│   ├── design/                     # Pipeline support scripts
│   │   ├── ism_design.py           # Design mutations + run inference
│   │   ├── ism_regressor.py        # Coefficient estimation
│   │   ├── select_variants.py      # Variant selection for itr2+
│   │   └── extract_gene_positions.R
│   │
│   ├── inference/                  # Vanilla-ISM baselines / prediction
│   │   ├── ism_vanilla.py          # Vanilla ISM baseline
│   │   ├── ism_vanilla_select.py   # Evaluate selected variants
│   │   ├── ism_vanilla_*_background.py  # With background mutations
│   │   ├── gene_pred.py            # Gene-level prediction
│   │   └── gene_pred_haplotypes.py
│   │
│   ├── evals/                      # Analysis and visualization
│   │   ├── eval_ism.py             # Main evaluation
│   │   ├── eval_ism_10k.py         # With vanilla comparison
│   │   ├── eval_ism_logo.py        # Logo visualization
│   │   └── feature_mean.py
│   │
│   └── tune/                       # Hyperparameter tuning
│       ├── elasticnet.py
│       └── elasticnet_tune.py
│
├── tests/                          # Tests and examples
│   ├── test_utils.py               # Unit tests for mism.core helpers
│   ├── test_designer/              # MutationDesigner benchmark + report
│   └── example_active_ism.sh       # Example workflow invocation
├── pyproject.toml                  # Package configuration
└── README.md                       # This file
```

## Workflow Overview

### Active Learning Pipeline

The main workflow (`run_active_hybrid.py`) runs iterative active learning:

**Iteration 1:**
1. **Design** (GPU): Generate initial mutations and run model inference
2. **Regression** (CPU): Estimate effect coefficients using elastic net

**Iterations 2+:**
1. **Select** (GPU): Choose top variants based on previous coefficients, , run single-variant ISM
2. **Design** (GPU): Generate new mutations excluding selected positions
3. **Regression** (CPU): Update coefficients with cumulative data

## Usage

### Per-gene pipeline: run_active_hybrid.py

Orchestrates the multi-job workflow for a single gene, using GPU nodes for
inference and CPU nodes for regression.

#### Basic Example

```bash
python scripts/workflow/run_active_hybrid.py \
    --outdir active_prune \
    --gtf /path/to/gene.gtf \
    --initial_n 20 \
    --itr_n 34 \
    --prune_pos 200 \
    --n_iter 5 \
    --n_cpu 100 \
    --hydra_env hydra \
    --script_dir ~/programs/source/python_packages/mism_dev/scripts \
    --genome /path/to/genome.fa \
    --targets /path/to/targets.txt \
    --params /path/to/params.json \
    --model /path/to/model.pth \
    --target_subset /path/to/target_subset.txt \
    --celltype K562
```

#### Full Example with All Options

```bash
python scripts/workflow/run_active_hybrid.py \
    --outdir active_prune \
    --gtf /path/to/gene.gtf \
    --warm \
    --initial_n 20 \
    --itr_n 34 \
    --prune_pos 200 \
    --k_background 4 \
    --n_iter 5 \
    --n_cpu 100 \
    --mut_len 500000 \
    --mut_distance_min 50 \
    --hydra_env hydra \
    --script_dir /path/to/mism_dev/scripts \
    --genome /path/to/genome.fa \
    --targets /path/to/targets.txt \
    --params /path/to/params.json \
    --model /path/to/model.pth \
    --target_subset /path/to/target_subset.txt \
    --vanilla_path /path/to/vanilla_ism/ism_out \
    --celltype K562
```

#### Key Parameters

**Iteration Control:**
- `--initial_n N`: Mutation replicates for iteration 1 (default: 20)
- `--itr_n N`: Mutation replicates for iterations 2+ (default: 40)
- `--prune_pos N`: Top positions to select each iteration (default: 200)
- `--n_iter N`: Number of iterations to run (default: 5)

**Mutation Parameters:**
- `--mut_len N`: Length of sequence to mutate (default: 500000)
- `--mut_distance_min N`: Minimum distance between mutations (default: 50)
- `--k_background N`: Background sequences for evaluation (default: 0)

**Resources:**
- `--n_cpu N`: CPU cores for regression (default: 64)
- `--hydra_env ENV`: Conda environment name

**Advanced Options:**
- `--warm`: Enable warm start (use previous coefficients as initial values)
- `--continue`: Resume from earliest incomplete iteration
- `--cleanup`: Remove all iteration folders except the last after completion
- `--vanilla_path PATH`: Path to vanilla ISM for comparison
- `--exclude_gpu_nodes LIST`: Comma-separated GPU nodes to avoid
- `--exclude_cpu_nodes LIST`: Comma-separated CPU nodes to avoid

**Required Paths:**
- `--gtf FILE`: Gene annotation GTF file
- `--genome FILE`: Genome FASTA file
- `--targets FILE`: Model targets file
- `--params FILE`: Model parameters JSON
- `--model FILE`: Trained model checkpoint
- `--target_subset FILE`: Target subset for analysis

## Resource Requirements

Estimates below reflect the reference CTCF setup (`--mut_len 500000`,
`--initial_n 20`, `--itr_n 34`, `--prune_pos 200`, `--k_background 4`,
`--n_iter 5`, `--n_cpu 100`) and the SLURM requests hard-coded in
`run_active_hybrid.py`. Treat them as order-of-magnitude guidance, not measured
peaks.

### GPU — forward passes (design & select_variants jobs)

- **Request:** 1× RTX 4090 (24 GB VRAM), 1 CPU, **20 GB system RAM** (fixed,
  independent of `mut_len`).
- Inference runs one mutated sequence at a time (batch size 1) in `bfloat16`
  over the model's fixed context window, so VRAM is dominated by the model's
  activations for a **single** input — not by the number of design sequences
  (those are streamed through). A 24 GB 4090 is comfortably sufficient; the same
  request works for every iteration because per-forward-pass cost does not grow.
- The `--mut_len` only changes *how many* forward passes run (wall-time), not the
  per-pass VRAM.

### CPU — ElasticNet / lasso regression (regress job)

- **Request:** `--n_cpu` cores (100), 2-day wall limit, and **RAM that scales
  with the iteration number**: `run_active_hybrid.py` requests
  `mem = 60 GB × itr`.

  | Iteration | Requested RAM (total, all targets) |
  |---|---|
  | itr1 | 60 GB |
  | itr2 | 120 GB |
  | itr3 | 180 GB |
  | itr4 | 240 GB |
  | itr5 | 300 GB |

- These numbers are the **total** memory for the whole regression job — all
  targets are solved together against one shared design matrix, *not* per target.
- **Why it grows:** each iteration stacks its new design matrix onto the
  cumulative sparse `X_mut` (CSR) used for regression, so `n_samples` — and the
  solver's residual/working set — grow roughly linearly across iterations. At
  `mut_len = 500k` the feature dimension is `3 × 500,000 = 1.5M`, and the
  later-iteration cumulative matrix is what drives the multi-hundred-GB request.
  Later iterations are the memory bottleneck; **budget for the itr5 peak
  (~300 GB), not itr1.**
- The single sparse `X_mut` is shared across all targets, so the number of
  targets in `--target_subset` adds only the per-target coefficient columns
  (`n_features × n_targets`), which is minor next to `X`. The primary drivers of
  total RAM are the cumulative `X` (iteration count, `mut_len`, `--itr_n`,
  `--prune_pos`, `--k_background`).

## Output Structure

```
outdir/
├── itr1/
│   ├── out/
│   │   ├── coefs.pt                    # Coefficient tensor
│   │   ├── coefs_annot.csv             # Coefficient annotations
│   │   ├── matrices_logSUM.h5          # Design matrix
│   │   └── ism_tracks_comparison.pdf   # Evaluation plots
│   ├── design.sh                        # SLURM script
│   └── regress.sh
│
├── itr2/
│   ├── select_variants/
│   │   ├── variants.csv                 # Selected variants
│   │   ├── variants.bed                 # BED format
│   │   └── matrices_logSED.h5           # Variant matrix
│   ├── out/
│   └── ...
│
└── ...
```

## Features

- **Active Learning**: Iterative variant selection and mutation design
- **Multi-Mutation ISM**: Design complex multi-mutation sequences
- **Deep Learning Integration**: Works with Baskerville PyTorch models
- **Flexible Orchestration**: Multi-job or single-job execution modes
- **Comprehensive Evaluation**: Comparison with vanilla ISM baselines

## Citation

If you use MISM in your research, please cite:

```
[Citation information to be added]
```

## License

Licensed under the Apache License, Version 2.0. See the original script headers for full license text.

## Contact

For questions and issues, please open an issue on the repository or contact the maintainers. 