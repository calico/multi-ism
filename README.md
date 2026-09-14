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

Install the pinned dependency set with:

```bash
pip install -r requirements.txt
```

Some components require additional packages not covered by `requirements.txt`:

```bash
# Genomic interval / FASTA handling
pip install pybedtools

# Baskerville (for deep learning models)
# Will be available at: https://github.com/calico/baskerville-torch
```

#### Choosing a runner

The project ships two runners:

- **`run_active_gpu_only.py`** (recommended) — self-contained: it writes and
  submits its own `sbatch` script, running the full pipeline as a single job on
  one GPU node.
  - *Pros:* simple, with no extra dependencies.
  - *Cons:* the regression step is CPU-only and does not benefit from
    parallelization across nodes.

- **`run_active_hybrid.py`** — submits the design and selection steps as GPU
  jobs and the regression step as a separate CPU job.
  - *Pros:* more resource-efficient and generally faster.
  - *Cons:* tied to the HPC scheduler. In our setup it runs on SLURM and uses
    the custom `slurmrunner` package to submit individual jobs
    (`pip install slurmrunner`).

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
mism-public/
├── mism/                           # Core Python package
│   └── core/                       # Core modules
│       ├── mutation_designer.py    # Mutation design logic
│       ├── elasticnet_cpu.py       # Elastic net regression
│       └── mism_utils.py           # Utility functions
│
├── scripts/                        # Pipeline scripts
│   ├── workflow/                   # Top-level orchestration
│   │   ├── run_active_gpu_only.py  # Single self-contained sbatch job (recommended)
│   │   └── run_active_hybrid.py    # Multi-job GPU+CPU SLURM queues (needs slurmrunner)
│   │
│   ├── design/                     # Pipeline support scripts
│   │   ├── ism_design.py           # Design mutations + run inference
│   │   ├── ism_regressor.py        # Coefficient estimation
│   │   └── select_variants.py      # Variant selection for itr2+
│   │
│   ├── inference/                  # Vanilla-ISM baselines / prediction
│   │   ├── ism_vanilla.py          # Vanilla ISM baseline
│   │   ├── ism_vanilla_select.py   # Evaluate selected variants
│   │   ├── ism_vanilla_mutate_background.py
│   │   ├── ism_vanilla_select_mutate_background.py
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
├── data/                           # Reference tables / example inputs
│   ├── GATA1.gtf
│   ├── flashzoi_f0c0.json          # Example model params
│   ├── flashzoi_targets_all.txt
│   └── flashzoi_targets_rna.txt
│
├── tests/                          # Tests and examples
│   ├── test_utils.py               # Unit tests for mism.core helpers
│   ├── test_assemble_sparse_coefs.py
│   ├── test_designer/              # MutationDesigner benchmark + report
│   ├── test_predict/               # Gene-prediction golden test
│   └── example_active_ism.sh       # Example workflow invocation
│
├── requirements.txt                # Pinned dependencies
├── pyproject.toml                  # Package configuration
├── LICENSE.md
└── README.md                       # This file
```

## Workflow Overview

### Active Learning Pipeline

The main workflow (`run_active_gpu_only.py`) runs iterative active learning:

**Iteration 1:**
1. **Design** (GPU): Generate initial mutations and run model inference
2. **Regression** (CPU): Estimate effect coefficients using elastic net

**Iterations 2+:**
1. **Select** (GPU): Choose top variants based on previous coefficients, then run single-variant ISM
2. **Design** (GPU): Generate new mutations excluding selected positions
3. **Regression** (CPU): Update coefficients with cumulative data

## Usage

### Per-gene pipeline: run_active_gpu_only.py (recommended)

Runs the full iterative workflow for a single gene as **one self-contained
SLURM job** on a single GPU node.

#### Basic Example

```bash
python scripts/workflow/run_active_gpu_only.py \
    --outdir active_prune \
    --gtf /path/to/gene.gtf \
    --initial_n 20 \
    --itr_n 34 \
    --prune_pos 200 \
    --n_iter 5 \
    --n_cpu 32 \
    --hydra_env hydra \
    --script_dir /path/to/mism/scripts \
    --genome /path/to/genome.fa \
    --targets /path/to/targets.txt \
    --params /path/to/params.json \
    --model /path/to/model.pth \
    --target_subset /path/to/target_subset.txt \
    --celltype K562
```

#### Full Example with All Options

```bash
python scripts/workflow/run_active_gpu_only.py \
    --outdir active_prune \
    --gtf /path/to/gene.gtf \
    --warm \
    --initial_n 20 \
    --itr_n 34 \
    --prune_pos 200 \
    --k_background 4 \
    --n_iter 5 \
    --n_cpu 32 \
    --mut_len 500000 \
    --mut_distance_min 50 \
    --mem 60000 \
    --time 7-0:0:0 \
    --hydra_env hydra \
    --script_dir /path/to/mism/scripts \
    --genome /path/to/genome.fa \
    --targets /path/to/targets.txt \
    --params /path/to/params.json \
    --model /path/to/model.pth \
    --target_subset /path/to/target_subset.txt \
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

**Resources (single SLURM job):**
- `--n_cpu N`: CPU cores for regression (default: 32)
- `--mem MB`: System RAM for the job in MB (default: 60000)
- `--time D-H:M:S`: SLURM wall-time limit (default: 7-0:0:0)
- `--hydra_env ENV`: Conda environment name

**Advanced Options:**
- `--warm`: Enable warm start (use previous coefficients as initial values)
- `--continue`: Resume from earliest incomplete iteration
- `--vanilla_path PATH`: Path to vanilla ISM for comparison

**Required Paths:**
- `--gtf FILE`: Gene annotation GTF file
- `--genome FILE`: Genome FASTA file
- `--targets FILE`: Model targets file
- `--params FILE`: Model parameters JSON
- `--model FILE`: Trained model checkpoint
- `--target_subset FILE`: Target subset for analysis

### Alternative: run_active_hybrid.py (multi-job)

`run_active_hybrid.py` splits each step into separate GPU and CPU SLURM jobs for
better cluster utilization on large runs. It adds per-iteration memory scaling
and node-exclusion options (`--cleanup`, `--exclude_gpu_nodes`,
`--exclude_cpu_nodes`), and **requires** `slurmrunner`
(`pip install slurmrunner`). See the "Resource Requirements" section below for
its SLURM request profile.

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
  `mem < 10 GB × itr`.

  | Iteration | Requested RAM (total, all targets) |
  |---|---|
  | itr1 | < 10 GB |
  | itr2 | < 20 GB |
  | itr3 | < 30 GB |
  | itr4 | < 40 GB |
  | itr5 | < 50 GB |


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