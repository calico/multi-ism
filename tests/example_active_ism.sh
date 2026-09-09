#!/bin/bash
# ---------------------------------------------------------------------------
# Example: iterative active-learning m-ISM for a single gene.
#
# This is the high-level workflow. `run_active_hybrid.py` orchestrates
# the full loop per gene (design -> Baskerville inference -> ElasticNet
# regression -> variant selection/pruning -> eval), submitting GPU/CPU SLURM
# jobs for each iteration.
# ---------------------------------------------------------------------------

SCRIPT_DIR=/path/to/mism_dev/scripts
DATA=/path/to/mism_data

python "$SCRIPT_DIR"/workflow/run_active_hybrid.py \
    --outdir active_prune \
    --gtf gene.gtf \
    --slurm_id CTCF \
    --continue \
    --warm \
    --cleanup \
    --initial_n 20 \
    --itr_n 34 \
    --n_iter 5 \
    --prune_pos 200 \
    --k_background 4 \
    --mut_len 500000 \
    --mut_distance_min 50 \
    --n_cpu 100 \
    --hydra_env hydra \
    --script_dir "$SCRIPT_DIR" \
    --genome /path/to/genome.fa \
    --params "$DATA"/model_flashzoi/pretrained_f0c0_params.json \
    --model "$DATA"/model_flashzoi/f0c0/train/model_best.pth \
    --targets "$DATA"/model_flashzoi/targets_human.txt \
    --target_subset "$DATA"/targets_flashzoi/targets_rna.txt \
    --celltype K562 \
    --exclude_gpu_nodes gpu-node-01,gpu-node-02
