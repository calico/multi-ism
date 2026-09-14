#!/usr/bin/env python3
"""
Active learning iterative m-ISM runner - Single SLURM job version

This script generates a single SLURM script that runs all iterations sequentially
on one GPU with 32 cores. Unlike run_active_hybrid.py which submits separate jobs
for each step, this creates one monolithic script.

- itr1: design -> regress
- itr2+: select_variants -> design -> regress

Usage:
    python run_active_gpu_only.py --initial_n <n> --itr_n <n> --prune_pos <n>
"""

import argparse
import os
import sys
import logging
import subprocess
from pathlib import Path
import shlex

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_select_cmd(
    itr: int,
    itr_dir: str,
    n_pos: int,
    hydra_env: str,
    script_dir: str,
    genome: str,
    targets: str,
    params: str,
    model: str,
    gtf: str,
    k_background: int = 0,
) -> str:
    """Generate command for variant selection step."""
    prev_itr = itr - 1

    # Choose script based on k_background value
    if k_background > 0:
        ism_script = f"{script_dir}/inference/ism_vanilla_select_mutate_background.py"
        k_bg_arg = f"--k_background {k_background} \\\n    "
    else:
        ism_script = f"{script_dir}/inference/ism_vanilla_select.py"
        k_bg_arg = ""

    return f"""# === Iteration {itr}: Variant Selection ===
cd {itr_dir}

python {script_dir}/design/select_variants.py \\
    --ism_scores ../itr{prev_itr}/out/coefs.pt \\
    --ism_annot ../itr{prev_itr}/out/coefs_annot.csv \\
    --n_pos {n_pos}

python {ism_script} \\
    -t {targets} \\
    -m bfloat16 \\
    -f {genome} \\
    -o select_variants \\
    --variants_csv select_variants/variants.csv \\
    {k_bg_arg}{params} \\
    {model} \\
    {gtf}
"""


def create_design_cmd(
    itr: int,
    itr_dir: str,
    mut_len: int,
    mut_reps: int,
    mut_distance_min: int,
    hydra_env: str,
    script_dir: str,
    genome: str,
    targets: str,
    params: str,
    model: str,
    gtf: str,
    bed_files: str = None,
) -> str:
    """Generate command for design step using ism_design.py, optionally with bed exclusion."""
    bed_args = ""
    if bed_files:
        bed_args = f"--bed_files {bed_files} \\\n    --bed_exclude \\\n    "

    return f"""# === Iteration {itr}: Design ===
cd {itr_dir}

python {script_dir}/design/ism_design.py \\
    --rc random \\
    -m bfloat16 \\
    --design bins \\
    -l {mut_len} -n {mut_reps} -d {mut_distance_min} \\
    -o out \\
    {bed_args}-f {genome} \\
    -t {targets} \\
    {params} \\
    {model} \\
    {gtf}
"""


def create_regress_cmd(
    itr: int,
    itr_dir: str,
    n_cpu: int,
    hydra_env: str,
    script_dir: str,
    target_subset: str,
    vanilla_path: str,
    celltype: str,
    gtf: str = None,
    warm_start: bool = False,
) -> str:
    """Generate command for regression and evaluation, handling all iterations."""
    # Use default GTF path if not provided
    if gtf is None:
        gtf = "/path/to/gencode_basic_protein.gtf"

    # Build matrix files string for all iterations up to current
    if itr == 1:
        matrix_files = "out/matrices_logSUM.h5"
    else:
        matrix_files = ",".join(
            [
                (
                    f"../itr{j}/out/matrices_logSUM.h5"
                    if j < itr
                    else "out/matrices_logSUM.h5"
                )
                for j in range(1, itr + 1)
            ]
        )

    # Build select files for itr2+
    select_args = ""
    if itr >= 2:
        select_sed_files = []
        select_annot_files = []
        for j in range(2, itr + 1):
            if j == itr:
                select_sed_files.append("select_variants/matrices_logSED.h5")
                select_annot_files.append("select_variants/variants_annotation.tsv")
            else:
                select_sed_files.append(f"../itr{j}/select_variants/matrices_logSED.h5")
                select_annot_files.append(
                    f"../itr{j}/select_variants/variants_annotation.tsv"
                )

        select_sed_arg = ",".join(select_sed_files)
        select_annot_arg = ",".join(select_annot_files)
        select_args = f"--ism_select_sed_files {select_sed_arg} \\\n    --ism_select_annot_files {select_annot_arg} \\\n    "

    # Add warm start option for itr2+ (use previous iteration coefficients)
    warm_args = ""
    if warm_start and itr >= 2:
        prev_itr = itr - 1
        warm_args = f"--prev_coefs ../itr{prev_itr}/out/coefs.pt \\\n    "

    # Build evaluation command
    celltype_arg = shlex.quote(celltype)

    if vanilla_path:
        eval_cmd = f"""
# Evaluation
python {script_dir}/evals/eval_ism_10k.py \\
    --ism_scores out/coefs.pt \\
    --ism_annot out/coefs_annot.csv \\
    --ism_target out/targets_subset.csv \\
    --gtf {gtf} \\
    --vanilla_path {vanilla_path} \\
    --celltype {celltype_arg} \\
    --outdir out
"""
    else:
        eval_cmd = f"""
# Evaluation
python {script_dir}/evals/eval_ism.py \\
    --ism_scores out/coefs.pt \\
    --ism_annot out/coefs_annot.csv \\
    --ism_target out/targets_subset.csv \\
    --gtf {gtf} \\
    --celltype {celltype_arg} \\
    --outdir out
"""

    return f"""# === Iteration {itr}: Regression ===
cd {itr_dir}

python {script_dir}/design/ism_regressor.py \\
    --mut_XY_files "{matrix_files}" \\
    --target_strand_file out/targets_strand.txt \\
    --target_subset_file {target_subset} \\
    --coef_annot_file out/x_mut_annotation.tsv \\
    {select_args}{warm_args}--njobs {n_cpu} \\
    --outdir out
{eval_cmd}
"""


def check_progress(base_dir: str, outdir: str, n_iter: int) -> int:
    """Check which iteration to start from based on existing output files.

    Returns:
        The iteration number to start from (1 to n_iter+1).
        Returns n_iter+1 if all iterations are complete.
    """
    outdir_path = Path(base_dir) / outdir

    # Check if output directory exists
    if not outdir_path.exists():
        logger.info(
            f"Output directory {outdir_path} does not exist - starting from iteration 1"
        )
        return 1

    # Find the earliest incomplete iteration
    for itr in range(1, n_iter + 1):
        itr_dir = outdir_path / f"itr{itr}"
        result_file = itr_dir / "out" / "ism_tracks_comparison.pdf"

        if not result_file.exists():
            logger.info(f"Iteration {itr} incomplete (missing {result_file})")
            return itr
        else:
            logger.info(f"Iteration {itr} complete (found {result_file})")

    logger.info(f"All {n_iter} iterations already completed")
    return n_iter + 1


def build_sbatch_header(
    job_name: str,
    out_file: str,
    queue: str,
    cpu: int,
    mem: int,
    time_limit: str,
    gpu: int,
) -> str:
    """Build the #SBATCH directive block for a self-contained sbatch script.

    For GPU jobs the queue name doubles as the gres type, e.g.
    ``nvidia_geforce_rtx_4090`` -> partition ``gpu`` + ``--gres=gpu:<type>:<n>``.
    """
    lines = ["#!/bin/bash", ""]
    if gpu > 0:
        if queue in ("", "gpu"):
            partition, gres = "gpu", "--gres=gpu"
        else:
            partition, gres = "gpu", f"--gres=gpu:{queue}"
        lines.append(f"#SBATCH -p {partition}")
        lines.append(f"#SBATCH {gres}:{gpu}")
    else:
        lines.append(f"#SBATCH -p {queue}")
    lines.append("#SBATCH -n 1")
    lines.append(f"#SBATCH -c {cpu}")
    if job_name:
        lines.append(f"#SBATCH -J {job_name}")
    if out_file:
        lines.append(f"#SBATCH -o {out_file}")
    if mem:
        lines.append(f"#SBATCH --mem {mem}")
    if time_limit:
        lines.append(f"#SBATCH --time {time_limit}")
    return "\n".join(lines)


def build_single_slurm_script(
    base_dir: str,
    outdir: str,
    n_iter: int,
    mut_len: int,
    initial_n: int,
    itr_n: int,
    mut_distance_min: int,
    n_cpu: int,
    prune_pos: int,
    hydra_env: str,
    script_dir: str,
    genome: str,
    targets: str,
    params: str,
    model: str,
    gtf: str,
    target_subset: str,
    vanilla_path: str,
    celltype: str,
    warm_start: bool = False,
    k_background: int = 0,
    start_itr: int = 1,
    job_name: str = "ism_all",
    out_file: str = None,
    queue: str = "nvidia_geforce_rtx_4090",
    mem: int = 60000,
    time_limit: str = "7-0:0:0",
    gpu: int = 1,
) -> str:
    """Build a single, self-contained sbatch script that runs iterations sequentially.

    The generated script embeds its own #SBATCH directives, so it can be submitted
    directly with ``sbatch`` without any external launcher.

    Args:
        start_itr: The iteration to start from (default 1). Used for --continue mode.
    """

    commands = []
    outdir_path = Path(base_dir) / outdir

    # Header
    if start_itr == 1:
        itr_range_str = f"all {n_iter}"
    else:
        itr_range_str = f"iterations {start_itr} to {n_iter}"

    sbatch_header = build_sbatch_header(
        job_name, out_file, queue, n_cpu, mem, time_limit, gpu
    )
    commands.append(
        f"""{sbatch_header}

# Exit immediately if any command fails
set -e
set -o pipefail

# Single SLURM script for {itr_range_str} iterations of active m-ISM
# Generated by run_active_gpu_only.py

source ~/.bashrc
conda activate {hydra_env}

echo "Starting active m-ISM pipeline at $(date)"
echo "Working directory: {base_dir}"
echo "Output directory: {outdir}"
echo "Running iterations: {start_itr} to {n_iter}"
echo ""
"""
    )

    # Iteration 1: design -> regress (only if start_itr == 1)
    if start_itr == 1:
        itr = 1
        itr_dir = outdir_path / f"itr{itr}"
        commands.append(f"\necho '{'='*60}'")
        commands.append(f"echo 'ITERATION {itr}: Design + Regression'")
        commands.append(f"echo '{'='*60}'")
        commands.append(f"mkdir -p {itr_dir}\n")

        commands.append(
            create_design_cmd(
                itr,
                str(itr_dir),
                mut_len,
                initial_n,
                mut_distance_min,
                hydra_env,
                script_dir,
                genome,
                targets,
                params,
                model,
                gtf,
            )
        )

        commands.append(
            create_regress_cmd(
                itr,
                str(itr_dir),
                n_cpu,
                hydra_env,
                script_dir,
                target_subset,
                vanilla_path,
                celltype,
                gtf=None,
                warm_start=False,
            )
        )

    # Iterations 2+: select -> design -> regress (start from max(2, start_itr))
    for itr in range(max(2, start_itr), n_iter + 1):
        itr_dir = outdir_path / f"itr{itr}"
        commands.append(f"\necho '{'='*60}'")
        commands.append(f"echo 'ITERATION {itr}: Select + Design + Regression'")
        commands.append(f"echo '{'='*60}'")
        commands.append(f"mkdir -p {itr_dir}\n")

        # Select
        commands.append(
            create_select_cmd(
                itr,
                str(itr_dir),
                prune_pos,
                hydra_env,
                script_dir,
                genome,
                targets,
                params,
                model,
                gtf,
                k_background,
            )
        )

        # Design with bed exclusion
        if itr == 2:
            bed_files = "select_variants/variants.bed"
        else:
            # itr3+ needs all previous select_variants
            bed_list = []
            for i in range(2, itr + 1):
                if i == itr:
                    bed_list.append("select_variants/variants.bed")
                else:
                    bed_list.append(f"../itr{i}/select_variants/variants.bed")
            bed_files = ",".join(bed_list)

        commands.append(
            create_design_cmd(
                itr,
                str(itr_dir),
                mut_len,
                itr_n,
                mut_distance_min,
                hydra_env,
                script_dir,
                genome,
                targets,
                params,
                model,
                gtf,
                bed_files,
            )
        )

        # Regress
        commands.append(
            create_regress_cmd(
                itr,
                str(itr_dir),
                n_cpu,
                hydra_env,
                script_dir,
                target_subset,
                vanilla_path,
                celltype,
                gtf=None,
                warm_start=warm_start,
            )
        )

    # Footer
    if start_itr == 1:
        completion_msg = f"All {n_iter} iterations completed successfully at $(date)!"
    else:
        completion_msg = (
            f"Iterations {start_itr} to {n_iter} completed successfully at $(date)!"
        )

    commands.append(
        f"""
echo ""
echo '{'='*60}'
echo "{completion_msg}"
echo '{'='*60}'
"""
    )

    return "\n".join(commands)


def main():
    """Main entry point for the active learning script."""
    parser = argparse.ArgumentParser(
        description="Single-job iterative m-ISM active learning with variant selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path arguments
    parser.add_argument(
        "--outdir",
        type=str,
        default="active_prune",
        help="Output directory name within current directory for iterations [Default: %(default)s]",
    )

    parser.add_argument(
        "--gtf",
        type=str,
        default="/path/to/gene.gtf",
        help="GTF file with gene annotation",
    )

    # New control parameters
    parser.add_argument(
        "--initial_n",
        type=int,
        default=20,
        help="Number of mutation replicates (-n) for itr1 design",
    )

    parser.add_argument(
        "--itr_n",
        type=int,
        default=40,
        help="Number of mutation replicates (-n) for itr2+ design",
    )

    parser.add_argument(
        "--prune_pos",
        type=int,
        default=200,
        help="Number of top positions (--n_pos) for variant selection",
    )

    parser.add_argument(
        "--k_background",
        type=int,
        default=0,
        help="Number of background sequences for variant evaluation. 0=use reference only (ism_vanilla_select.py), >0=use mutated backgrounds (ism_vanilla_select_mutate_background.py) [Default: %(default)s]",
    )

    # Run configuration arguments
    parser.add_argument(
        "--n_iter", type=int, default=5, help="Number of iterations to run (1-3)"
    )

    parser.add_argument(
        "--n_cpu",
        type=int,
        default=32,
        help="Number of CPU cores for regression [Default: 32 for single-job version]",
    )

    parser.add_argument(
        "-l",
        "--mut_len",
        type=int,
        default=500000,
        help="Length of center sequence to mutate",
    )

    parser.add_argument(
        "-d",
        "--mut_distance_min",
        type=int,
        default=50,
        help="Minimum allowed distance between mutations",
    )

    parser.add_argument(
        "--hydra_env", type=str, default="hydra", help="Conda environment name"
    )

    parser.add_argument(
        "--script_dir",
        type=str,
        default="~/programs/source/python_packages/mism_dev/scripts",
        help="Directory containing scripts",
    )

    parser.add_argument(
        "--genome", type=str, default="/path/to/genome.fa", help="Genome FASTA file"
    )

    parser.add_argument(
        "--targets", type=str, default="/path/to/targets_human.txt", help="Targets file"
    )

    parser.add_argument(
        "--params",
        type=str,
        default="/path/to/params.json",
        help="Model parameters JSON file",
    )

    parser.add_argument(
        "--model", type=str, default="/path/to/model.pth", help="Trained model file"
    )

    parser.add_argument(
        "--target_subset",
        type=str,
        default="/path/to/target_subset.txt",
        help="Target subset file",
    )

    parser.add_argument(
        "--vanilla_path",
        type=str,
        default=None,
        help="Path to vanilla ISM output for evaluation. If None, run eval_ism.py without vanilla-ISM.",
    )

    parser.add_argument(
        "--celltype",
        type=str,
        default="K562",
        help="Cell type for evaluation [Default: %(default)s]",
    )

    parser.add_argument(
        "--warm",
        action="store_true",
        help="Enable warm start: use previous iteration coefficients as initial values for itr2+ regression",
    )

    parser.add_argument(
        "--slurm_id",
        type=str,
        default="ism",
        help="Prefix for SLURM job name [Default: %(default)s]",
    )

    parser.add_argument(
        "--time",
        type=str,
        default="7-0:0:0",
        help="SLURM time limit [Default: %(default)s]",
    )

    parser.add_argument(
        "--mem", type=int, default=60000, help="Memory in MB [Default: %(default)s]"
    )

    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue from the earliest incomplete iteration (checks for out/ism_tracks_comparison.pdf)",
    )

    args = parser.parse_args()

    # Use current working directory as base
    base_dir = str(Path.cwd())

    # Create subdirectory for iterations
    outdir_path = Path(base_dir) / args.outdir
    outdir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Working directory: {base_dir}")
    logger.info(f"Output directory: {args.outdir}")
    logger.info(f"Full iteration path: {outdir_path}")
    logger.info(
        f"Configuration: n_iter={args.n_iter}, initial_n={args.initial_n}, "
        f"itr_n={args.itr_n}, prune_pos={args.prune_pos}"
    )
    logger.info(
        f"Mutation params: mut_len={args.mut_len}, mut_distance_min={args.mut_distance_min}"
    )
    logger.info(f"Resources: n_cpu={args.n_cpu}, mem={args.mem}MB, time={args.time}")

    # Determine starting iteration
    if args.continue_run:
        start_itr = check_progress(base_dir, args.outdir, args.n_iter)
        if start_itr > args.n_iter:
            logger.info("Nothing to do - all iterations already complete")
            return
        logger.info(f"Continue mode: starting from iteration {start_itr}")
    else:
        start_itr = 1
        logger.info("Normal mode: running all iterations from the beginning")

    # Build the single SLURM script
    logger.info("Building single SLURM script...")
    script_path = outdir_path / "run_all_iterations.sh"
    log_path = outdir_path / "run_all_iterations.%j.log"

    script_content = build_single_slurm_script(
        base_dir,
        args.outdir,
        args.n_iter,
        args.mut_len,
        args.initial_n,
        args.itr_n,
        args.mut_distance_min,
        args.n_cpu,
        args.prune_pos,
        args.hydra_env,
        args.script_dir,
        args.genome,
        args.targets,
        args.params,
        args.model,
        args.gtf,
        args.target_subset,
        args.vanilla_path,
        args.celltype,
        args.warm,
        args.k_background,
        start_itr,
        job_name=f"{args.slurm_id}_all",
        out_file=str(log_path),
        queue="nvidia_geforce_rtx_4090",
        mem=args.mem,
        time_limit=args.time,
        gpu=1,
    )

    # Write the self-contained sbatch script
    with open(script_path, "w") as f:
        f.write(script_content)

    logger.info(f"SLURM script written to: {script_path}")

    # Submit the job with sbatch
    result = subprocess.run(
        ["sbatch", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Parse job ID from output like "Submitted batch job 13861989"
    job_id = result.stdout.strip().split()[-1]

    logger.info(f"\n{'='*60}")
    logger.info(f"Submitted single SLURM job: {job_id}")
    logger.info(f"Job name: {args.slurm_id}_all")
    logger.info(f"Queue: nvidia_geforce_rtx_4090")
    logger.info(f"Resources: {args.n_cpu} CPUs, 1 GPU, {args.mem}MB RAM")
    logger.info(f"Time limit: {args.time}")
    logger.info(f"Log file: {log_path.name}")
    logger.info(f"Script file: {script_path}")
    logger.info(f"\nMonitor with: squeue -j {job_id}")
    logger.info(f"Cancel with: scancel {job_id}")
    logger.info(
        f"View log: tail -f {log_path.parent}/{log_path.name.replace('%j', str(job_id))}"
    )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
