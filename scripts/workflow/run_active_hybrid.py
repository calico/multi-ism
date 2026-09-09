#!/usr/bin/env python3
"""
Active learning iterative m-ISM runner with variant selection (itr1–itr3)

This script manages iterative runs of active m-ISM experiments using SLURM job submission.
- itr1: design -> regress
- itr2+: select_variants -> design -> regress

Usage:
    python run_active_hybrid.py --initial_n <n> --itr_n <n> --prune_pos <n>
"""

import argparse
import os
import sys
import logging
import time
import shutil
from pathlib import Path
from slurmrunner import Job
import shlex

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def wait_for_job(
    job: Job,
    check_interval: int = 30,
    max_none_checks: int = 3,
    initial_delay: int = 10,
):
    """Wait for a job to complete by checking its status periodically.

    Args:
        job: The SLURM job to wait for
        check_interval: Seconds between status checks
        max_none_checks: Number of consecutive None status checks before assuming completion
        initial_delay: Seconds to wait before first status check (gives SLURM time to register job)

    Raises:
        RuntimeError: If the job fails (status is not COMPLETED)
    """
    logger.info(f"Waiting for job {job.id} ({job.name}) to complete...")

    # Give SLURM time to register the job in its database (important when cluster is busy)
    time.sleep(initial_delay)

    none_count = 0

    while True:
        job.update_status()

        if job.status not in ["PENDING", "RUNNING"]:
            logger.info(f"Job {job.id} completed with status: {job.status}")

            # Handle None status: job may have completed and been purged from queue
            if job.status is None:
                none_count += 1
                if none_count < max_none_checks:
                    logger.warning(
                        f"Could not find job {job.id} in queue (attempt {none_count}/{max_none_checks}), will retry..."
                    )
                    time.sleep(check_interval)
                    continue
                else:
                    logger.warning(
                        f"Job {job.id} no longer in queue after {max_none_checks} checks. "
                        f"Assuming it completed. Please verify output files."
                    )
                    break
            elif job.status != "COMPLETED":
                logger.error(
                    f"Job {job.id} ({job.name}) failed with status: {job.status}"
                )
                raise RuntimeError(
                    f"Job {job.id} ({job.name}) failed with status: {job.status}"
                )
            break

        # Reset none_count if we get a valid status
        none_count = 0
        time.sleep(check_interval)


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

    return f"""source ~/.bashrc
conda activate {hydra_env}
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
    {gtf}"""


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

    return f"""source ~/.bashrc
conda activate {hydra_env}
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
    {gtf}"""


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
        eval_cmd = f"""# evaluation
python {script_dir}/evals/eval_ism_10k.py \\
    --ism_scores out/coefs.pt \\
    --ism_annot out/coefs_annot.csv \\
    --ism_target out/targets_subset.csv \\
    --gtf {gtf} \\
    --vanilla_path {vanilla_path} \\
    --celltype {celltype_arg} \\
    --outdir out"""
    else:
        eval_cmd = f"""# evaluation
python {script_dir}/evals/eval_ism.py \\
    --ism_scores out/coefs.pt \\
    --ism_annot out/coefs_annot.csv \\
    --ism_target out/targets_subset.csv \\
    --gtf {gtf} \\
    --celltype {celltype_arg} \\
    --outdir out"""

    return f"""source ~/.bashrc
conda activate {hydra_env}
cd {itr_dir}

# lasso regression
python {script_dir}/design/ism_regressor.py \\
    --mut_XY_files "{matrix_files}" \\
    --target_strand_file out/targets_strand.txt \\
    --target_subset_file {target_subset} \\
    --coef_annot_file out/x_mut_annotation.tsv \\
    {select_args}{warm_args}--njobs {n_cpu} \\
    --outdir out
{eval_cmd}"""


def run_iteration_1(
    base_dir: str,
    outdir: str,
    mut_len: int,
    initial_n: int,
    mut_distance_min: int,
    n_cpu: int,
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
    verbose: bool = True,
    slurm_id: str = "ism",
    exclude_gpu_nodes: str = None,
    exclude_cpu_nodes: str = None,
):
    """Run iteration 1: design -> regress."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting iteration 1")
    logger.info(f"{'='*60}")

    itr_dir = Path(base_dir) / outdir / "itr1"
    itr_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Design (GPU job)
    logger.info(f"Step 1/2: Running design for itr1 with n={initial_n}")
    design_cmd = create_design_cmd(
        1,
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
    gpu_sbatch_options = {"exclude": exclude_gpu_nodes} if exclude_gpu_nodes else {}
    cpu_sbatch_options = {"exclude": exclude_cpu_nodes} if exclude_cpu_nodes else {}
    design_job = Job(
        cmd=design_cmd,
        name=f"{slurm_id}_design_itr1",
        out_file=str(itr_dir / f"design.%j.log"),
        sb_file=str(itr_dir / "design.sh"),
        queue="nvidia_geforce_rtx_4090",
        cpu=1,
        mem=20000,
        gpu=1,
        sbatch_options=gpu_sbatch_options,
    )
    design_job.launch()
    if verbose:
        print(f"  Launched design job {design_job.id}")
    wait_for_job(design_job)

    # Step 2: Regression (CPU job)
    logger.info(f"Step 2/2: Running regression for itr1")
    regress_cmd = create_regress_cmd(
        1,
        str(itr_dir),
        n_cpu,
        hydra_env,
        script_dir,
        target_subset,
        vanilla_path,
        celltype,
    )
    regress_job = Job(
        cmd=regress_cmd,
        name=f"{slurm_id}_regress_itr1",
        out_file=str(itr_dir / f"regress.%j.log"),
        sb_file=str(itr_dir / "regress.sh"),
        queue="standard",
        cpu=n_cpu,
        mem=60000,
        time="2-0:0:0",
        sbatch_options=cpu_sbatch_options,
    )
    regress_job.launch()
    if verbose:
        print(f"  Launched regression job {regress_job.id}")
    wait_for_job(regress_job)

    logger.info(f"Iteration 1 completed successfully")


def run_iteration_n(
    itr: int,
    base_dir: str,
    outdir: str,
    mut_len: int,
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
    verbose: bool = True,
    warm_start: bool = False,
    k_background: int = 0,
    slurm_id: str = "ism",
    exclude_gpu_nodes: str = None,
    exclude_cpu_nodes: str = None,
):
    """Run iteration N (N>=2): select -> design -> regress."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting iteration {itr}")
    logger.info(f"{'='*60}")

    itr_dir = Path(base_dir) / outdir / f"itr{itr}"
    itr_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Select variants (GPU job)
    logger.info(
        f"Step 1/3: Running variant selection for itr{itr} with n_pos={prune_pos}"
    )
    select_cmd = create_select_cmd(
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
    gpu_sbatch_options = {"exclude": exclude_gpu_nodes} if exclude_gpu_nodes else {}
    cpu_sbatch_options = {"exclude": exclude_cpu_nodes} if exclude_cpu_nodes else {}
    select_job = Job(
        cmd=select_cmd,
        name=f"{slurm_id}_select_itr{itr}",
        out_file=str(itr_dir / f"select_ism.%j.log"),
        sb_file=str(itr_dir / "select_ism.sh"),
        queue="nvidia_geforce_rtx_4090",
        cpu=1,
        mem=20000,
        gpu=1,
        sbatch_options=gpu_sbatch_options,
    )
    select_job.launch()
    if verbose:
        print(f"  Launched select job {select_job.id}")
    wait_for_job(select_job)

    # Step 2: Design (GPU job) using ism_design.py
    logger.info(f"Step 2/3: Running design for itr{itr} with n={itr_n}")
    # Build bed_files list for all previous iterations
    if itr == 2:
        bed_files = "select_variants/variants.bed"
    else:
        # itr3 needs itr2/select_variants and itr3/select_variants
        bed_list = []
        for i in range(2, itr + 1):
            if i == itr:
                bed_list.append("select_variants/variants.bed")
            else:
                bed_list.append(f"../itr{i}/select_variants/variants.bed")
        bed_files = ",".join(bed_list)

    design_cmd = create_design_cmd(
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
    design_job = Job(
        cmd=design_cmd,
        name=f"{slurm_id}_design_itr{itr}",
        out_file=str(itr_dir / f"design.%j.log"),
        sb_file=str(itr_dir / "design.sh"),
        queue="nvidia_geforce_rtx_4090",
        cpu=1,
        mem=20000,
        gpu=1,
        sbatch_options=gpu_sbatch_options,
    )
    design_job.launch()
    if verbose:
        print(f"  Launched design job {design_job.id}")
    wait_for_job(design_job)

    # Step 3: Regression (CPU job)
    logger.info(f"Step 3/3: Running regression for itr{itr}")
    regress_cmd = create_regress_cmd(
        itr,
        str(itr_dir),
        n_cpu,
        hydra_env,
        script_dir,
        target_subset,
        vanilla_path,
        celltype,
        warm_start=warm_start,
    )
    regress_job = Job(
        cmd=regress_cmd,
        name=f"{slurm_id}_regress_itr{itr}",
        out_file=str(itr_dir / f"regress.%j.log"),
        sb_file=str(itr_dir / "regress.sh"),
        queue="standard",
        cpu=n_cpu,
        mem=60000 * itr,
        time="2-0:0:0",
        sbatch_options=cpu_sbatch_options,
    )
    regress_job.launch()
    if verbose:
        print(f"  Launched regression job {regress_job.id}")
    wait_for_job(regress_job)

    logger.info(f"Iteration {itr} completed successfully")


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


def cleanup_old_iterations(base_dir: str, outdir: str, n_iter: int):
    """Remove all iteration folders except the last one.

    Args:
        base_dir: Base directory path
        outdir: Output directory name
        n_iter: Total number of iterations (last iteration to keep)
    """
    outdir_path = Path(base_dir) / outdir

    logger.info(f"\nCleaning up old iterations, keeping only itr{n_iter}...")

    for itr in range(1, n_iter):
        itr_dir = outdir_path / f"itr{itr}"
        if itr_dir.exists():
            logger.info(f"Removing {itr_dir}")
            shutil.rmtree(itr_dir)
        else:
            logger.debug(f"Iteration folder {itr_dir} does not exist, skipping")

    logger.info(f"Cleanup complete. Only itr{n_iter} remains.")


def main():
    """Main entry point for the active learning script."""
    parser = argparse.ArgumentParser(
        description="Iterative m-ISM active learning with variant selection",
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
        "--n_cpu", type=int, default=64, help="Number of CPU cores for regression"
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
        "--verbose",
        action="store_true",
        help="Print verbose job submission information",
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
        help="Prefix for SLURM job names [Default: %(default)s]",
    )

    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue from the earliest incomplete iteration (checks for out/ism_tracks_comparison.pdf)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove all iteration folders except the last one after successful completion",
    )

    parser.add_argument(
        "--exclude_gpu_nodes",
        type=str,
        default=None,
        help="Comma-separated list of nodes to exclude from GPU jobs (e.g., 'node001,node002')",
    )

    parser.add_argument(
        "--exclude_cpu_nodes",
        type=str,
        default=None,
        help="Comma-separated list of nodes to exclude from CPU jobs (e.g., 'node003,node004')",
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

    if not args.continue_run:
        # Normal mode: run all iterations from the beginning
        run_iteration_1(
            base_dir,
            args.outdir,
            args.mut_len,
            args.initial_n,
            args.mut_distance_min,
            args.n_cpu,
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
            args.verbose,
            args.slurm_id,
            args.exclude_gpu_nodes,
            args.exclude_cpu_nodes,
        )

        for itr in range(2, args.n_iter + 1):
            run_iteration_n(
                itr,
                base_dir,
                args.outdir,
                args.mut_len,
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
                args.verbose,
                args.warm,
                args.k_background,
                args.slurm_id,
                args.exclude_gpu_nodes,
                args.exclude_cpu_nodes,
            )
    else:
        # Continue mode: check progress and resume from earliest incomplete iteration
        start_itr = check_progress(base_dir, args.outdir, args.n_iter)

        if start_itr > args.n_iter:
            logger.info("Nothing to do - all iterations already complete")
            return

        logger.info(f"Continuing from iteration {start_itr}")

        if start_itr == 1:
            run_iteration_1(
                base_dir,
                args.outdir,
                args.mut_len,
                args.initial_n,
                args.mut_distance_min,
                args.n_cpu,
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
                args.verbose,
                args.slurm_id,
                args.exclude_gpu_nodes,
                args.exclude_cpu_nodes,
            )
            start_itr = 2

        for itr in range(start_itr, args.n_iter + 1):
            run_iteration_n(
                itr,
                base_dir,
                args.outdir,
                args.mut_len,
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
                args.verbose,
                args.warm,
                args.k_background,
                args.slurm_id,
                args.exclude_gpu_nodes,
                args.exclude_cpu_nodes,
            )

    # Cleanup old iterations if requested
    if args.cleanup:
        cleanup_old_iterations(base_dir, args.outdir, args.n_iter)

    logger.info(f"\n{'='*60}")
    logger.info(f"All {args.n_iter} iterations completed successfully!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
