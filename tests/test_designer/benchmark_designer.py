#!/usr/bin/env python3
"""Scaling benchmark for MutationDesigner design methods.

Runs each design method across a set of window sizes (mut_len) with a fixed
number of replicates and writes a markdown report with key design statistics
and run-time.

Each (method, window) combination runs in an isolated subprocess so that
run-time and peak memory are measured independently and a blow-up in one
configuration cannot corrupt the others.

Usage
-----
Driver (default) -- run the full sweep and write report.md:

    python tests/test_designer/benchmark_designer.py

Single worker (used internally by the driver):

    python tests/test_designer/benchmark_designer.py --worker <method> <window> <nreps> <min_dist>
"""

import argparse
import contextlib
import io
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from mism.core.mutation_designer import MutationDesigner

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
WINDOWS = [1_000, 10_000, 100_000]
NREPS = 20
MIN_DIST = 50
METHODS = ["vanilla", "bins", "star-and-bar", "random"]

# random design is rate-based (does not use nreps); fixed here for reference
RANDOM_MUT_RATE = 0.01
RANDOM_NUM_SEQ = 1_000

# vanilla is a dense (3L, L, 3) bool tensor -> skip when it would exceed this
MAX_VANILLA_BYTES = 2_000_000_000  # ~2 GB

PER_RUN_TIMEOUT_S = 1_800  # 30 min hard cap per (method, window)

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.md")


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def compute_stats(X_mut3, mut_len):
    """Compute key design statistics from a design output.

    Handles the dense ndarray output (vanilla) and the list-of-coo output.
    """
    all_rows = []
    all_cols = []
    min_within = np.inf
    dist_sum = 0.0
    dist_count = 0

    if isinstance(X_mut3, np.ndarray):
        n_seq = int(X_mut3.shape[0])
        muts_per_seq = X_mut3.reshape(n_seq, -1).sum(axis=1).astype(np.int64)
        for i in range(n_seq):
            rows, cols = np.where(X_mut3[i])
            all_rows.append(rows.astype(np.int64))
            all_cols.append(cols.astype(np.int64))
            if rows.size > 1:
                d = np.diff(np.sort(rows))
                min_within = min(min_within, int(d.min()))
                dist_sum += float(d.sum())
                dist_count += int(d.size)
    else:
        n_seq = len(X_mut3)
        muts_per_seq = np.zeros(n_seq, dtype=np.int64)
        for i, seq in enumerate(X_mut3):
            coo = seq if hasattr(seq, "row") else seq.tocoo()
            rows = coo.row.astype(np.int64)
            cols = coo.col.astype(np.int64)
            muts_per_seq[i] = coo.nnz
            all_rows.append(rows)
            all_cols.append(cols)
            if rows.size > 1:
                sr = np.sort(rows)
                d = np.diff(sr)
                if d.size:
                    min_within = min(min_within, int(d.min()))
                    dist_sum += float(d.sum())
                    dist_count += int(d.size)

    if all_rows:
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        flat = rows * 3 + cols
        mut_cov_flat = np.bincount(flat, minlength=mut_len * 3)
    else:
        mut_cov_flat = np.zeros(mut_len * 3, dtype=np.int64)

    mut_cov = mut_cov_flat.reshape(mut_len, 3)
    pos_cov = mut_cov.sum(axis=1)
    nonzero_mut = mut_cov[mut_cov > 0]

    return {
        "n_seq": int(n_seq),
        "total_mut": int(muts_per_seq.sum()) if n_seq else 0,
        "mut_per_seq_min": int(muts_per_seq.min()) if n_seq else 0,
        "mut_per_seq_max": int(muts_per_seq.max()) if n_seq else 0,
        "mut_per_seq_mean": float(muts_per_seq.mean()) if n_seq else 0.0,
        "min_within_dist": (int(min_within) if np.isfinite(min_within) else None),
        "mean_within_dist": (dist_sum / dist_count if dist_count else None),
        "pos_covered_frac": float((pos_cov > 0).mean()),
        "pos_cov_min": int(pos_cov.min()),
        "pos_cov_max": int(pos_cov.max()),
        "mut_reps_min": int(nonzero_mut.min()) if nonzero_mut.size else 0,
        "mut_reps_max": int(nonzero_mut.max()) if nonzero_mut.size else 0,
        "mut_reps_mean": float(nonzero_mut.mean()) if nonzero_mut.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# worker: run a single (method, window)
# --------------------------------------------------------------------------- #
def run_worker(method, window, nreps, min_dist):
    result = {"method": method, "window": window, "nreps": nreps, "min_dist": min_dist}

    # feasibility guard for the dense vanilla tensor
    if method == "vanilla":
        est_bytes = 9 * window * window  # (3L) * L * 3 bool bytes
        if est_bytes > MAX_VANILLA_BYTES:
            result["status"] = "skipped"
            result["reason"] = f"dense tensor ~{est_bytes/1e9:.1f} GB exceeds cap"
            return result

    if method == "vanilla":
        design_params = {}
    elif method in ("bins", "star-and-bar"):
        design_params = {"mut_reps": nreps, "mut_distance_min": min_dist}
    elif method == "random":
        design_params = {"mut_rate": RANDOM_MUT_RATE, "num_seq": RANDOM_NUM_SEQ}
    else:
        result["status"] = "error"
        result["reason"] = f"unknown method {method}"
        return result

    designer = MutationDesigner()

    # time only the design work; silence the method's internal prints
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            designer.design_initial_batch(window, method, design_params)
            design_time = time.perf_counter() - t0
            X_mut3 = designer.get_mutation_matrix()
            stats = compute_stats(X_mut3, window)
    except MemoryError:
        result["status"] = "error"
        result["reason"] = "MemoryError"
        return result

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    result["status"] = "ok"
    result["design_time_s"] = design_time
    result["peak_rss_mb"] = peak_rss_kb / 1024.0
    result.update(stats)
    return result


# --------------------------------------------------------------------------- #
# driver: sweep all combinations via subprocesses
# --------------------------------------------------------------------------- #
def run_driver():
    results = []
    for window in WINDOWS:
        for method in METHODS:
            label = f"{method} @ L={window}"
            print(f"[run] {label} ...", file=sys.stderr, flush=True)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                method,
                str(window),
                str(NREPS),
                str(MIN_DIST),
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S
                )
            except subprocess.TimeoutExpired:
                print(f"[timeout] {label}", file=sys.stderr, flush=True)
                results.append(
                    {
                        "method": method,
                        "window": window,
                        "status": "timeout",
                        "reason": f">{PER_RUN_TIMEOUT_S}s",
                    }
                )
                continue

            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            try:
                res = json.loads(line)
            except (json.JSONDecodeError, IndexError):
                print(
                    f"[error] {label}: could not parse worker output\n{proc.stderr}",
                    file=sys.stderr,
                    flush=True,
                )
                res = {
                    "method": method,
                    "window": window,
                    "status": "error",
                    "reason": "no parseable output",
                }
            status = res.get("status")
            extra = ""
            if status == "ok":
                extra = f"{res['design_time_s']:.2f}s, {res['n_seq']} seqs"
            else:
                extra = res.get("reason", status)
            print(f"[done] {label}: {status} ({extra})", file=sys.stderr, flush=True)
            results.append(res)

    write_report(results)
    print(f"\nReport written to {REPORT_PATH}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt(v, spec=""):
    if v is None:
        return "-"
    if spec:
        return format(v, spec)
    return str(v)


def _get(results, method, window):
    for r in results:
        if r.get("method") == method and r.get("window") == window:
            return r
    return None


def write_report(results):
    lines = []
    lines.append("# MutationDesigner benchmark report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Windows (mut_len):** {', '.join(f'{w:,}' for w in WINDOWS)}")
    lines.append(f"- **Replicates (nreps / mut_reps):** {NREPS} (fixed)")
    lines.append(f"- **mut_distance_min:** {MIN_DIST} (for `bins` and `star-and-bar`)")
    lines.append(
        f"- **random design:** rate-based (mut_rate={RANDOM_MUT_RATE}, "
        f"num_seq={RANDOM_NUM_SEQ}); does not use nreps"
    )
    lines.append(
        f"- **vanilla design:** single-mutation ISM; ignores nreps; "
        f"skipped when dense tensor > {MAX_VANILLA_BYTES/1e9:.0f} GB"
    )
    lines.append("")

    # ---- run-time summary table -----------------------------------------
    lines.append("## Run-time summary (seconds)")
    lines.append("")
    header = "| Method | " + " | ".join(f"L={w:,}" for w in WINDOWS) + " |"
    sep = "|" + "---|" * (len(WINDOWS) + 1)
    lines.append(header)
    lines.append(sep)
    for method in METHODS:
        row = [method]
        for w in WINDOWS:
            r = _get(results, method, w)
            if r is None:
                row.append("-")
            elif r.get("status") == "ok":
                row.append(f"{r['design_time_s']:.2f}")
            elif r.get("status") == "skipped":
                row.append("skip")
            elif r.get("status") == "timeout":
                row.append("timeout")
            else:
                row.append("err")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- peak memory summary --------------------------------------------
    lines.append("## Peak memory summary (MB, process RSS)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for method in METHODS:
        row = [method]
        for w in WINDOWS:
            r = _get(results, method, w)
            if r and r.get("status") == "ok":
                row.append(f"{r['peak_rss_mb']:.0f}")
            elif r and r.get("status") == "skipped":
                row.append("skip")
            elif r and r.get("status") == "timeout":
                row.append("timeout")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- per-window detailed stats --------------------------------------
    lines.append("## Design statistics per window")
    lines.append("")
    stat_cols = [
        ("n_seq", "# sequences", ",d"),
        ("total_mut", "total mutations", ",d"),
        ("mut_per_seq_mean", "mut/seq (mean)", ".1f"),
        ("mut_per_seq_min", "mut/seq (min)", "d"),
        ("mut_per_seq_max", "mut/seq (max)", "d"),
        ("min_within_dist", "min gap", "d"),
        ("mean_within_dist", "mean gap", ".1f"),
        ("pos_covered_frac", "pos covered", ".2%"),
        ("pos_cov_min", "pos cov (min)", "d"),
        ("mut_reps_mean", "reps/mut (mean)", ".1f"),
        ("mut_reps_min", "reps/mut (min)", "d"),
        ("mut_reps_max", "reps/mut (max)", "d"),
        ("design_time_s", "time (s)", ".2f"),
        ("peak_rss_mb", "peak MB", ".0f"),
    ]
    for w in WINDOWS:
        lines.append(f"### Window L = {w:,}")
        lines.append("")
        head = "| Statistic | " + " | ".join(METHODS) + " |"
        lines.append(head)
        lines.append("|" + "---|" * (len(METHODS) + 1))
        for key, label, spec in stat_cols:
            row = [label]
            for method in METHODS:
                r = _get(results, method, w)
                if r is None or r.get("status") != "ok":
                    tag = r.get("status", "-") if r else "-"
                    row.append(tag if tag != "ok" else "-")
                else:
                    row.append(_fmt(r.get(key), spec))
            lines.append("| " + " | ".join(row) + " |")
        # note any non-ok status
        notes = []
        for method in METHODS:
            r = _get(results, method, w)
            if r and r.get("status") != "ok":
                notes.append(f"`{method}`: {r.get('status')} ({r.get('reason', '')})")
        if notes:
            lines.append("")
            lines.append("_" + "; ".join(notes) + "_")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **min gap** = smallest distance between two mutated positions "
        "within a single sequence (spacing guarantee)."
    )
    lines.append(
        "- **reps/mut** = how many times each individual (position, "
        "nucleotide) mutation appears across all sequences."
    )
    lines.append(
        "- **pos covered** = fraction of the window's positions that "
        "receive at least one mutation."
    )
    lines.append(
        "- `vanilla` is single-mutation ISM: one mutation per sequence "
        "(min gap is undefined), coverage is exactly 1 per (pos, nt)."
    )
    lines.append(
        "- Each configuration ran in an isolated subprocess; peak MB is "
        "that process's peak RSS."
    )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        nargs=4,
        metavar=("METHOD", "WINDOW", "NREPS", "MIN_DIST"),
        help="Run a single design configuration and print JSON stats.",
    )
    args = parser.parse_args()

    if args.worker:
        method, window, nreps, min_dist = args.worker
        res = run_worker(method, int(window), int(nreps), int(min_dist))
        print(json.dumps(res))
    else:
        run_driver()


if __name__ == "__main__":
    main()
