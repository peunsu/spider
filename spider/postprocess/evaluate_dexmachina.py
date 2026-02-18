# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute object tracking metrics (pos, rot, arti distances) from
trajectory_dexmachina.npz files produced by run_dexmachina.py, and
optionally compare against DexMachina RL rollout/evaluation results.

Usage:
    # Evaluate SPIDER results per (task, seed)
    python spider/postprocess/evaluate_dexmachina.py

    # Compare SPIDER vs RL across seeds (mean +/- std tables)
    python spider/postprocess/evaluate_dexmachina.py --compare --seeds 0 1 2 3 4

    # Compare with LaTeX output
    python spider/postprocess/evaluate_dexmachina.py --compare --latex

    # Specific tasks / dataset config
    python spider/postprocess/evaluate_dexmachina.py --tasks box-30-230 laptop-30-230
    python spider/postprocess/evaluate_dexmachina.py --dataset_name arctic --robot_type inspire_hand
"""

import argparse
import glob
import os

import numpy as np

import spider
from spider.io import get_all_tasks, get_processed_data_dir


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention, matching Genesis / DexMachina)
# ---------------------------------------------------------------------------

def _quat_conjugate(q):
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _rotation_distance(q1, q2):
    """Geodesic rotation distance (radians)."""
    qd = _quat_mul(q1, _quat_conjugate(q2))
    return 2.0 * np.arcsin(
        np.clip(np.linalg.norm(qd[..., 1:4], axis=-1), a_min=0, a_max=1.0)
    )


# ---------------------------------------------------------------------------
# Spider trajectory evaluation
# ---------------------------------------------------------------------------

def _load_demo_arti(task: str, hand: str = "inspire_hand",
                     subject: str = "s01") -> np.ndarray | None:
    """Load demo articulation trajectory from DexMachina retargeted data.

    Returns the obj_arti slice for the clip (e.g., frames 30-230), or None
    if the file cannot be found.
    """
    # task format: "box-30-230" or "box-30-230-s01-u02"
    parts = task.split("-")
    obj_name = parts[0]
    start, end = int(parts[1]), int(parts[2])
    if len(parts) >= 5:
        subject = parts[3]
        use_clip = parts[4].replace("u", "")
    else:
        use_clip = "01"

    # Try to find the retargeted .pt file
    asset_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../dexmachina/dexmachina/assets/retargeted",
        hand, subject,
    )
    clip_tag = f"{obj_name}_use_{use_clip}"
    pt_path = os.path.join(asset_dir, f"{clip_tag}_vector_para.pt")
    if not os.path.exists(pt_path):
        return None

    import torch
    data = torch.load(pt_path, weights_only=False)
    obj_arti = np.array(data["demo_data"]["obj_arti"])  # (total_frames,)
    return obj_arti[start:end]  # sliced to clip range


def evaluate_trajectory(npz_path: str, task: str = "") -> dict:
    """Load a trajectory_dexmachina.npz and compute mean/std of the 3 object
    distance metrics across all timesteps.

    The npz stores per-control-step, per-sub-step distances:
        obj_pos_dist:  (num_ctrl_steps, ctrl_steps)  -- L2 position distance
        obj_quat_dist: (num_ctrl_steps, ctrl_steps)  -- geodesic rotation distance
        obj_arti_dist: (num_ctrl_steps, ctrl_steps)  -- L2 articulation distance
                       NOTE: obj_arti_dist may be inflated due to a shape-
                       broadcast bug. When possible, we recompute it from the
                       saved qpos and demo data.

    Returns a dict with mean and std for each metric.
    """
    data = np.load(npz_path)
    metrics = {}

    for key in ["obj_pos_dist", "obj_quat_dist"]:
        vals = data[key].flatten()
        metrics[f"{key}_mean"] = vals.mean()
        metrics[f"{key}_std"] = vals.std()

    # Recompute arti_dist from qpos + demo if available, because saved
    # obj_arti_dist has a broadcast bug (demo shape (B,) vs obj shape (B,1)
    # produces (B,B) instead of (B,)).
    demo_arti = _load_demo_arti(task) if task else None
    if demo_arti is not None and "qpos" in data and "sim_step" in data:
        qpos = data["qpos"]            # (num_ctrl, sub_steps, 14)
        sim_steps = data["sim_step"]   # (num_ctrl,)
        sub_steps = qpos.shape[1]
        # qpos layout: [pos(3), quat(4), all_dofs(7)] -- revolute DOF at index 13
        obj_arti_vals = qpos[:, :, 13]  # (num_ctrl, sub_steps)

        # Build corresponding demo arti values
        demo_vals = np.zeros_like(obj_arti_vals)
        for i, s in enumerate(sim_steps):
            for j in range(sub_steps):
                t = int(s) + j + 1  # +1 because step_env increments before read
                t = min(t, len(demo_arti) - 1)
                demo_vals[i, j] = demo_arti[t]

        arti_dist = np.abs(obj_arti_vals - demo_vals)
        metrics["obj_arti_dist_mean"] = arti_dist.mean()
        metrics["obj_arti_dist_std"] = arti_dist.std()
    else:
        # Fall back to saved (potentially buggy) values
        vals = data["obj_arti_dist"].flatten()
        metrics["obj_arti_dist_mean"] = vals.mean()
        metrics["obj_arti_dist_std"] = vals.std()

    return metrics


# ---------------------------------------------------------------------------
# RL evaluation -- recompute distances from saved obj_state / demo_state
# ---------------------------------------------------------------------------

def evaluate_rl_rollout(npz_path: str) -> dict:
    """Load a rollout_rl.npz and compute pos/rot/arti distances.

    obj_state:  (T, num_envs, 8)  -- [pos(3), quat(4), arti(1)]
    demo_state: (T, 8)

    Returns a dict with unified metric keys (pos_dist, rot_dist, arti_dist).
    """
    data = np.load(npz_path)
    obj = data["obj_state"][:, 0, :]  # (T, 8)
    demo = data["demo_state"]         # (T, 8)

    pos_dist = np.linalg.norm(obj[:, :3] - demo[:, :3], axis=-1)
    rot_dist = _rotation_distance(obj[:, 3:7], demo[:, 3:7])
    arti_dist = np.abs(obj[:, 7] - demo[:, 7])

    return {
        "pos_dist": float(pos_dist.mean()),
        "rot_dist": float(rot_dist.mean()),
        "arti_dist": float(arti_dist.mean()),
    }


def evaluate_rl_eval(npy_path: str) -> dict:
    """Load an eval_ep*.npy from DexMachina RL evaluation and recompute
    pos / rot / arti distances from the raw obj_state and demo_state so that
    the metrics are comparable to spider (L2 for pos & arti, geodesic for rot).

    obj_state:  (T, num_envs, 8)  -- [pos(3), quat(4), arti(1)]
    demo_state: (T, 8)
    """
    data = np.load(npy_path, allow_pickle=True).item()
    obj = data["obj_state"][:, 0, :]   # (T, 8)
    demo = data["demo_state"]          # (T, 8)

    pos_dist = np.linalg.norm(obj[:, :3] - demo[:, :3], axis=-1)
    rot_dist = _rotation_distance(obj[:, 3:7], demo[:, 3:7])
    arti_dist = np.abs(obj[:, 7] - demo[:, 7])

    return {
        "obj_pos_dist_mean": pos_dist.mean(),
        "obj_pos_dist_std": pos_dist.std(),
        "obj_quat_dist_mean": rot_dist.mean(),
        "obj_quat_dist_std": rot_dist.std(),
        "obj_arti_dist_mean": arti_dist.mean(),
        "obj_arti_dist_std": arti_dist.std(),
    }


def discover_rl_evals(rl_eval_dir: str) -> dict:
    """Scan DexMachina RL log directory for eval .npy files.

    Expected structure:
        rl_eval_dir/<run_name>/<ckpt_name>_eval/eval_ep*.npy

    The task name is extracted from the run name (e.g. 'box' from
    'inspire-inspire_box_box30-230-...').

    Returns {task_name: npy_path} mapping.
    """
    results = {}
    eval_npys = glob.glob(os.path.join(rl_eval_dir, "*", "*_eval", "eval_ep0.npy"))
    for npy_path in eval_npys:
        # extract task from the run directory name
        run_dir = npy_path.split(os.sep)[-3]
        # pattern: inspire-inspire_<obj>_<obj><start>-<end>-...
        # e.g. inspire-inspire_box_box30-230-s01-u01_B4096_...
        parts = run_dir.split("_")
        # find the object name: it's the part after "inspire-inspire"
        # and construct the clip string
        obj = None
        for i, p in enumerate(parts):
            if p.startswith("box") or p.startswith("ketchup") or \
               p.startswith("laptop") or p.startswith("mixer") or \
               p.startswith("notebook") or p.startswith("waffleiron"):
                clip_part = parts[i + 1] if i + 1 < len(parts) else p
                for obj_name in ["box", "ketchup", "laptop", "mixer", "notebook", "waffleiron"]:
                    if clip_part.startswith(obj_name):
                        rest = clip_part[len(obj_name):]  # "30-230-s01-u01"
                        nums = rest.split("-")
                        if len(nums) >= 2:
                            task_key = f"{obj_name}-{nums[0]}-{nums[1]}"
                            obj = task_key
                            break
                if obj:
                    break
        if obj:
            results[obj] = npy_path
    return results


# ---------------------------------------------------------------------------
# Multi-seed comparison: collect results for SPIDER and RL across seeds
# ---------------------------------------------------------------------------

# Unified metric keys used by the comparison functions
METRIC_KEYS = ["pos_dist", "rot_dist", "arti_dist"]
METRIC_LABELS = {
    "pos_dist": "Position Distance (L2)",
    "rot_dist": "Rotation Distance (geodesic)",
    "arti_dist": "Articulation Distance (L1)",
}


def _spider_metrics_to_unified(metrics: dict) -> dict:
    """Convert evaluate_trajectory() output to unified metric keys."""
    return {
        "pos_dist": metrics["obj_pos_dist_mean"],
        "rot_dist": metrics["obj_quat_dist_mean"],
        "arti_dist": metrics["obj_arti_dist_mean"],
    }


def collect_comparison_results(
    tasks: list[str],
    seeds: list[int],
    dataset_dir: str,
    dataset_name: str = "arctic",
    robot_type: str = "inspire_hand",
    embodiment_type: str = "bimanual",
) -> dict:
    """Collect SPIDER and RL metrics for each (task, seed).

    Uses spider.io.get_processed_data_dir to resolve paths consistently with
    the rest of the codebase.

    Returns {method: {task: {seed: {metric: value}}}}.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    results = {"SPIDER": {}, "RL": {}}

    for task in tasks:
        results["SPIDER"][task] = {}
        results["RL"][task] = {}
        for seed in seeds:
            seed_dir = get_processed_data_dir(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                robot_type=robot_type,
                embodiment_type=embodiment_type,
                task=task,
                data_id=seed,
            )

            # SPIDER
            spider_npz = os.path.join(seed_dir, "trajectory_dexmachina.npz")
            if os.path.exists(spider_npz):
                metrics = evaluate_trajectory(spider_npz, task=task)
                results["SPIDER"][task][seed] = _spider_metrics_to_unified(metrics)
            else:
                print(f"WARNING: missing SPIDER file: {spider_npz}")

            # RL rollout
            rl_npz = os.path.join(seed_dir, "rollout_rl.npz")
            if os.path.exists(rl_npz):
                results["RL"][task][seed] = evaluate_rl_rollout(rl_npz)
            else:
                print(f"WARNING: missing RL file: {rl_npz}")

    return results


def _aggregate(task_seeds: dict, metric: str):
    """Compute mean and std across seeds for one metric.

    task_seeds: {seed: {metric: value}} -- may have missing seeds.
    Returns (mean, std) or (nan, nan) if no data.
    """
    vals = [
        task_seeds[s][metric]
        for s in sorted(task_seeds)
        if metric in task_seeds[s]
    ]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(mean, std):
    return f"{mean:.4f} \u00b1 {std:.4f}"


def _fmt_pm(mean, std):
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.4f} +/- {std:.4f}"


def _fmt_latex(mean, std):
    if np.isnan(mean):
        return "N/A"
    return f"${mean:.4f} \\pm {std:.4f}$"


# ---------------------------------------------------------------------------
# Single-run tables (original per-task, per-data_id view)
# ---------------------------------------------------------------------------

def _print_table(title, results, show_id=True):
    print(f"\n{title}")
    if show_id:
        header = f"  {'Task':<30} {'ID':>3}  {'pos_dist':>18}  {'rot_dist':>18}  {'arti_dist':>18}"
    else:
        header = f"  {'Task':<30}       {'pos_dist':>18}  {'rot_dist':>18}  {'arti_dist':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        id_str = f"{r['data_id']:>3}  " if show_id else "      "
        print(
            f"  {r['task']:<30} {id_str}"
            f"{_fmt(r['obj_pos_dist_mean'], r['obj_pos_dist_std'])}  "
            f"{_fmt(r['obj_quat_dist_mean'], r['obj_quat_dist_std'])}  "
            f"{_fmt(r['obj_arti_dist_mean'], r['obj_arti_dist_std'])}"
        )


def _print_comparison(spider_results, rl_results):
    """Print side-by-side comparison for tasks present in both."""
    common_tasks = sorted(
        set(r["task"] for r in spider_results) & set(rl_results.keys())
    )
    if not common_tasks:
        print("\nNo overlapping tasks to compare.")
        return

    spider_by_task = {r["task"]: r for r in spider_results}

    print(f"\n{'='*100}")
    print("Comparison: Spider (SPIDER) vs DexMachina RL (RL)")
    print(f"{'='*100}")
    header = f"  {'Task':<25} {'Method':<8} {'pos_dist':>18}  {'rot_dist':>18}  {'arti_dist':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for task in common_tasks:
        sp = spider_by_task[task]
        rl = rl_results[task]
        print(
            f"  {task:<25} {'SPIDER':<8} "
            f"{_fmt(sp['obj_pos_dist_mean'], sp['obj_pos_dist_std'])}  "
            f"{_fmt(sp['obj_quat_dist_mean'], sp['obj_quat_dist_std'])}  "
            f"{_fmt(sp['obj_arti_dist_mean'], sp['obj_arti_dist_std'])}"
        )
        print(
            f"  {'':<25} {'RL':<8} "
            f"{_fmt(rl['obj_pos_dist_mean'], rl['obj_pos_dist_std'])}  "
            f"{_fmt(rl['obj_quat_dist_mean'], rl['obj_quat_dist_std'])}  "
            f"{_fmt(rl['obj_arti_dist_mean'], rl['obj_arti_dist_std'])}"
        )
        print()


# ---------------------------------------------------------------------------
# Multi-seed comparison tables
# ---------------------------------------------------------------------------

def print_markdown_tables(results, tasks, methods=("SPIDER", "RL")):
    """Print one Markdown table per metric (mean +/- std across seeds)."""
    for mk in METRIC_KEYS:
        print(f"\n## {METRIC_LABELS[mk]} (lower is better)")
        print()

        cols = ["Task"] + list(methods) + ["Winner"]
        header = "| " + " | ".join(
            f"{c:<30}" if i == 0 else f"{c:<20}" for i, c in enumerate(cols)
        ) + " |"
        sep = "| " + " | ".join(
            "-" * (30 if i == 0 else 20) for i in range(len(cols))
        ) + " |"
        print(header)
        print(sep)

        method_avgs = {m: [] for m in methods}
        for task in tasks:
            row = [task]
            means = {}
            for method in methods:
                m, s = _aggregate(results[method].get(task, {}), mk)
                row.append(_fmt_pm(m, s))
                means[method] = m
                if not np.isnan(m):
                    method_avgs[method].append(m)

            valid = {m: v for m, v in means.items() if not np.isnan(v)}
            winner = min(valid, key=valid.get) if len(valid) >= 2 else "-"
            row.append(winner)

            print("| " + " | ".join(
                f"{c:<30}" if i == 0 else f"{c:<20}" for i, c in enumerate(row)
            ) + " |")

        # Average row
        row = ["**Average**"]
        avg_means = {}
        for method in methods:
            vals = method_avgs[method]
            if vals:
                m, s = np.mean(vals), np.std(vals)
                row.append(f"**{_fmt_pm(m, s)}**")
                avg_means[method] = m
            else:
                row.append("N/A")
                avg_means[method] = float("nan")

        valid_avg = {m: v for m, v in avg_means.items() if not np.isnan(v)}
        winner = f"**{min(valid_avg, key=valid_avg.get)}**" if len(valid_avg) >= 2 else "-"
        row.append(winner)

        print("| " + " | ".join(
            f"{c:<30}" if i == 0 else f"{c:<20}" for i, c in enumerate(row)
        ) + " |")
        print()


def print_latex_tables(results, tasks, methods=("SPIDER", "RL")):
    """Print one LaTeX tabular per metric."""
    n_methods = len(methods)
    col_spec = "l" + "c" * n_methods

    for mk in METRIC_KEYS:
        print(f"\n% {METRIC_LABELS[mk]} (lower is better)")
        print("\\begin{table}[h]")
        print("\\centering")
        print(f"\\caption{{{METRIC_LABELS[mk]} (lower is better)}}")
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print("\\toprule")
        print("Task & " + " & ".join(methods) + " \\\\")
        print("\\midrule")

        method_avgs = {m: [] for m in methods}
        for task in tasks:
            cells = [task.replace("_", "\\_")]
            best_val = float("inf")
            best_method = None
            for method in methods:
                m, s = _aggregate(results[method].get(task, {}), mk)
                if not np.isnan(m):
                    method_avgs[method].append(m)
                    if m < best_val:
                        best_val = m
                        best_method = method

            for method in methods:
                m, s = _aggregate(results[method].get(task, {}), mk)
                cell = _fmt_latex(m, s)
                if method == best_method and not np.isnan(m):
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)

            print(" & ".join(cells) + " \\\\")

        print("\\midrule")
        cells = ["\\textbf{Average}"]
        for method in methods:
            vals = method_avgs[method]
            if vals:
                cells.append(_fmt_latex(np.mean(vals), np.std(vals)))
            else:
                cells.append("N/A")
        print(" & ".join(cells) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print()


def print_per_seed_detail(results, tasks, seeds, methods=("SPIDER", "RL")):
    """Print a detailed per-seed table for each metric."""
    for mk in METRIC_KEYS:
        print(f"\n### {METRIC_LABELS[mk]} -- per-seed detail")
        print()

        seed_cols = [f"s{s}" for s in seeds]
        header_parts = ["Task", "Method"] + seed_cols + ["Mean", "Std"]
        header = "| " + " | ".join(f"{c:>10}" for c in header_parts) + " |"
        sep = "| " + " | ".join("-" * 10 for _ in header_parts) + " |"
        print(header)
        print(sep)

        for task in tasks:
            for method in methods:
                row = [task if method == methods[0] else "", method]
                vals = []
                for seed in seeds:
                    v = results[method].get(task, {}).get(seed, {}).get(mk, float("nan"))
                    row.append(f"{v:.4f}" if not np.isnan(v) else "N/A")
                    vals.append(v)
                arr = np.array([v for v in vals if not np.isnan(v)])
                if len(arr) > 0:
                    row.append(f"{arr.mean():.4f}")
                    row.append(f"{arr.std():.4f}")
                else:
                    row.extend(["N/A", "N/A"])

                print("| " + " | ".join(f"{c:>10}" for c in row) + " |")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DexMachina trajectory metrics"
    )
    # Dataset path resolution (matches Config / dexmachina.yaml defaults)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(spider.ROOT, "..", "example_datasets"),
        help="Root dataset directory (default: example_datasets/)",
    )
    parser.add_argument("--dataset_name", type=str, default="arctic")
    parser.add_argument("--robot_type", type=str, default="inspire_hand")
    parser.add_argument("--embodiment_type", type=str, default="bimanual")
    # Legacy --data_dir for backwards compatibility
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override: explicit path to task directories (bypasses dataset_dir resolution)",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Specific task names to evaluate (default: all found)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--trajectory_name", type=str, default="trajectory_dexmachina.npz",
        help="Name of the trajectory file to look for",
    )
    parser.add_argument(
        "--rl_eval_dir", type=str, default=None,
        help="DexMachina RL logs directory containing eval .npy results",
    )
    # Comparison mode
    parser.add_argument(
        "--compare", action="store_true",
        help="Multi-seed SPIDER vs RL comparison (mean +/- std tables)",
    )
    parser.add_argument("--latex", action="store_true", help="Print LaTeX tables")
    parser.add_argument("--detail", action="store_true", help="Print per-seed detail tables")
    args = parser.parse_args()

    # Resolve the data directory
    dataset_dir = os.path.abspath(args.dataset_dir)
    if args.data_dir is not None:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = os.path.join(
            dataset_dir, "processed",
            args.dataset_name, args.robot_type, args.embodiment_type,
        )

    # Resolve task list
    if args.tasks is not None:
        task_names = args.tasks
    elif os.path.isdir(data_dir):
        task_names = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
    else:
        print(f"Warning: data directory not found: {data_dir}")
        return

    # ── Multi-seed comparison mode ────────────────────────────────────────
    if args.compare:
        print("=" * 70)
        print("SPIDER vs RL Comparison")
        print(f"Tasks: {len(task_names)}  |  Seeds: {args.seeds}")
        print("=" * 70)

        results = collect_comparison_results(
            tasks=task_names,
            seeds=args.seeds,
            dataset_dir=dataset_dir,
            dataset_name=args.dataset_name,
            robot_type=args.robot_type,
            embodiment_type=args.embodiment_type,
        )

        for method in ("SPIDER", "RL"):
            count = sum(len(results[method].get(t, {})) for t in task_names)
            expected = len(task_names) * len(args.seeds)
            print(f"  {method}: {count} / {expected} results found")
        print()

        print("=" * 70)
        print("MARKDOWN TABLES")
        print("=" * 70)
        print_markdown_tables(results, task_names)

        if args.detail:
            print("=" * 70)
            print("PER-SEED DETAIL")
            print("=" * 70)
            print_per_seed_detail(results, task_names, args.seeds)

        if args.latex:
            print("=" * 70)
            print("LATEX TABLES")
            print("=" * 70)
            print_latex_tables(results, task_names)

        return

    # ── Single-run evaluation mode (original behavior) ────────────────────
    spider_results = []
    if os.path.isdir(data_dir):
        for task in task_names:
            task_dir = os.path.join(data_dir, task)
            if not os.path.isdir(task_dir):
                print(f"Warning: task directory not found: {task_dir}")
                continue
            data_id_dirs = sorted(
                d for d in os.listdir(task_dir)
                if os.path.isdir(os.path.join(task_dir, d)) and d.isdigit()
            )
            for data_id in data_id_dirs:
                npz_path = os.path.join(task_dir, data_id, args.trajectory_name)
                if not os.path.exists(npz_path):
                    print(f"Warning: {npz_path} not found, skipping")
                    continue
                metrics = evaluate_trajectory(npz_path, task=task)
                metrics["task"] = task
                metrics["data_id"] = int(data_id)
                spider_results.append(metrics)

        if spider_results:
            _print_table("SPIDER Results", spider_results)

    # RL eval (legacy .npy from eval_rl_games.py)
    rl_eval_dir = args.rl_eval_dir
    if rl_eval_dir is None:
        rl_eval_dir = os.path.join(
            os.path.dirname(__file__),
            "../../../dexmachina/logs/rl_games/inspire_hand",
        )
    rl_eval_dir = os.path.abspath(rl_eval_dir)
    rl_results = {}
    if os.path.isdir(rl_eval_dir):
        task_to_npy = discover_rl_evals(rl_eval_dir)
        rl_result_list = []
        for task, npy_path in sorted(task_to_npy.items()):
            metrics = evaluate_rl_eval(npy_path)
            metrics["task"] = task
            rl_results[task] = metrics
            rl_result_list.append(metrics)

        if rl_result_list:
            _print_table("DexMachina RL Results", rl_result_list, show_id=False)

    if spider_results and rl_results:
        _print_comparison(spider_results, rl_results)


if __name__ == "__main__":
    main()
