"""Compare keyframe sampling methods inside one LiDAR SLAM pipeline.

    .venv/bin/python run.py --config configs/kitti00.yaml
    .venv/bin/python run.py --config configs/kitti00.yaml --methods all msa --max-frames 1000
"""

import argparse
import json
import os

import yaml

from msa_slam.data import make_dataset
from msa_slam.descriptors import make_descriptor
from msa_slam.frontend import make_frontend
from msa_slam.pipeline import run_frontend, run_method

COLUMNS = [
    ("method", "Method", "{}"),
    ("keyframes", "Keyframes", "{}"),
    ("memory_raw_scans_MB", "Memory raw (MB)", "{:.0f}"),
    ("memory_backend_MB", "Memory back-end (MB)", "{:.0f}"),
    ("time_sampling_s", "Sampling (s)", "{:.1f}"),
    ("time_backend_s", "Back-end (s)", "{:.1f}"),
    ("time_total_s", "Total (s)", "{:.1f}"),
    ("loops", "Loops", "{}"),
    ("loop_precision", "Loop prec.", "{:.2f}"),
    ("f1_max", "F1-max", "{:.3f}"),
    ("ate_opt_m", "ATE (m)", "{:.2f}"),
    ("ate_improvement_pct", "ATE impr. (%)", "{:.1f}"),
]


def table(results):
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = ["| " + " | ".join(fmt.format(r[k]) if k in r else "-" for k, _, fmt in COLUMNS) + " |"
            for r in results]
    return "\n".join([head, sep] + rows)


def plot(results, gt, odom, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6), squeeze=False)
    for ax, r in zip(axes[0], results):
        if gt is not None:
            ax.plot(gt[:, 0, 3], gt[:, 1, 3], color="0.75", lw=3, label="GT")
        ax.plot(odom[:, 0, 3], odom[:, 1, 3], "--", color="tab:orange", lw=1, label="KISS-ICP (odometry)")
        t = r["_traj"]
        ax.plot(t[:, 0, 3], t[:, 1, 3], color="tab:blue", lw=1, label="optimised")
        kf = r["_keyframes"]
        ax.scatter(t[kf, 0, 3], t[kf, 1, 3], s=2, color="k", label=f"keyframes ({len(kf)})", zorder=3)
        for lp in r["_loops"]:
            q, m = lp["query_frame"], lp["match_frame"]
            ax.plot([t[q, 0, 3], t[m, 0, 3]], [t[q, 1, 3], t[m, 1, 3]], color="tab:red", lw=0.8)
        ax.plot([], [], color="tab:red", label=f"loop closures ({len(r['_loops'])})")
        title = r["method"]
        if "ate_opt_m" in r:
            title += f"  |  ATE {r['ate_opt_m']:.2f} m"
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kitti00.yaml")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    methods = args.methods or cfg["sampling"]["methods"]
    dataset = make_dataset(cfg["dataset"])
    n = min(len(dataset), args.max_frames or len(dataset))
    out_root = os.path.join(cfg.get("output", "results"), f"{dataset.name}_{n}")

    cache = os.path.join(out_root, "stage1_cache.npz")
    if os.path.exists(cache):
        frontend = descriptor = None
    else:
        frontend = make_frontend(cfg["frontend"])
        descriptor = make_descriptor(cfg["descriptor"])
    stage1 = run_frontend(dataset, frontend, descriptor, n, cfg["backend"].get("fine_voxel", 0.5), cache)
    print(f"[stage 1] front-end {float(stage1['t_frontend']):.1f} s, descriptors {float(stage1['t_descriptor']):.1f} s")

    results = []
    for m in methods:
        r = run_method(m, dataset, stage1, n, cfg, os.path.join(out_root, m))
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, indent=1))
        results.append(r)

    md = table(results)
    print("\n" + md)
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write(md + "\n")
    gt = dataset.gt_poses[:n] if dataset.gt_poses is not None else None
    plot(results, gt, stage1["odom"], os.path.join(out_root, "trajectories.png"))
    print(f"\nresults: {out_root}")


if __name__ == "__main__":
    main()
