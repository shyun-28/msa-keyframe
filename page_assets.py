"""Make the project-page figures from a finished run.

    .venv/bin/python page_assets.py --results results/kitti_00_4541

Writes assets/kitti00_demo.gif (w/o vs w/ MSA replay), assets/kitti00_traj.png
(full trajectory + zoom) and assets/kitti00_results.json (numbers for the page).
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

NAVY, ORANGE, GRAY, INK, RED = "#1b3557", "#c0601c", "#a8b3c4", "#1d2430", "#d1352b"
PAIR = [("all", "w/o MSA (all frames)"), ("msa", "w/ MSA")]


GT_XY = None  # set in main(); trajectories are drawn SE(3)-aligned to GT, as in the ATE


def load(res_dir, method):
    from msa_slam.metrics import umeyama_se3
    d = os.path.join(res_dir, method)
    m = json.load(open(os.path.join(d, "metrics.json")))
    xy = np.loadtxt(os.path.join(d, "poses_opt.txt")).reshape(-1, 3, 4)[:, :, 3]
    T = umeyama_se3(xy, GT_XY)
    xy = (T[:3, :3] @ xy.T).T + T[:3, 3]
    kf = np.loadtxt(os.path.join(d, "keyframes.txt"), dtype=int).reshape(-1)
    loops = json.load(open(os.path.join(d, "loops.json")))
    return m, xy, kf, loops


def style(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dde3ea")


def gt_xy(cfg_root, seq, n):
    from msa_slam.data import KittiOdometry
    return KittiOdometry(cfg_root, seq).gt_poses[:n, :3, 3]


def static_figure(res_dir, gt, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), gridspec_kw={"height_ratios": [1.15, 1]})
    zoom = (180, 300, -60, 40)  # x0, x1, y0, y1: revisited crossroads
    for col, (method, title) in enumerate(PAIR):
        m, xy, kf, loops = load(res_dir, method)
        color = NAVY if method == "msa" else GRAY
        for row, ax in enumerate(axes[:, col]):
            ax.plot(gt[:, 0], gt[:, 1], color="#e6e9ee", lw=6 if row else 4, zorder=1)
            ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=0.6, zorder=2)
            for lp in loops:
                q, mm = lp["query_frame"], lp["match_frame"]
                ax.plot([xy[q, 0], xy[mm, 0]], [xy[q, 1], xy[mm, 1]], color=RED, lw=0.9 if row else 0.6, zorder=3)
            ax.scatter(xy[kf, 0], xy[kf, 1], s=14 if row else 2.5, color=color if method == "msa" else "#6b7a90",
                       zorder=4, linewidths=0)
            style(ax)
            if row == 1:
                ax.set_xlim(zoom[0], zoom[1])
                ax.set_ylim(zoom[2], zoom[3])
        axes[0, col].add_patch(plt.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                                             fill=False, ec=ORANGE, lw=1.2, zorder=5))
        axes[0, col].set_title(f"{title}\n{m['keyframes']:,} keyframes · {m['loops']} loops · ATE {m['ate_opt_m']:.2f} m",
                               fontsize=12, color=NAVY if method == "msa" else INK)
    axes[1, 0].set_title("zoom: revisited crossroads", fontsize=10, color=ORANGE)
    axes[1, 1].set_title("zoom: revisited crossroads", fontsize=10, color=ORANGE)
    handles = [plt.Line2D([], [], color="#e6e9ee", lw=4, label="GT"),
               plt.Line2D([], [], color=INK, lw=1, label="optimised trajectory"),
               plt.Line2D([], [], marker="o", ls="", color=NAVY, label="keyframe"),
               plt.Line2D([], [], color=RED, lw=1, label="loop closure")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def demo_gif(res_dir, gt, out, step=45, fps=12):
    data = [load(res_dir, m) for m, _ in PAIR]
    n = len(gt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pad = 25
    lim = (gt[:, 0].min() - pad, gt[:, 0].max() + pad, gt[:, 1].min() - pad, gt[:, 1].max() + pad)
    arts = []
    for ax, (method, title), (m, xy, kf, loops) in zip(axes, PAIR, data):
        ax.plot(gt[:, 0], gt[:, 1], color="#eef0f3", lw=4, zorder=1)
        style(ax)
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_title(title, fontsize=13, color=NAVY if method == "msa" else INK, fontweight="bold")
        traj, = ax.plot([], [], color=INK, lw=0.7, zorder=2)
        pts = ax.scatter([], [], s=4, color=NAVY if method == "msa" else "#6b7a90", zorder=4, linewidths=0)
        cur = ax.scatter([], [], s=40, color=ORANGE, zorder=6)
        txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=13, family="monospace", color=INK)
        lp_lines = []
        arts.append((xy, np.sort(kf), loops, traj, pts, cur, txt, lp_lines, ax))

    frames = list(range(step, n, step)) + [n - 1] * 12

    def update(i):
        for xy, kf, loops, traj, pts, cur, txt, lp_lines, ax in arts:
            traj.set_data(xy[:i + 1, 0], xy[:i + 1, 1])
            k = kf[kf <= i]
            pts.set_offsets(xy[k, :2])
            cur.set_offsets(xy[i:i + 1, :2])
            done = [lp for lp in loops if lp["query_frame"] <= i]
            for lp in done[len(lp_lines):]:
                q, mm = lp["query_frame"], lp["match_frame"]
                ln, = ax.plot([xy[q, 0], xy[mm, 0]], [xy[q, 1], xy[mm, 1]], color=RED, lw=1.0, zorder=3)
                lp_lines.append(ln)
            txt.set_text(f"frame {i + 1:>4}/{n}\nkeyframes {len(k):>4}\nloops     {len(done):>4}")
        return []

    fig.tight_layout()
    FuncAnimation(fig, update, frames=frames, blit=False).save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def numbers(res_dir, out):
    rows = {}
    for method in ["all", "msa", "fixed_1", "fixed_2", "entropy", "spaciousness"]:
        p = os.path.join(res_dir, method, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            rows[method] = {k: m[k] for k in [
                "keyframes", "memory_raw_scans_MB", "time_sampling_s", "time_backend_s", "time_total_s",
                "loops", "loop_precision", "f1_max", "ate_odom_m", "ate_opt_m", "ate_improvement_pct"]}
    json.dump(rows, open(out, "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/kitti_00_4541")
    ap.add_argument("--kitti-root", default="/HDD/KITTI/odometry/dataset")
    ap.add_argument("--sequence", default="00")
    a = ap.parse_args()
    n = len(np.loadtxt(os.path.join(a.results, "all", "poses_opt.txt")))
    gt = gt_xy(a.kitti_root, a.sequence, n)
    global GT_XY
    GT_XY = gt
    os.makedirs("assets", exist_ok=True)
    numbers(a.results, "assets/kitti00_results.json")
    static_figure(a.results, gt, "assets/kitti00_traj.png")
    demo_gif(a.results, gt, "assets/kitti00_demo.gif")
    for f in ["assets/kitti00_traj.png", "assets/kitti00_demo.gif", "assets/kitti00_results.json"]:
        print(f, f"{os.path.getsize(f) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
