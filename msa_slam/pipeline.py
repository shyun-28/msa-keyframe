"""Runs the pipeline in two stages.

Stage 1 (shared by all sampling methods): front-end odometry, descriptors and a
downsampled copy of every scan. The front-end gets no feedback from the back-end
(loosely coupled), so its output is identical for every method and is cached once.

Stage 2 (per method): frames are replayed in order; the sampler decides online
which frames become keyframes, and only those reach the back-end.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator

import numpy as np
from tqdm import tqdm

from .backend import make_backend
from .metrics import ate_rmse, complete_trajectory, loop_correctness, pr_metrics
from .descriptors import descriptor_ops
from .sampling import make_sampler


def _prefetch(fn, n: int, ahead: int = 16) -> Iterator:
    with ThreadPoolExecutor(4) as ex:
        futures = [ex.submit(fn, i) for i in range(min(ahead, n))]
        for i in range(n):
            if i + ahead < n:
                futures.append(ex.submit(fn, i + ahead))
            yield futures[i].result()
            futures[i] = None


# --- stage 1 ----------------------------------------------------------------
def slice_stage1(data: Dict, n: int) -> Dict:
    """First n frames of a longer stage-1 result. Valid because stage 1 is causal."""
    off = data["offsets"][:n + 1]
    out = dict(data)
    out.update(odom=data["odom"][:n], descs=data["descs"][:n], offsets=off, clouds=data["clouds"][:off[-1]])
    return out


def run_frontend(dataset, frontend, descriptor, n_frames: int, cloud_voxel: float, cache_path: str,
                 progress=None, extra_caches=()) -> Dict:
    for path in (cache_path, *extra_caches):
        if path and os.path.exists(path):
            data = dict(np.load(path))
            if len(data["odom"]) >= n_frames:
                print(f"[stage 1] loaded cache {path}")
                return slice_stage1(data, n_frames)
    import small_gicp

    odom = np.zeros((n_frames, 4, 4))
    descs = np.zeros((n_frames, descriptor.dim), dtype=np.float32)
    clouds, offsets = [], [0]
    t_front = t_desc = 0.0
    for i, points in enumerate(tqdm(_prefetch(dataset.scan, n_frames), total=n_frames, desc="[stage 1] KISS-ICP + OT")):
        t0 = time.perf_counter()
        if len(points) >= 10:
            odom[i] = frontend.step(points)
        else:  # (almost) empty scan, e.g. open water: hold the last pose
            odom[i] = odom[i - 1] if i else np.eye(4)
        t1 = time.perf_counter()
        descs[i] = descriptor(points) if len(points) else (descs[i - 1] if i else 0)
        t2 = time.perf_counter()
        t_front += t1 - t0
        t_desc += t2 - t1
        if len(points):
            ds = small_gicp.voxelgrid_sampling(points, cloud_voxel).points()[:, :3].astype(np.float32)
        else:
            ds = np.zeros((0, 3), dtype=np.float32)
        clouds.append(ds)
        offsets.append(offsets[-1] + len(ds))
        if progress and (i % 10 == 0 or i == n_frames - 1):
            progress(i + 1, n_frames)
    data = {"odom": odom, "descs": descs, "clouds": np.concatenate(clouds),
            "offsets": np.asarray(offsets), "t_frontend": t_front, "t_descriptor": t_desc}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **data)
    return data


# --- stage 2 ----------------------------------------------------------------
def run_method(method: str, dataset, stage1: Dict, n_frames: int, cfg: dict, out_dir: str,
               progress=None) -> Dict:
    sampler = make_sampler(method, cfg.get("sampling", {}))
    odom, descs = stage1["odom"], stage1["descs"]
    clouds, offsets = stage1["clouds"], stage1["offsets"]
    ops = descriptor_ops(cfg["descriptor"])
    backend_cfg = dict(cfg["backend"])
    if "loop_threshold" in cfg["descriptor"]:  # similarity scales differ per descriptor
        backend_cfg["sim_threshold"] = cfg["descriptor"]["loop_threshold"]
    backend = make_backend(backend_cfg, capacity=n_frames, descriptor_dim=descs.shape[1],
                           similarity=ops.similarity)
    gt = dataset.gt_poses[:n_frames] if dataset.gt_poses is not None else None
    delta = cfg.get("eval", {}).get("delta", 3.0)

    scans = _prefetch(dataset.scan, n_frames) if sampler.needs_scan else None
    keyframes, pr_sims, pr_dists = [], [], []
    t_sampling = t_backend = 0.0
    for i in tqdm(range(n_frames), desc=f"[stage 2] {method}"):
        points = next(scans) if scans is not None else None
        t0 = time.perf_counter()
        new = sampler.step(i, odom[i], ops.for_sampler(descs[i]), points)
        t1 = time.perf_counter()
        for k in new:
            backend.add_keyframe(k, odom[k], descs[k], clouds[offsets[k]:offsets[k + 1]])
        t2 = time.perf_counter()
        t_sampling += t1 - t0
        t_backend += t2 - t1
        keyframes.extend(new)
        if progress and (i % 20 == 0 or i == n_frames - 1):
            progress(i + 1, n_frames)

        # Place-recognition PR, official protocol: every frame queries the keyframe map.
        # Evaluation only, not timed.
        if gt is not None:
            best = backend.query(descs[i], i, 1)
            if best:
                sim, key = best[0]
                pr_sims.append(sim)
                pr_dists.append(float(np.linalg.norm(gt[i, :3, 3] - gt[backend.frames[key], :3, 3])))

    kf_poses = backend.optimized_poses()
    traj = complete_trajectory(n_frames, odom, kf_poses)
    mem = backend.memory_bytes()
    res = {
        "method": method,
        "frames": n_frames,
        "keyframes": len(keyframes),
        "keyframe_ratio": len(keyframes) / n_frames,
        "time_sampling_s": t_sampling,
        "time_backend_s": t_backend,
        "time_backend_breakdown_s": dict(backend.t),
        "time_total_s": t_sampling + t_backend,
        "loop_candidates_verified": backend.n_verified,
        "memory_raw_scans_MB": sum(dataset.raw_scan_bytes(k) for k in keyframes) / 1e6,
        "memory_backend_MB": (mem["descriptors"] + mem["clouds"]) / 1e6,
    }
    if gt is not None:
        ate_odom, ate_opt = ate_rmse(odom, gt), ate_rmse(traj, gt)
        pr = pr_metrics(pr_sims, pr_dists, delta)
        res.update({
            "ate_odom_m": ate_odom,
            "ate_opt_m": ate_opt,
            "ate_improvement_pct": 100.0 * (ate_odom - ate_opt) / ate_odom,
            "pr_auc": pr["auc"],
            "f1_max": pr["f1_max"],
            **loop_correctness(backend.loops, gt, max_trans_err=delta),
        })

    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "poses_opt.txt"), traj[:, :3, :].reshape(n_frames, 12), fmt="%.6e")
    np.savetxt(os.path.join(out_dir, "keyframes.txt"), np.asarray(keyframes, dtype=int), fmt="%d")
    with open(os.path.join(out_dir, "loops.json"), "w") as f:
        json.dump(backend.loops, f, indent=1)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    res["_traj"], res["_keyframes"], res["_loops"] = traj, keyframes, backend.loops
    return res
