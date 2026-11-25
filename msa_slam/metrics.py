"""Evaluation: ATE, place-recognition precision/recall, loop correctness."""

from typing import Dict, List

import numpy as np


def complete_trajectory(n_frames: int, odom: np.ndarray, kf_poses: Dict[int, np.ndarray]) -> np.ndarray:
    """Pose for every frame: optimised pose of the latest keyframe at or before it,
    chained with the odometry increment from that keyframe. Frames before the first
    keyframe keep their odometry pose. Makes ATE comparable across sampling methods."""
    out = odom.copy()
    kf = sorted(kf_poses)
    j = -1
    for i in range(n_frames):
        while j + 1 < len(kf) and kf[j + 1] <= i:
            j += 1
        if j >= 0:
            k = kf[j]
            out[i] = kf_poses[k] @ np.linalg.inv(odom[k]) @ odom[i]
    return out


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform T minimising ||dst - T(src)|| (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, mu_d - R @ mu_s
    return T


def ate_rmse(est: np.ndarray, gt: np.ndarray) -> float:
    p, q = est[:, :3, 3], gt[:, :3, 3]
    T = umeyama_se3(p, q)
    aligned = (T[:3, :3] @ p.T).T + T[:3, 3]
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))


def pr_metrics(similarities: List[float], distances: List[float], delta: float) -> Dict[str, float]:
    """Same thresholds and TP/FP/FN definition as the official MSA evaluation
    (opt-key examples/pr_example.py: compute_pr_metrics_from_arrays)."""
    sim, dist = np.asarray(similarities), np.asarray(distances)
    if not np.any(dist <= delta):  # no revisit within delta: precision/recall undefined
        return {"auc": float("nan"), "f1_max": float("nan"), "precisions": [], "recalls": []}
    precisions, recalls, f1s = [], [], []
    for theta in np.arange(0, 100, 1):
        thr = theta / 100
        tp = np.sum((sim >= thr) & (dist <= delta))
        fp = np.sum((sim > thr) & (dist > delta))
        fn = np.sum((sim < thr) & (dist <= delta))
        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return {"auc": float(-trapz(precisions, recalls)), "f1_max": float(np.max(f1s)),
            "precisions": precisions, "recalls": recalls}


def loop_correctness(loops: List[dict], gt: np.ndarray, max_trans_err: float = 3.0) -> Dict[str, float]:
    """A verified loop is correct if its relative pose agrees with GT within max_trans_err metres.
    KITTI GT itself is off by up to ~2.5 m at some revisits (GICP started from the GT
    relative pose converges to our loop result), so a tighter bound would count
    correct loops as wrong."""
    errs = []
    for lp in loops:
        q, m = lp["query_frame"], lp["match_frame"]
        T_gt = np.linalg.inv(gt[m]) @ gt[q]
        err = float(np.linalg.norm(np.asarray(lp["T_match_query"])[:3, 3] - T_gt[:3, 3]))
        lp["gt_trans_error"] = err
        errs.append(err)
    errs = np.asarray(errs)
    n = len(loops)
    good = int(np.sum(errs <= max_trans_err))
    return {"loops": n, "correct_loops": good,
            "loop_precision": float(good / n) if n else float("nan"),
            "loop_gt_error_median_m": float(np.median(errs)) if n else float("nan")}
