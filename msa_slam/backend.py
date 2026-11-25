"""Back-end: place recognition -> loop verification -> pose graph optimisation.
Swappable through the Backend interface.

LoopClosureBackend:
  6. Place recognition : nearest descriptors (OverlapTransformer) among keyframes
                         at least `min_frame_gap` frames old.
  7. Loop verification : candidates farther than `max_estimate_distance` from the
                         current estimate are skipped; the rest go through small_gicp
                         GICP, coarse-to-fine, accepted by inlier ratio.
  8. PGO               : GTSAM iSAM2, odometry + robust (Cauchy) loop factors.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Backend(ABC):
    name = "backend"

    @abstractmethod
    def add_keyframe(self, frame_idx: int, odom_pose: np.ndarray,
                     descriptor: np.ndarray, cloud: np.ndarray) -> None:
        ...

    @abstractmethod
    def optimized_poses(self) -> Dict[int, np.ndarray]:
        """frame index -> optimised 4x4 pose, for every keyframe."""


def _noise(rot_sigma: float, trans_sigma: float):
    import gtsam
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([rot_sigma] * 3 + [trans_sigma] * 3))


class LoopClosureBackend(Backend):
    name = "loop_closure_isam2"

    def __init__(self, capacity: int, descriptor_dim: int = 256,
                 sim_threshold: float = 0.7, top_k: int = 3, min_frame_gap: int = 100,
                 coarse_voxel: float = 1.0, fine_voxel: float = 0.5,
                 coarse_max_corr: float = 3.0, fine_max_corr: float = 1.0,
                 min_inlier_ratio: float = 0.5, max_estimate_distance: float = 30.0,
                 num_threads: int = 8,
                 odom_sigmas=(0.01, 0.1), loop_sigmas=(0.05, 0.3), cauchy_k: float = 1.0):
        import gtsam

        self.gtsam = gtsam
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.min_frame_gap = min_frame_gap
        self.coarse_voxel, self.fine_voxel = coarse_voxel, fine_voxel
        self.coarse_max_corr, self.fine_max_corr = coarse_max_corr, fine_max_corr
        self.min_inlier_ratio = min_inlier_ratio
        self.max_estimate_distance = max_estimate_distance
        self.num_threads = num_threads

        self.odom_noise = _noise(*odom_sigmas)
        self.loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(cauchy_k), _noise(*loop_sigmas))
        self.prior_noise = _noise(1e-6, 1e-6)
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)

        # keyframe database
        self.frames: List[int] = []
        self.descs = np.zeros((capacity, descriptor_dim), dtype=np.float32)
        self.clouds: List[np.ndarray] = []
        self.odom: List[np.ndarray] = []

        self.loops: List[dict] = []
        self.n_verified = 0
        self.t = {"place_recognition": 0.0, "verification": 0.0, "pgo": 0.0}

    # --- database -----------------------------------------------------------
    def query(self, descriptor: np.ndarray, frame_idx: int, k: int):
        """Top-k (similarity, keyframe key) among keyframes old enough to be a loop."""
        n = int(np.searchsorted(self.frames, frame_idx - self.min_frame_gap, side="right"))
        if n == 0:
            return []
        sims = 1.0 / (1.0 + np.linalg.norm(self.descs[:n] - descriptor, axis=1))
        k = min(k, n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(float(sims[j]), int(j)) for j in top]

    # --- verification -------------------------------------------------------
    def _register(self, target: np.ndarray, source: np.ndarray, init: np.ndarray):
        import small_gicp
        tgt, tgt_tree = small_gicp.preprocess_points(target.astype(np.float64), self.coarse_voxel,
                                                     num_threads=self.num_threads)
        src, _ = small_gicp.preprocess_points(source.astype(np.float64), self.coarse_voxel,
                                              num_threads=self.num_threads)
        coarse = small_gicp.align(tgt, src, tgt_tree, init, "GICP",
                                  max_correspondence_distance=self.coarse_max_corr,
                                  num_threads=self.num_threads, max_iterations=30)
        tgt, tgt_tree = small_gicp.preprocess_points(target.astype(np.float64), self.fine_voxel,
                                                     num_threads=self.num_threads)
        src, _ = small_gicp.preprocess_points(source.astype(np.float64), self.fine_voxel,
                                              num_threads=self.num_threads)
        fine = small_gicp.align(tgt, src, tgt_tree, coarse.T_target_source, "GICP",
                                max_correspondence_distance=self.fine_max_corr,
                                num_threads=self.num_threads, max_iterations=30)
        return fine, fine.num_inliers / max(src.size(), 1)

    def _verify(self, cand_key: int, cur_cloud: np.ndarray, cur_est: np.ndarray):
        if len(cur_cloud) < 50 or len(self.clouds[cand_key]) < 50:
            return None  # too few points to register reliably
        cand_est = self.isam.calculateEstimatePose3(cand_key).matrix()
        best = None
        for init in (np.linalg.inv(cand_est) @ cur_est, np.eye(4)):
            res, ratio = self._register(self.clouds[cand_key], cur_cloud, init)
            if res.converged and (best is None or ratio > best[1]):
                best = (res.T_target_source.copy(), ratio)
        return best

    # --- main entry ---------------------------------------------------------
    def add_keyframe(self, frame_idx, odom_pose, descriptor, cloud):
        gtsam = self.gtsam
        key = len(self.frames)
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()

        if key == 0:
            cur_est = odom_pose
            graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(odom_pose), self.prior_noise))
        else:
            rel = np.linalg.inv(self.odom[-1]) @ odom_pose
            cur_est = self.isam.calculateEstimatePose3(key - 1).matrix() @ rel
            graph.add(gtsam.BetweenFactorPose3(key - 1, key, gtsam.Pose3(rel), self.odom_noise))
        values.insert(key, gtsam.Pose3(cur_est))

        # 6. place recognition (against the database before this keyframe is added)
        t0 = time.perf_counter()
        candidates = [c for c in self.query(descriptor, frame_idx, self.top_k)
                      if c[0] >= self.sim_threshold]
        t1 = time.perf_counter()

        # 7. verification
        loop_added = False
        for sim, cand in candidates:
            # Look-alike streets far away pass both the descriptor and GICP, so only
            # verify candidates whose current estimate is near ours (drift budget).
            cand_pos = self.isam.calculateEstimatePose3(cand).translation()
            if np.linalg.norm(np.asarray(cand_pos) - cur_est[:3, 3]) > self.max_estimate_distance:
                continue
            self.n_verified += 1
            best = self._verify(cand, cloud, cur_est)
            if best is not None and best[1] >= self.min_inlier_ratio:
                T, ratio = best
                graph.add(gtsam.BetweenFactorPose3(cand, key, gtsam.Pose3(T), self.loop_noise))
                self.loops.append({"query_frame": int(frame_idx), "match_frame": int(self.frames[cand]),
                                   "similarity": sim, "inlier_ratio": float(ratio),
                                   "T_match_query": T.tolist()})
                loop_added = True
                break  # one verified loop per keyframe is enough
        t2 = time.perf_counter()

        # 8. PGO
        self.isam.update(graph, values)
        if loop_added:
            for _ in range(3):
                self.isam.update()
        t3 = time.perf_counter()

        self.t["place_recognition"] += t1 - t0
        self.t["verification"] += t2 - t1
        self.t["pgo"] += t3 - t2

        self.frames.append(int(frame_idx))
        self.descs[key] = descriptor
        self.clouds.append(cloud.astype(np.float32))
        self.odom.append(odom_pose)

    def optimized_poses(self):
        est = self.isam.calculateEstimate()
        return {f: est.atPose3(k).matrix() for k, f in enumerate(self.frames)}

    def memory_bytes(self) -> Dict[str, int]:
        n = len(self.frames)
        return {"descriptors": int(self.descs[:n].nbytes),
                "clouds": int(sum(c.nbytes for c in self.clouds))}


def make_backend(cfg: dict, capacity: int, descriptor_dim: int) -> Backend:
    cfg = dict(cfg)
    name = cfg.pop("name").lower()
    if name == "loop_closure_isam2":
        return LoopClosureBackend(capacity=capacity, descriptor_dim=descriptor_dim, **cfg)
    raise ValueError(f"Unknown back-end: {name}")
