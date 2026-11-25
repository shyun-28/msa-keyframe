"""Front-end: LiDAR odometry. Swappable through the Frontend interface."""

from abc import ABC, abstractmethod

import numpy as np


class Frontend(ABC):
    name = "frontend"

    @abstractmethod
    def step(self, points: np.ndarray) -> np.ndarray:
        """Register one scan (Nx3) and return its 4x4 pose in the odometry frame."""


class KissICPFrontend(Frontend):
    name = "kiss_icp"

    def __init__(self, max_range: float = 100.0, min_range: float = 0.0,
                 deskew: bool = False, voxel_size: float = None):
        from kiss_icp.config import load_config
        from kiss_icp.kiss_icp import KissICP

        config = load_config(None)
        config.data.max_range = max_range
        config.data.min_range = min_range
        config.data.deskew = deskew
        config.mapping.voxel_size = voxel_size if voxel_size else max_range / 100.0
        self._odom = KissICP(config)

    def step(self, points: np.ndarray) -> np.ndarray:
        self._odom.register_frame(points, np.array([]))
        return self._odom.last_pose.copy()


class GicpOdometryFrontend(Frontend):
    """Scan-to-scan GICP odometry (small_gicp) for sparse scan streams, e.g. the SLAM
    keyframes DiTer ships (~1 m / 1-2 s apart, up to ~30 deg of turn between scans),
    where KISS-ICP's assumption of dense 10 Hz input does not hold.
    Each scan is registered to the previous one from several yaw guesses
    (`yaw_guesses_deg`); the result with the most inlier correspondences wins."""
    name = "gicp"

    def __init__(self, voxel_size: float = 0.5, max_correspondence: float = 2.0,
                 max_range: float = 100.0, yaw_guesses_deg=(0, -15, 15, -30, 30),
                 num_threads: int = 8):
        self.voxel_size, self.max_corr, self.max_range = voxel_size, max_correspondence, max_range
        self.yaws, self.num_threads = [np.radians(y) for y in yaw_guesses_deg], num_threads
        self.pose, self.prev = np.eye(4), None

    def step(self, points: np.ndarray) -> np.ndarray:
        import small_gicp
        pts = points[np.linalg.norm(points, axis=1) < self.max_range].astype(np.float64)
        if self.prev is not None and len(pts) >= 10:
            best = None
            for yaw in self.yaws:
                guess = np.eye(4)
                guess[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
                res = small_gicp.align(self.prev, pts, init_T_target_source=guess, registration_type="GICP",
                                       downsampling_resolution=self.voxel_size,
                                       max_correspondence_distance=self.max_corr, num_threads=self.num_threads)
                if best is None or res.num_inliers > best.num_inliers:
                    best = res
            self.pose = self.pose @ best.T_target_source
        if len(pts) >= 10:
            self.prev = pts
        return self.pose.copy()


def make_frontend(cfg: dict) -> Frontend:
    cfg = dict(cfg)
    name = cfg.pop("name").lower()
    if name == "kiss_icp":
        return KissICPFrontend(**cfg)
    if name == "gicp":
        return GicpOdometryFrontend(**cfg)
    raise ValueError(f"Unknown front-end: {name}")
