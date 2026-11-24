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


def make_frontend(cfg: dict) -> Frontend:
    cfg = dict(cfg)
    name = cfg.pop("name").lower()
    if name == "kiss_icp":
        return KissICPFrontend(**cfg)
    raise ValueError(f"Unknown front-end: {name}")
