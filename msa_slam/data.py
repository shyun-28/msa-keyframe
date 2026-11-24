"""Dataset loaders. Each loader yields Nx3 float64 scans and (optionally) GT poses
expressed in the LiDAR frame of the first scan."""

import os
from typing import Optional

import numpy as np


class KittiOdometry:
    """KITTI odometry sequence, read with KISS-ICP's own KITTI loader so the
    vertical-angle scan correction and the camera->LiDAR GT conversion match KISS-ICP."""

    def __init__(self, root: str, sequence: str):
        from kiss_icp.datasets.kitti import KITTIOdometryDataset

        self.name = f"kitti_{str(sequence).zfill(2)}"
        self._ds = KITTIOdometryDataset(root, sequence)
        self.gt_poses: Optional[np.ndarray] = getattr(self._ds, "gt_poses", None)
        self.scan_files = self._ds.scan_files

    def __len__(self) -> int:
        return len(self.scan_files)

    def scan(self, idx: int) -> np.ndarray:
        return self._ds.read_point_cloud(self.scan_files[idx])

    def raw_scan_bytes(self, idx: int) -> int:
        """Size of the scan as stored on disk (float32 x,y,z,intensity)."""
        return os.path.getsize(self.scan_files[idx])


def make_dataset(cfg: dict):
    name = cfg["name"].lower()
    if name == "kitti":
        return KittiOdometry(cfg["root"], cfg["sequence"])
    raise ValueError(f"Unknown dataset: {cfg['name']}")
