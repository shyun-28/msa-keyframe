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


SCAN_EXTS = (".bin", ".pcd")


def read_pcd(path: str) -> np.ndarray:
    """x, y, z of a PCD file (DATA ascii or binary; binary_compressed is not supported)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line")
            parts = line.decode("ascii", "ignore").strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        fields = header["FIELDS"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        n = int(header["POINTS"][0])
        kind = header["DATA"][0].lower()
        if kind == "ascii":
            cols = []
            for fld, c in zip(fields, counts):
                cols += [fld] if c == 1 else [f"{fld}_{k}" for k in range(c)]
            data = np.loadtxt(f, dtype=np.float64, ndmin=2)
            if data.size == 0:
                return np.zeros((0, 3))
            idx = [cols.index(a) for a in ("x", "y", "z")]
            pts = data[:, idx]
        elif kind == "binary":
            sizes = [int(s) for s in header["SIZE"]]
            types = header["TYPE"]
            np_t = {("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
                    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("I", 8): "<i8", ("U", 8): "<u8"}
            dtype = np.dtype([(fld, np_t[(t, s)], (c,)) if c > 1 else (fld, np_t[(t, s)])
                              for fld, s, t, c in zip(fields, sizes, types, counts)])
            rec = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
            pts = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
        else:
            raise ValueError(f"{path}: PCD DATA {kind} is not supported (use ascii or binary)")
    return pts[np.isfinite(pts).all(axis=1)]


class FolderScans:
    """A folder of scans in file-name order: KITTI-style .bin (float32 x,y,z,i) or .pcd.
    Optional GT: poses.txt (KITTI format, 12 numbers per line, one line per scan).
    If calib.txt with a `Tr:` line is present, poses are converted from the KITTI
    camera frame to the LiDAR frame, as KISS-ICP does."""

    def __init__(self, folder: str, name: str = None):
        self.name = name or os.path.basename(os.path.normpath(folder))
        self.scan_files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                                 if f.lower().endswith(SCAN_EXTS))
        if not self.scan_files:
            raise ValueError("no .bin or .pcd scans found")
        self.gt_poses = None
        pose_file = os.path.join(folder, "poses.txt")
        if os.path.exists(pose_file):
            p = np.loadtxt(pose_file, ndmin=2)
            if p.shape[1] == 12 and len(p) >= len(self.scan_files):
                P = np.tile(np.eye(4), (len(p), 1, 1))
                P[:, :3, :4] = p.reshape(-1, 3, 4)
                Tr = self._read_tr(os.path.join(folder, "calib.txt"))
                if Tr is not None:
                    P = np.linalg.inv(Tr) @ P @ Tr
                self.gt_poses = P[:len(self.scan_files)]

    @staticmethod
    def _read_tr(path):
        if not os.path.exists(path):
            return None
        for line in open(path):
            if line.startswith("Tr:"):
                T = np.eye(4)
                T[:3, :4] = np.array(line.split()[1:13], dtype=float).reshape(3, 4)
                return T
        return None

    def __len__(self) -> int:
        return len(self.scan_files)

    def scan(self, idx: int) -> np.ndarray:
        f = self.scan_files[idx]
        if f.lower().endswith(".bin"):
            return np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        return read_pcd(f)

    def raw_scan_bytes(self, idx: int) -> int:
        return os.path.getsize(self.scan_files[idx])


def make_dataset(cfg: dict):
    name = cfg["name"].lower()
    if name == "kitti":
        return KittiOdometry(cfg["root"], cfg["sequence"])
    if name == "folder":
        return FolderScans(cfg["root"], cfg.get("label"))
    raise ValueError(f"Unknown dataset: {cfg['name']}")
