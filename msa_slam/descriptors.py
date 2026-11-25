"""Global place descriptors shared by MSA (feature space) and the back-end
(place recognition). Each descriptor provides:
  __call__(points)      -> flat float32 vector (stored per frame in the stage-1 cache)
  for_sampler(vector)   -> the shape the official MSA code receives
  similarity(q, db)     -> similarity of query q to every row of db (higher = more alike)
The paper evaluates MSA with OverlapTransformer and Scan Context (Sec. V)."""

import numpy as np

from . import OT_DIR  # noqa: F401  (sets sys.path for the OverlapTransformer submodule)


class OverlapTransformerDescriptor:
    """OverlapTransformer (Ma et al., RA-L 2022) with the authors' KITTI weights.
    Output: L2-normalised 256-d vector. Same preprocessing as the official MSA code
    (range image 64x900 of the homogeneous scan)."""

    name = "overlap_transformer"
    dim = 256

    def __init__(self, weights: str, device: str = None):
        import torch
        from modules.overlap_transformer import featureExtracter
        from tools.utils.utils import range_projection

        self._torch = torch
        self._range_projection = range_projection
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = featureExtracter(channels=1, use_transformer=True)
        checkpoint = torch.load(weights, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(self.device).eval()

    def __call__(self, points: np.ndarray) -> np.ndarray:
        scan = np.ones((points.shape[0], 4), dtype=np.float32)
        scan[:, :3] = points[:, :3]
        proj_range = self._range_projection(scan)[0]
        x = self._torch.from_numpy(proj_range).float()[None, None].to(self.device)
        with self._torch.no_grad():
            return self.model(x).cpu().numpy().reshape(-1).astype(np.float32)

    @staticmethod
    def for_sampler(d: np.ndarray) -> np.ndarray:
        return d

    @staticmethod
    def similarity(q: np.ndarray, db: np.ndarray) -> np.ndarray:
        """1 / (1 + L2), as in the official MSA code (RMIP.calculate_similarity)."""
        return 1.0 / (1.0 + np.linalg.norm(db - q, axis=1))


class ScanContextDescriptor:
    """Scan Context (Kim & Kim, IROS 2018 / T-RO 2022), as in the official MSA code
    (opt-key optkey_utils/SC_Handler.py) with the parameters of its PR example:
    0.5 m voxel downsampling, 20 rings x 60 sectors over 80 m, each cell = max point
    height + lidar_height (empty cells 0).
    The official ptcloud2sc loops over points in Python; this is the same computation
    vectorised. Downsampling uses small_gicp's voxel grid instead of Open3D's (both
    average the points in a 0.5 m voxel; the grids' origins differ).
    Similarity: max over the 60 sector shifts of the cosine similarity (official
    calculate_similarity), so a revisit in the opposite direction still matches."""

    name = "scan_context"

    def __init__(self, downcell_size: float = 0.5, lidar_height: float = 2.0,
                 sector_res: int = 60, ring_res: int = 20, max_length: float = 80.0):
        self.downcell_size, self.lidar_height = downcell_size, lidar_height
        self.sector_res, self.ring_res, self.max_length = sector_res, ring_res, max_length
        self.dim = ring_res * sector_res

    def scan_context(self, pts: np.ndarray) -> np.ndarray:
        """ptcloud2sc of the official code on already-downsampled points."""
        x, y, z = (np.asarray(pts[:, k], dtype=np.float64) for k in range(3))
        x = np.where(x == 0.0, 0.001, x)
        y = np.where(y == 0.0, 0.001, y)
        theta = np.degrees(np.arctan2(y, x)) % 360.0  # == official quadrant formula
        ring = np.minimum((np.hypot(x, y) // (self.max_length / self.ring_res)).astype(int), self.ring_res - 1)
        sector = (theta // (360.0 / self.sector_res)).astype(int) % self.sector_res
        sc = np.zeros((self.ring_res, self.sector_res))
        np.maximum.at(sc, (ring, sector), z + self.lidar_height)  # zeros = empty-cell floor
        return sc

    def __call__(self, points: np.ndarray) -> np.ndarray:
        import small_gicp
        if len(points) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        down = small_gicp.voxelgrid_sampling(np.asarray(points[:, :3], dtype=np.float64),
                                             self.downcell_size).points()[:, :3]
        return self.scan_context(down).reshape(-1).astype(np.float32)

    def for_sampler(self, d: np.ndarray) -> np.ndarray:
        return d.reshape(self.ring_res, self.sector_res)  # the official code passes 20x60

    def similarity(self, q: np.ndarray, db: np.ndarray) -> np.ndarray:
        q = q.reshape(self.ring_res, self.sector_res).astype(np.float64)
        D = db.reshape(len(db), self.ring_res, self.sector_res).astype(np.float64)
        # cosine of every column shift of q against every db entry, via circular
        # cross-correlation along the sector axis (FFT), then max over shifts
        corr = np.fft.irfft(np.conj(np.fft.rfft(q, axis=-1))[None] * np.fft.rfft(D, axis=-1),
                            n=self.sector_res, axis=-1).sum(axis=1)
        denom = np.linalg.norm(q) * np.linalg.norm(D.reshape(len(D), -1), axis=1)
        sims = np.where(denom > 0, corr.max(axis=1) / np.where(denom > 0, denom, 1), 0.0)
        return sims


DESCRIPTORS = {"overlap_transformer": OverlapTransformerDescriptor, "scan_context": ScanContextDescriptor}


def make_descriptor(cfg: dict):
    cfg = dict(cfg)
    cfg.pop("loop_threshold", None)  # back-end setting, see pipeline.run_method
    name = cfg.pop("name").lower()
    if name not in DESCRIPTORS:
        raise ValueError(f"Unknown descriptor: {name}")
    return DESCRIPTORS[name](**cfg)


def descriptor_ops(cfg: dict):
    """for_sampler / similarity / dim without loading model weights (stage 2 runs from
    the stage-1 cache and never needs the network)."""
    name = cfg["name"].lower()
    if name == "overlap_transformer":
        return OverlapTransformerDescriptor
    params = {k: v for k, v in cfg.items() if k not in ("name", "loop_threshold")}
    return DESCRIPTORS[name](**params)
