"""Global place descriptors shared by MSA (feature space) and the back-end
(place recognition)."""

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


def make_descriptor(cfg: dict):
    cfg = dict(cfg)
    name = cfg.pop("name").lower()
    if name == "overlap_transformer":
        return OverlapTransformerDescriptor(**cfg)
    raise ValueError(f"Unknown descriptor: {name}")
