"""Keyframe sampling: the fixed middle block of the pipeline.

All samplers except `all` are the official implementations from
third_party/opt-key, called with the same arguments as its examples/pr_example.py.
Each wrapper turns them into one interface: step() returns the frame indices
that became keyframes at this step (MSA commits a batch whenever its sliding
window is optimised, so its keyframes arrive with a short delay).
"""

from typing import List

import numpy as np

from . import OPTKEY_DIR  # noqa: F401  (sets sys.path for the opt-key submodule)


def _homogeneous(points: np.ndarray) -> np.ndarray:
    """Official samplers receive scans as Nx4 [x, y, z, 1]."""
    scan = np.ones((points.shape[0], 4))
    scan[:, :3] = points[:, :3]
    return scan


class KeyframeSampler:
    name = "sampler"
    needs_scan = False

    def step(self, idx: int, pose: np.ndarray, descriptor: np.ndarray,
             points: np.ndarray = None) -> List[int]:
        raise NotImplementedError


class AllFrames(KeyframeSampler):
    """w/o MSA: every frame is a keyframe."""
    name = "all"

    def step(self, idx, pose, descriptor, points=None):
        return [idx]


class _OptKeySampler(KeyframeSampler):
    def __init__(self):
        self._committed = 0
        self._last = -1

    def _new_keyframes(self) -> List[int]:
        kfs = self.impl.keyframes.keyframes
        new = [kf.index for kf in kfs[self._committed:]]
        self._committed = len(kfs)
        # Only the latest keyframe's scan is ever read again (entropy/spaciousness);
        # drop older scans so memory stays bounded.
        for kf in kfs[:-1]:
            kf.pointcloud = None
        new = sorted(i for i in new if i > self._last)
        if new:
            self._last = new[-1]
        return new


class FixedDistanceSampler(_OptKeySampler):
    def __init__(self, distance: float):
        super().__init__()
        from scripts.FixedDistance import FixedDistanceOnline
        self.name = f"fixed_{distance:g}"
        self.impl = FixedDistanceOnline(dist_threshold=distance)

    def step(self, idx, pose, descriptor, points=None):
        self.impl.sample(pose=pose, descriptor=descriptor, index=idx)
        return self._new_keyframes()


class EntropySampler(_OptKeySampler):
    name = "entropy"
    needs_scan = True

    def __init__(self, entropy_threshold: float = 0.05, dist_threshold: float = 5.0):
        super().__init__()
        from scripts.Entropy import EntropyOnline
        self.impl = EntropyOnline(entropy_threshold=entropy_threshold, dist_threshold=dist_threshold)

    def step(self, idx, pose, descriptor, points=None):
        self.impl.sample(pose=pose, scan=_homogeneous(points), descriptor=descriptor, index=idx)
        return self._new_keyframes()


class SpaciousnessSampler(_OptKeySampler):
    name = "spaciousness"
    needs_scan = True

    def __init__(self, **kwargs):
        super().__init__()
        from scripts.Spaciousness import SpaciousnessOnline
        params = dict(alpha=0.85, beta=0.15, delta_min=1.0, delta_mid=2.0, delta_max=3.0,
                      theta_min=3.0, theta_mid=5.0, theta_max=10.0, queue_size=10)
        params.update(kwargs)
        self.impl = SpaciousnessOnline(**params)

    def step(self, idx, pose, descriptor, points=None):
        self.impl.sample(pose=pose, scan=_homogeneous(points), descriptor=descriptor, index=idx)
        return self._new_keyframes()


class MSASampler(_OptKeySampler):
    """Minimal Subset Approach (RMIP_Online in the official code)."""
    name = "msa"

    def __init__(self, window_size: int = 10, n_neighbours: int = 2,
                 alpha: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__()
        from scripts.RMIP import RMIP_Online
        self.impl = RMIP_Online(window_size=window_size, n_neighbours=n_neighbours,
                                alpha=alpha, beta=beta, verbose=False, **kwargs)

    def step(self, idx, pose, descriptor, points=None):
        # RMIP only stores the scan, it never reads it.
        self.impl.sample_rmip(pose=pose, scan=None, descriptor=descriptor, index=idx)
        return self._new_keyframes()


def make_sampler(method: str, params: dict = None) -> KeyframeSampler:
    params = dict(params or {})
    m = method.lower()
    if m == "all":
        return AllFrames()
    if m.startswith("fixed_"):
        return FixedDistanceSampler(float(m.split("_", 1)[1]))
    if m == "entropy":
        return EntropySampler(**params.get("entropy", {}))
    if m == "spaciousness":
        return SpaciousnessSampler(**params.get("spaciousness", {}))
    if m == "msa":
        return MSASampler(**params.get("msa", {}))
    raise ValueError(f"Unknown sampling method: {method}")
