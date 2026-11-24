"""Modular LiDAR SLAM pipeline for comparing keyframe sampling methods.

Front-end (odometry) -> Keyframe sampling (MSA, fixed, entropy, spaciousness, all)
-> Back-end (place recognition, loop verification, pose graph optimization).

The sampling methods come unchanged from the official MSA code
(third_party/opt-key, MIT). The OverlapTransformer model comes from
third_party/OverlapTransformer (GPL-3) and is loaded from the submodule, not copied.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTKEY_DIR = os.path.join(ROOT, "third_party", "opt-key")
OT_DIR = os.path.join(ROOT, "third_party", "OverlapTransformer")

for _p in (OPTKEY_DIR, OT_DIR, os.path.join(OT_DIR, "tools")):
    if _p not in sys.path:
        sys.path.append(_p)
