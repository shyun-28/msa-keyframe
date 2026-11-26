"""Turn SeePerSea sequences into scan folders the pipeline (and Try It Yourself) can read.

    python3 tools/prepare_seepersea.py /HDD/SeePerSea/target_folder/separate /HDD/SeePerSea/msa_ready

For every sequence folder (point_clouds/*.pcd, timestamp/timestamp.txt, motion_data/gps.csv)
this writes <out>/<sequence>/ with
  - symlinks to the .pcd scans (no copy),
  - poses.txt: KITTI-format poses from GPS. Latitude/longitude are converted to a local
    east-north frame (metres) around the first fix and interpolated to each scan time.
    GPS has no attitude, so the rotation is a yaw along the direction of travel
    (kept from the last fix with motion); only the translation is reliable, i.e.
    t. ATE and F1-max are meaningful, R. ATE is not. Altitude is not logged (z = 0).
Scans without a valid timestamp take the neighbouring scan's time. Some sequences
(Sunapee) log GPS on a clock ~20 days off the LiDAR clock; when the two time ranges do
not overlap, the GPS log is shifted so both start together (both cover the same ~60 s
clip), and the sequence is flagged in the output.
"""

import argparse
import os

import numpy as np

EARTH_R = 6378137.0


def load_gps(path):
    g = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)  # timestamp, lat, lon
    g = g[np.argsort(g[:, 0])]
    lat0, lon0 = np.radians(g[0, 1]), np.radians(g[0, 2])
    east = EARTH_R * (np.radians(g[:, 2]) - lon0) * np.cos(lat0)
    north = EARTH_R * (np.radians(g[:, 1]) - lat0)
    return g[:, 0], np.stack([east, north], axis=1)


def load_scan_times(path, n):
    t = np.full(n, np.nan)
    for line in open(path):
        i, ts = line.strip().split(",")
        i, ts = int(i), float(ts)
        if 0 <= i < n and ts > 0:
            t[i] = ts
    ok = np.flatnonzero(np.isfinite(t))
    if len(ok) == 0:
        raise ValueError("no valid scan timestamps")
    return np.interp(np.arange(n), ok, t[ok])  # fill invalid entries from neighbours


def poses_from_gps(scan_t, gps_t, gps_xy, min_step=0.3):
    x = np.interp(scan_t, gps_t, gps_xy[:, 0])
    y = np.interp(scan_t, gps_t, gps_xy[:, 1])
    # heading from motion between GPS fixes, held while (nearly) stationary
    d = np.diff(gps_xy, axis=0)
    moving = np.linalg.norm(d, axis=1) > min_step
    yaw_fix = np.full(len(gps_t), np.nan)
    yaw_fix[1:][moving] = np.arctan2(d[moving, 1], d[moving, 0])
    last = 0.0
    for i in range(len(yaw_fix)):
        last = yaw_fix[i] if np.isfinite(yaw_fix[i]) else last
        yaw_fix[i] = last
    yaw = np.unwrap(yaw_fix)
    yaw = np.interp(scan_t, gps_t, yaw)
    P = np.tile(np.eye(4), (len(scan_t), 1, 1))
    c, s = np.cos(yaw), np.sin(yaw)
    P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1] = c, -s, s, c
    P[:, 0, 3], P[:, 1, 3] = x, y
    return P


def prepare(seq_dir, out_dir):
    pcd_dir = os.path.join(seq_dir, "point_clouds")
    scans = sorted(f for f in os.listdir(pcd_dir) if f.endswith(".pcd"))
    scan_t = load_scan_times(os.path.join(seq_dir, "timestamp", "timestamp.txt"), len(scans))
    gps_t, gps_xy = load_gps(os.path.join(seq_dir, "motion_data", "gps.csv"))
    overlap = min(scan_t[-1], gps_t[-1]) - max(scan_t[0], gps_t[0])
    shifted = overlap < 0.5 * (scan_t[-1] - scan_t[0])
    if shifted:  # clocks disagree: align the starts
        gps_t = gps_t - gps_t[0] + scan_t[0]
    P = poses_from_gps(scan_t, gps_t, gps_xy)
    os.makedirs(out_dir, exist_ok=True)
    for f in scans:
        dst = os.path.join(out_dir, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(os.path.abspath(pcd_dir), f), dst)
    np.savetxt(os.path.join(out_dir, "poses.txt"), P[:, :3, :4].reshape(len(P), 12), fmt="%.6f")
    path_len = np.linalg.norm(np.diff(P[:, :2, 3], axis=0), axis=1).sum()
    return len(scans), path_len, scan_t[-1] - scan_t[0], shifted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="folder holding SeePerSea sequence folders (e.g. target_folder/separate)")
    ap.add_argument("out", help="output folder, one sub-folder per sequence")
    a = ap.parse_args()
    for name in sorted(os.listdir(a.src)):
        seq = os.path.join(a.src, name)
        if not os.path.isdir(os.path.join(seq, "point_clouds")):
            continue
        try:
            n, length, dur, shifted = prepare(seq, os.path.join(a.out, name))
            note = "  (GPS clock off, starts aligned)" if shifted else ""
            print(f"{name:32s} {n:4d} scans  {dur:5.1f} s  path {length:6.1f} m{note}")
        except (OSError, ValueError) as e:
            print(f"{name:32s} skipped: {e}")


if __name__ == "__main__":
    main()
