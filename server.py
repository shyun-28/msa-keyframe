"""Local demo server for the project page's "Try It Yourself" section.

    .venv/bin/python server.py            # then open http://localhost:8000

Serves index.html and assets/, accepts LiDAR scans uploaded from the page or
reads a scan folder on this PC directly (no upload, no size cap), runs the same pipeline as run.py (KISS-ICP -> sampling -> OverlapTransformer /
small_gicp / iSAM2) and returns metrics, trajectories, keyframes and loops.
Binds to 127.0.0.1 only. Jobs run one at a time.
"""

import argparse
import glob
import hashlib
import json
import mimetypes
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

from msa_slam import ROOT
from msa_slam.data import SCAN_EXTS, FolderScans, find_sidecars
from msa_slam.descriptors import make_descriptor
from msa_slam.frontend import make_frontend
from msa_slam.metrics import umeyama_se3
from msa_slam.pipeline import run_frontend, run_method

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "kitti00.yaml")))
UPLOAD_DIR = os.path.join(ROOT, "results", "uploads")
DEMO_DIR = os.path.join(ROOT, "results", "demo")
LOCAL_CACHE_DIR = os.path.join(ROOT, "results", "local_cache")
BROWSE_START = "/HDD" if os.path.isdir("/HDD") else os.path.expanduser("~")
ALLOWED_METHODS = {"all", "msa", "fixed_1", "fixed_2", "fixed_3", "entropy", "spaciousness"}
UPLOAD_NAMES = re.compile(r"^[\w.\-]+\.(bin|pcd)$|^(poses|calib)\.txt$", re.I)
MAX_FRAMES = 6000

# KITTI sequences on this PC, offered as one-click shortcuts in the folder browser
KITTI00_CACHE = os.path.join(ROOT, "results", "kitti_00_4541", "stage1_cache.npz")  # from run.py
DATASETS = yaml.safe_load(open(os.path.join(ROOT, "configs", "demo_datasets.yaml")))


def dataset_for(path):
    """The demo_datasets.yaml group a folder belongs to (by root prefix), or {}."""
    for d in DATASETS:
        root = os.path.realpath(d["root"])
        if path == root or path.startswith(root + os.sep):
            return d
    return {}


def frontend_cfg(path):
    return dataset_for(path).get("frontend") or CFG["frontend"]


def presets():
    """Dataset shortcuts for the page, grouped, from configs/demo_datasets.yaml."""
    out = []
    for d in DATASETS:
        items = []
        for path in sorted(glob.glob(os.path.join(d["root"], d["pattern"]))):
            if os.path.isdir(path) and scan_folder_info(path)["scans"]:
                rel = os.path.relpath(path, d["root"]).split(os.sep)
                items.append({"name": d.get("label", "{0}").format(*rel), "path": os.path.realpath(path)})
        if items:
            out.append({"group": d["group"], "items": items})
    return out


def scan_folder_info(path):
    """Scans and GT files found in a folder (no recursion)."""
    n, size = 0, 0
    with os.scandir(path) as it:
        for e in it:
            if e.name.lower().endswith(SCAN_EXTS) and e.is_file():
                n += 1
                size += e.stat().st_size
    pose, calib = find_sidecars(path) if n else (None, None)
    return {"scans": n, "mb": round(size / 1048576, 1), "poses": pose, "calib": calib}


def browse(path):
    path = os.path.realpath(path or BROWSE_START)
    if not os.path.isdir(path):
        raise ValueError("폴더가 아닙니다")
    dirs = []
    with os.scandir(path) as it:
        for e in sorted(it, key=lambda e: e.name):
            if e.name.startswith(".") or not e.is_dir():
                continue
            try:  # cheap peek: does it hold scans directly?
                with os.scandir(e.path) as sub:
                    has = any(x.name.lower().endswith(SCAN_EXTS) for x in sub)
            except OSError:
                continue
            dirs.append({"name": e.name, "scans": has})
            if len(dirs) >= 500:
                break
    parent = os.path.dirname(path)
    ds = dataset_for(path)
    return {"path": path, "parent": parent if parent != path else None, "dirs": dirs,
            "frontend": frontend_cfg(path)["name"], "note": ds.get("note"), **scan_folder_info(path)}


def local_cache_path(path, n_scans, frontend):
    key = hashlib.sha1(f"{path}|{n_scans}|{os.stat(path).st_mtime_ns}|{frontend}".encode()).hexdigest()[:16]
    return os.path.join(LOCAL_CACHE_DIR, key, "stage1_cache.npz")


FE_LABEL = {"kiss_icp": "KISS-ICP", "gicp": "GICP"}
PRESETS = []
JOBS = {}
QUEUE = queue.Queue()
_descriptor = None


def descriptor():
    global _descriptor
    if _descriptor is None:
        _descriptor = make_descriptor(CFG["descriptor"])
    return _descriptor


def _xy(a):
    return np.round(a[:, :2], 2).tolist()


def run_job(job):
    job.update(state="running", stage="데이터 확인", progress=0.0)
    if job["source"] == "local":
        pose, calib = find_sidecars(job["folder"])
        ds = FolderScans(job["folder"], pose_file=pose, calib_file=calib)
        fe_cfg = frontend_cfg(job["folder"])
        extra = [local_cache_path(job["folder"], len(ds), fe_cfg)]  # stage 1 kept per folder for re-runs
        k00 = os.path.realpath(os.path.join(CFG["dataset"]["root"], "sequences", "00", "velodyne"))
        if job["folder"] == k00:  # stage 1 already computed by run.py for KITTI 00
            extra.append(KITTI00_CACHE)
    else:
        ds = FolderScans(job["folder"], name="upload")
        fe_cfg = CFG["frontend"]
        extra = []
    n = min(len(ds), job["max_frames"] or len(ds), MAX_FRAMES)
    out = os.path.join(DEMO_DIR, job["id"])
    methods = job["methods"]
    steps = 1 + len(methods)

    def prog(stage_idx, label):
        def f(i, total):
            job["stage"] = f"{label} {i}/{total}"
            job["progress"] = (stage_idx + i / total) / steps
        return f

    cache = extra[0] if job["source"] == "local" else os.path.join(out, "stage1_cache.npz")
    have_cache = any(os.path.exists(p) and len(np.load(p, mmap_mode="r")["odom"]) >= n for p in [cache] + extra)
    fe = None if have_cache else make_frontend(fe_cfg)
    desc = None if have_cache else descriptor()
    stage1 = run_frontend(ds, fe, desc, n, CFG["backend"].get("fine_voxel", 0.5), cache,
                          progress=prog(0, f"1단계 · {FE_LABEL.get(fe_cfg['name'], fe_cfg['name'])} + OverlapTransformer"), extra_caches=extra)

    gt = ds.gt_poses[:n] if ds.gt_poses is not None else None
    res = {"frames": n, "total_frames": len(ds), "has_gt": gt is not None, "methods": [],
           "frontend": fe_cfg["name"]}
    T_align = None
    if gt is not None:
        res["gt"] = _xy(gt[:, :3, 3])
    for k, m in enumerate(methods):
        r = run_method(m, ds, stage1, n, CFG, os.path.join(out, m), progress=prog(1 + k, f"2단계 · {m}"))
        traj = r["_traj"][:, :3, 3]
        if gt is not None:
            T_align = umeyama_se3(traj, gt[:, :3, 3])
            traj = (T_align[:3, :3] @ traj.T).T + T_align[:3, 3]
        metrics = {k2: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k2, v in r.items() if not k2.startswith("_") and not isinstance(v, dict)}
        res["methods"].append({
            "method": m, "metrics": metrics, "traj": _xy(traj),
            "keyframes": [int(x) for x in r["_keyframes"]],
            "loops": [[lp["query_frame"], lp["match_frame"]] for lp in r["_loops"]],
        })
    odom = stage1["odom"][:, :3, 3]
    if gt is not None:
        T = umeyama_se3(odom, gt[:, :3, 3])
        odom = (T[:3, :3] @ odom.T).T + T[:3, 3]
    res["odom"] = _xy(odom)
    # a reused cache carries the timings of the run that built it, not of this one
    res["stage1_s"] = None if have_cache else {"frontend": float(stage1["t_frontend"]),
                                               "descriptor": float(stage1["t_descriptor"])}
    job.update(state="done", stage="완료", progress=1.0, result=res)


def worker():
    while True:
        job = QUEUE.get()
        try:
            run_job(job)
        except Exception as e:  # report to the page instead of dying
            traceback.print_exc()
            job.update(state="error", stage="오류", error=f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/job" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/health":
            return self._send(200, {"ok": True, "busy": any(j["state"] == "running" for j in JOBS.values()),
                                    "presets": PRESETS})
        if url.path == "/api/browse":
            try:
                return self._send(200, browse(q.get("path", [""])[0]))
            except (OSError, ValueError) as e:
                return self._send(400, {"error": f"열 수 없습니다: {e}"})
        if url.path == "/api/job":
            job = JOBS.get(q.get("id", [""])[0])
            if not job:
                return self._send(404, {"error": "unknown job"})
            return self._send(200, {k: v for k, v in job.items() if k != "folder"})
        # static: index.html and assets/ only
        rel = "index.html" if url.path in ("/", "/index.html") else url.path.lstrip("/")
        path = os.path.realpath(os.path.join(ROOT, rel))
        if not (rel == "index.html" or path.startswith(os.path.join(ROOT, "assets") + os.sep)) or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            self._send(200, f.read(), mimetypes.guess_type(path)[0] or "application/octet-stream")

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        length = int(self.headers.get("Content-Length", 0))
        if url.path == "/api/upload":
            session = q.get("session", [""])[0]
            name = os.path.basename(q.get("name", [""])[0])
            if not re.fullmatch(r"[0-9a-f]{32}", session) or not UPLOAD_NAMES.match(name):
                self.rfile.read(length)
                return self._send(400, {"error": "지원하지 않는 파일입니다 (.bin, .pcd, poses.txt, calib.txt)"})
            folder = os.path.join(UPLOAD_DIR, session)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, name), "wb") as f:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            return self._send(200, {"ok": True})
        if url.path == "/api/run":
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})
            methods = ["all"] + [m for m in req.get("methods", []) if m in ALLOWED_METHODS and m != "all"]
            job = {"id": uuid.uuid4().hex, "state": "queued", "stage": "대기 중", "progress": 0.0,
                   "methods": methods, "max_frames": int(req.get("max_frames") or 0), "created": time.time()}
            if req.get("path"):
                folder = os.path.realpath(str(req["path"]))
                if not os.path.isdir(folder):
                    return self._send(400, {"error": "폴더를 찾을 수 없습니다"})
                if not scan_folder_info(folder)["scans"]:
                    return self._send(400, {"error": "이 폴더에 .bin 또는 .pcd 스캔이 없습니다"})
                job.update(source="local", folder=folder)
            else:
                session = str(req.get("session", ""))
                folder = os.path.join(UPLOAD_DIR, session)
                if not re.fullmatch(r"[0-9a-f]{32}", session) or not os.path.isdir(folder):
                    return self._send(400, {"error": "업로드된 파일이 없습니다"})
                if not any(f.lower().endswith(SCAN_EXTS) for f in os.listdir(folder)):
                    return self._send(400, {"error": ".bin 또는 .pcd 스캔이 없습니다"})
                job.update(source="upload", folder=folder)
            JOBS[job["id"]] = job
            QUEUE.put(job)
            return self._send(200, {"id": job["id"]})
        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--clean", action="store_true", help="delete previous uploads and demo results")
    a = ap.parse_args()
    if a.clean:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        shutil.rmtree(DEMO_DIR, ignore_errors=True)
    global PRESETS
    PRESETS = presets()  # scanned once at start-up (counting scans in every folder)
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    groups = ", ".join("%s %d" % (g["group"], len(g["items"])) for g in PRESETS) or "none"
    print(f"MSA demo server: http://localhost:{a.port}  (shortcuts: {groups})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
