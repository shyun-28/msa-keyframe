"""Download the DiTer scan folders (not the ROS bags) into scan folders the pipeline reads.

    curl -L -o diter.torrent https://academictorrents.com/download/c604afddbe24fff0b873acbed370ff89df0c1614.torrent
    python3 tools/fetch_diter.py /HDD/DiTer --torrent diter.torrent

DiTer (Jeong et al., IEEE Sensors Letters 2024, CC BY-NC-SA 4.0) is distributed as a
755 GB torrent (Academic Torrents c604afddbe24fff0b873acbed370ff89df0c1614). Next to the
bags it holds, per sequence, `scans/*.pcd` with `pose/optimized_poses.txt` (KITTI format)
and `time/times.txt`, about 0.5 GB in total. This script reads the file list from the
torrent and fetches only those files over the torrent's HTTP web seed (archive.org), into

    <out>/<SeqA_HILL01>/{000000.pcd, ..., poses.txt, times.txt}

Note: these scans are the SLAM keyframes the authors kept (about one per metre), not the
full 10 Hz LiDAR stream, which only exists inside the bags.
Files already present with the right size are skipped, so the script can be re-run.
"""

import argparse
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WEB_SEED = "https://archive.org/download/DiTer/"
WANTED = ("/scans/", "/pose/", "/time/")


def bdecode(b, i=0):
    c = b[i:i + 1]
    if c == b"i":
        j = b.index(b"e", i)
        return int(b[i + 1:j]), j + 1
    if c in (b"l", b"d"):
        i += 1
        items = []
        while b[i:i + 1] != b"e":
            v, i = bdecode(b, i)
            items.append(v)
        return (dict(zip(items[::2], items[1::2])) if c == b"d" else items), i + 1
    j = b.index(b":", i)
    n = int(b[i:j])
    return b[j + 1:j + 1 + n], j + 1 + n


def file_list(torrent):
    with open(torrent, "rb") as f:
        meta, _ = bdecode(f.read())
    for f in meta[b"info"][b"files"]:
        path = "/".join(p.decode() for p in f[b"path"])
        if any(w in "/" + path for w in WANTED) and not path.endswith(".tmp"):
            yield path, f[b"length"]


def target(out, path):
    seq_set, seq, kind, name = path.split("/")[-4:]
    folder = os.path.join(out, f"{seq_set.replace(' ', '')}_{seq}")
    name = {"pose": "poses.txt", "time": "times.txt"}.get(kind, name)
    return os.path.join(folder, name)


def fetch(job):
    path, size, dst = job
    url = WEB_SEED + urllib.parse.quote(path)
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f:
                f.write(r.read())
            if os.path.getsize(dst + ".part") == size:
                os.replace(dst + ".part", dst)
                return None
        except OSError:
            pass
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output folder, one sub-folder per sequence")
    ap.add_argument("--torrent", required=True, help="the DiTer .torrent file (see the command above)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for path, size in file_list(a.torrent):
        dst = target(a.out, path)
        if not (os.path.exists(dst) and os.path.getsize(dst) == size):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            jobs.append((path, size, dst))
    print(f"{len(jobs)} files to download")
    with ThreadPoolExecutor(a.workers) as ex:
        failed = [p for p in ex.map(fetch, jobs) if p]
    for p in failed:
        print("FAILED", p, file=sys.stderr)
    for seq in sorted(os.listdir(a.out)):
        d = os.path.join(a.out, seq)
        n = sum(f.endswith(".pcd") for f in os.listdir(d))
        print(f"{seq:16s} {n:4d} scans  poses.txt={'yes' if os.path.exists(os.path.join(d, 'poses.txt')) else 'no'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
