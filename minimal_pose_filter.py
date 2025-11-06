#!/usr/bin/env python3
"""
minimal_pose_filter.py

Purpose
-------
Drop bad *poses* (images) from an existing COLMAP sparse model using a simple,
robust median-distance rule. Writes manifests and a cleaned COLMAP model that
you can pass to Nerfstudio via --colmap-path.

What it does (and doesn't)
--------------------------
✓ Identifies outlier camera centers by distance-from-median > K × median(distance).
✓ Writes manifests: inliers.txt / outliers.txt / outliers_ids.txt.
✓ Writes a cleaned model:
   - Fast path: uses `colmap model_converter --image_list_path` to BIN.
   - Fallback: converts original BIN→TXT, prunes outliers + tracks, writes TXT,
               and (optionally) converts back to BIN.
✗ Does NOT edit the COLMAP database.
✗ No PCA/kNN/plots—just median distance.

Usage
-----
# Default: keep poses within 5× median distance; write cleaned model + manifests
python3 minimal_pose_filter.py \
  --sparse_dir /path/to/colmap/sparse/0 \
  --out_dir runs/seq1

# Manifests only (no cleaned model)
python3 minimal_pose_filter.py \
  --sparse_dir /path/to/colmap/sparse/0 \
  --out_dir runs/seq1 \
  --apply manifest

# Loosen/tighten filtering; override guard (>10% removals)
python3 minimal_pose_filter.py \
  --sparse_dir /path/to/colmap/sparse/0 \
  --out_dir runs/seq1 \
  --mult 6.0 \
  --max_outlier_frac 0.25 \
  --force

# Nerfstudio (after cleaning)
ns-process-data colmap \
  --data /path/to/images \
  --colmap-path runs/seq1/sparse/0_cleaned_bin \
  --output-dir datasets/seq1_clean
# (if BIN conversion wasn't possible, point --colmap-path to runs/seq1/sparse/0_cleaned instead; TXT is fine)
"""

import argparse, json, math, os, re, subprocess, tempfile, shutil
from pathlib import Path
import numpy as np

# deps: pycolmap (for reading model), numpy. colmap binary for conversions.
import pycolmap  # pip install pycolmap


# ----------------- helpers: basic parsing -----------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def qvec2rotmat(q):
    qx, qy, qz, qw = q
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=float)

def load_centers(sparse_dir: str):
    """Read COLMAP sparse model, return (reconstruction, centers Nx3, ordered_image_ids, id2name)."""
    rec = pycolmap.Reconstruction(sparse_dir)
    records = []
    for iid, img in rec.images.items():
        q = img.cam_from_world().rotation.quat  # (qx,qy,qz,qw)
        R = qvec2rotmat(q)
        t = img.cam_from_world().translation
        C = (-R.T @ t).astype(float)
        records.append((iid, img.name, C))
    records.sort(key=lambda r: natural_key(r[1]))
    ordered_ids = [i for i, _, _ in records]
    centers = np.stack([c for _, _, c in records]) if records else np.zeros((0, 3))
    id2name = {i: n for i, n, _ in records}
    return rec, centers, ordered_ids, id2name

def detect_median_distance(positions: np.ndarray, mult: float = 5.0):
    """Flag any camera whose distance from the median center exceeds mult × median(distance)."""
    if len(positions) == 0:
        return np.zeros(0, dtype=bool), {"median_distance": 0.0, "threshold": 0.0, "num_outliers": 0}
    centroid = np.median(positions, axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    med = float(np.median(dists))
    thr = float(mult * med)
    mask = dists > thr
    return mask, {"median_distance": med, "threshold": thr, "num_outliers": int(mask.sum())}

def write_manifests(out_dir: Path, keep_ids, drop_ids, id2name):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inliers.txt").write_text("\n".join(id2name[i] for i in keep_ids) + ("\n" if keep_ids else ""))
    (out_dir / "outliers.txt").write_text("\n".join(id2name[i] for i in drop_ids) + ("\n" if drop_ids else ""))
    (out_dir / "outliers_ids.txt").write_text("\n".join(str(i) for i in drop_ids) + ("\n" if drop_ids else ""))

# ----------------- colmap conversion helpers -----------------
def colmap_has_image_list_path() -> bool:
    try:
        res = subprocess.run(["colmap", "model_converter", "--help"], capture_output=True, text=True)
        return res.returncode == 0 and ("--image_list_path" in (res.stdout + res.stderr))
    except Exception:
        return False

def colmap_convert(input_path: Path, output_path: Path, out_type: str) -> None:
    """Run `colmap model_converter` to convert model formats."""
    cmd = ["colmap", "model_converter", "--input_path", str(input_path), "--output_path", str(output_path), "--output_type", out_type]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or f"model_converter failed: {' '.join(cmd)}")

def write_clean_model_fast(input_sparse: Path, output_sparse: Path, keep_names):
    """Fast path: use --image_list_path to filter directly to BIN."""
    output_sparse.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for n in keep_names:
            f.write(n + "\n")
        list_path = f.name
    try:
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(input_sparse),
            "--output_path", str(output_sparse),
            "--output_type", "BIN",
            "--image_list_path", list_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or "model_converter fast path failed")
    finally:
        try: os.unlink(list_path)
        except Exception: pass

# ----------------- TXT round-trip pruning -----------------
def parse_cameras_txt(p: Path):
    cams = {}
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            toks = line.split()
            cam_id = int(toks[0])
            cams[cam_id] = line
    return cams

def parse_images_txt(p: Path):
    # images.txt has 2 lines per registered image:
    # L1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # L2: POINTS2D[]: x y POINT3D_ID ...
    images = {}
    name2id = {}
    points2d = {}
    with open(p, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        hdr = lines[i].split()
        img_id = int(hdr[0])
        name = hdr[9]
        images[img_id] = lines[i]
        name2id[name] = img_id
        i += 1
        pts = []
        if i < len(lines):
            toks = lines[i].split()
            for j in range(0, len(toks), 3):
                x = float(toks[j]); y = float(toks[j+1]); pid = int(toks[j+2])
                pts.append((x,y,pid))
        points2d[img_id] = pts
        i += 1
    return images, name2id, points2d

def parse_points3d_txt(p: Path):
    pts = {}
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            toks = line.split()
            pid = int(toks[0])
            xyz = toks[1:4]
            rgb = toks[4:7]
            err = toks[7]
            track_pairs = toks[8:]
            track = [(int(track_pairs[k]), int(track_pairs[k+1])) for k in range(0, len(track_pairs), 2)]
            pts[pid] = (xyz, rgb, err, track)
    return pts

def write_cameras_txt(p: Path, cams):
    with open(p, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...\n")
        f.write("# Number of cameras: {}\n".format(len(cams)))
        for _, line in sorted(cams.items()):
            f.write(line + "\n")

def write_images_txt(p: Path, images, points2d):
    with open(p, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(images)))
        for img_id in sorted(images.keys()):
            f.write(images[img_id] + "\n")
            pts = points2d.get(img_id, [])
            line = " ".join("{} {} {}".format(x, y, pid) for (x,y,pid) in pts)
            f.write(line + "\n")

def write_points3d_txt(p: Path, pts):
    with open(p, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(pts)))
        for pid in sorted(pts.keys()):
            xyz, rgb, err, track = pts[pid]
            track_str = " ".join("{} {}".format(i, j) for (i,j) in track)
            f.write("{} {} {} {} {} {} {} {} {}\n".format(pid, *xyz, *rgb, err, track_str))

def prune_txt_model(txt_dir: Path, out_dir: Path, outlier_names: list[str], min_track_len: int = 2):
    """Read TXT model, drop images in outlier_names, prune tracks, and write TXT output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = parse_cameras_txt(txt_dir / "cameras.txt")
    images, name2id, points2d = parse_images_txt(txt_dir / "images.txt")
    pts3d = parse_points3d_txt(txt_dir / "points3D.txt")

    drop_ids = set(name2id[n] for n in outlier_names if n in name2id)
    keep_ids = set(images.keys()) - drop_ids

    # Filter images + their 2D arrays
    images_kept = {i: hdr for i, hdr in images.items() if i in keep_ids}
    points2d_kept = {i: [(x,y,pid) for (x,y,pid) in points2d[i]] for i in keep_ids}

    # Remove dropped images from tracks; drop points with short remaining tracks
    pts3d_kept = {}
    for pid, (xyz, rgb, err, track) in pts3d.items():
        new_track = [(img_id, idx) for (img_id, idx) in track if img_id in keep_ids]
        if len(new_track) >= min_track_len:
            pts3d_kept[pid] = (xyz, rgb, err, new_track)
        else:
            # For refs that survive in keep_ids but point loses support, null the POINT3D_ID in 2D table
            for (img_id, idx) in new_track:
                if img_id in points2d_kept and 0 <= idx < len(points2d_kept[img_id]):
                    x,y,_ = points2d_kept[img_id][idx]
                    points2d_kept[img_id][idx] = (x,y,-1)

    # Ensure 2D point pid references are only to kept 3D points
    keep_pid = set(pts3d_kept.keys())
    for img_id, pts in points2d_kept.items():
        new_pts = []
        for (x,y,pid) in pts:
            new_pts.append((x,y, pid if pid in keep_pid else -1))
        points2d_kept[img_id] = new_pts

    # Write filtered TXT
    write_cameras_txt(out_dir / "cameras.txt", cams)
    write_images_txt(out_dir / "images.txt", images_kept, points2d_kept)
    write_points3d_txt(out_dir / "points3D.txt", pts3d_kept)

    return len(drop_ids), len(pts3d_kept)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Filter bad poses using a median-distance rule; emit cleaned COLMAP model.")
    ap.add_argument("--sparse_dir", required=True, help="Path to COLMAP sparse model dir (with images.bin etc).")
    ap.add_argument("--out_dir", default="outlier_filter_output", help="Where to write manifests / cleaned model.")
    ap.add_argument("--apply", choices=["manifest", "model"], default="model", help="Write only manifests, or also a cleaned model.")
    ap.add_argument("--mult", type=float, default=5.0, help="Threshold = mult × median(distance from median center).")
    ap.add_argument("--max_outlier_frac", type=float, default=0.10, help="Guardrail: refuse to drop more than this fraction unless --force.")
    ap.add_argument("--force", action="store_true", help="Override guardrail.")
    ap.add_argument("--bin_output", action="store_true", help="If possible, convert cleaned TXT to BIN (fallback path).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sparse_dir = Path(args.sparse_dir)

    # Load centers and names
    rec, centers, ordered_ids, id2name = load_centers(str(sparse_dir))
    mask, info = detect_median_distance(centers, mult=args.mult)
    drop_ids = [ordered_ids[i] for i in np.where(mask)[0]]
    keep_ids = [i for i in ordered_ids if i not in set(drop_ids)]
    keep_names = [id2name[i] for i in keep_ids]
    outlier_names = [id2name[i] for i in drop_ids]

    total = len(ordered_ids)
    num_out = len(drop_ids)
    frac = (num_out / total) if total else 0.0

    # Always write manifests
    write_manifests(out_dir, keep_ids, drop_ids, id2name)

    guard_blocked = (frac > args.max_outlier_frac) and (not args.force)

    cleaned_txt_dir = out_dir / "sparse" / "0_cleaned"          # TXT cleaned model
    cleaned_bin_dir = out_dir / "sparse" / "0_cleaned_bin"      # BIN cleaned model (if available)
    model_written_path = None
    model_error = None
    path_notes = []

    if args.apply == "model" and not guard_blocked:
        try:
            if colmap_has_image_list_path():
                # Fast path: write BIN directly
                write_clean_model_fast(sparse_dir, cleaned_bin_dir, keep_names)
                model_written_path = str(cleaned_bin_dir)
                path_notes.append("Cleaned BIN model created via --image_list_path fast path.")
            else:
                # Fallback: BIN -> TXT, prune, (optionally) TXT -> BIN
                # 1) convert original to TXT temp
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_txt = Path(tmpdir) / "txt"
                    tmp_txt.mkdir(parents=True, exist_ok=True)
                    colmap_convert(sparse_dir, tmp_txt, "TXT")
                    # 2) prune & write cleaned TXT
                    dropped, kept_pts = prune_txt_model(tmp_txt, cleaned_txt_dir, outlier_names, min_track_len=2)
                model_written_path = str(cleaned_txt_dir)
                path_notes.append("Cleaned TXT model created via round-trip fallback.")
                # 3) optionally convert cleaned TXT to BIN
                if args.bin_output:
                    try:
                        cleaned_bin_dir.mkdir(parents=True, exist_ok=True)
                        colmap_convert(cleaned_txt_dir, cleaned_bin_dir, "BIN")
                        model_written_path = str(cleaned_bin_dir)
                        path_notes.append("Converted cleaned TXT to BIN.")
                    except Exception as e:
                        path_notes.append(f"TXT→BIN conversion failed; using TXT. Error: {e}")
        except Exception as e:
            model_error = str(e)

    # Create JSON summary
    summary = {
        "summary": {
            "total_images": total,
            "num_outliers": num_out,
            "fraction_outliers": frac,
            "guard_blocked": guard_blocked,
            "apply": args.apply,
            "cleaned_model_path": model_written_path,
        },
        "median_rule": info,
        "manifests": {
            "inliers": str(out_dir / "inliers.txt"),
            "outliers": str(out_dir / "outliers.txt"),
            "outliers_ids": str(out_dir / "outliers_ids.txt"),
        },
        "outlier_names": outlier_names,
        "errors": {"model_write": model_error} if model_error else {},
        "notes": [
            *path_notes,
            "Pass --colmap-path to the cleaned BIN if available; else point to the cleaned TXT dir.",
            "No database edits performed.",
        ],
    }
    
    # Write JSON to file
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also print to stdout for logging
    print(json.dumps(summary, indent=2))

    # Friendly console message
    if guard_blocked:
        print(f"[guard] refusing to drop {num_out}/{total} images (> {args.max_outlier_frac:.0%}). Use --force to override.", flush=True)
    elif args.apply == "model" and model_written_path:
        print(f"[ok] cleaned model written to: {model_written_path}", flush=True)
    else:
        print("[ok] manifests written.", flush=True)


if __name__ == "__main__":
    main()
