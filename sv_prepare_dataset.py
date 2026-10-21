"""
sv_prepare_dataset.py -- turn the Kaggle Box archive into a clean YOLO dataset.

    python3 sv_prepare_dataset.py --zip ~/Downloads/archive.zip
    python3 sv_prepare_dataset.py --src data/raw/BOX        # already extracted

What it fixes (all of these are real problems in that archive):
  * the label directory is called `label/`, but Ultralytics requires `labels/`
  * there is a second, duplicate copy of the labels under `label/label/`
  * several files are named "WhatsApp Image ....jpg" with spaces
  * at least one training image has no matching label file
  * the bundled coco128.yaml is the generic 80-class COCO file and does NOT
    describe this data at all -- we write a correct single-class yaml instead
  * train/val are near-duplicate burst frames, so a random split leaks. We group
    by near-duplicate image hash and split by GROUP, which gives an honest
    validation number instead of a flattering one.

Output: data/processed/{images,labels}/{train,val} + data/processed/box.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import sv_config as cfg

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_name(name: str) -> str:
    """Spaces and parentheses in filenames break a lot of tooling. Remove them."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return re.sub(r"_+", "_", stem).strip("_")


def extract(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            # Refuse absolute paths and ../ traversal from an untrusted archive.
            p = Path(m)
            if p.is_absolute() or ".." in p.parts:
                print(f"  skipping unsafe entry: {m}")
                continue
            z.extract(m, dest)
    print(f"extracted -> {dest}")
    return dest


def find_pairs(root: Path) -> list:
    """
    Match every image to its label file.

    We deliberately prefer labels under a path containing 'BOX' so the duplicate
    `label/label/` tree does not produce two competing matches.
    """
    images, labels = [], defaultdict(list)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXT:
            images.append(p)
        elif p.suffix.lower() == ".txt" and p.name != "classes.txt":
            labels[p.stem].append(p)

    pairs, missing = [], []
    for img in images:
        cands = labels.get(img.stem, [])
        if not cands:
            missing.append(img)
            continue
        cands.sort(key=lambda q: (0 if "BOX" in q.parts else 1, len(q.parts)))
        pairs.append((img, cands[0]))

    print(f"images found      : {len(images)}")
    print(f"label stems found : {len(labels)}")
    print(f"matched pairs     : {len(pairs)}")
    if missing:
        print(f"DROPPED (no label): {len(missing)}")
        for m in missing[:6]:
            print(f"    {m.name}")
    return pairs


def dhash(path: Path, size: int = 8) -> str:
    """Perceptual hash, for grouping near-duplicate burst frames."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return hashlib.md5(path.name.encode()).hexdigest()
    img = cv2.resize(img, (size + 1, size))
    return "".join("1" if a else "0" for a in (img[:, 1:] > img[:, :-1]).flatten())


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def group_near_duplicates(pairs: list, threshold: int = 8) -> list:
    """Union-find over perceptual hashes -> a group id per image."""
    hashes = [dhash(img) for img, _ in pairs]
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = [find(i) for i in range(len(pairs))]
    print(f"near-duplicate groups: {len(set(groups))} across {len(pairs)} images")
    return groups


def valid_label(path: Path) -> tuple:
    """Return (lines, n_bad). Clamps coordinates and forces class id 0."""
    lines, bad = [], 0
    for ln in path.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            bad += 1
            continue
        try:
            vals = [float(v) for v in parts[1:5]]
        except ValueError:
            bad += 1
            continue
        if any(not np.isfinite(v) for v in vals):
            bad += 1
            continue
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        if vals[2] <= 0 or vals[3] <= 0:
            bad += 1
            continue
        lines.append(f"0 {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f}")
    return lines, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=None)
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(cfg.DATA_DIR / "processed"))
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.zip:
        src = extract(Path(args.zip).expanduser(), cfg.DATA_DIR / "raw")
    elif args.src:
        src = Path(args.src).expanduser()
    else:
        raise SystemExit("Pass --zip <archive.zip> or --src <folder>")

    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit("No image/label pairs found. Check the path.")

    groups = group_near_duplicates(pairs)

    # Split by GROUP, not by image.
    uniq = sorted(set(groups))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * args.val_frac)))
    val_groups = set(uniq[:n_val])

    out = Path(args.out)
    for sp in ("train", "val"):
        for kind in ("images", "labels"):
            d = out / kind / sp
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    total_bad = 0
    for (img, lbl), g in zip(pairs, groups):
        sp = "val" if g in val_groups else "train"
        lines, bad = valid_label(lbl)
        total_bad += bad
        if not lines:
            continue
        stem = safe_name(img.stem)
        dst_i = out / "images" / sp / f"{stem}{img.suffix.lower()}"
        n = 1
        while dst_i.exists():
            dst_i = out / "images" / sp / f"{stem}_{n}{img.suffix.lower()}"
            n += 1
        shutil.copy2(img, dst_i)
        (out / "labels" / sp / f"{dst_i.stem}.txt").write_text("\n".join(lines) + "\n")
        counts[sp] += 1
        boxes[sp] += len(lines)

    yaml_path = out / "box.yaml"
    yaml_path.write_text(
        f"# Single-class parcel dataset, built by sv_prepare_dataset.py\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n  0: box\n"
    )

    print(f"\ntrain images: {counts['train']:4d}   boxes: {boxes['train']}")
    print(f"val   images: {counts['val']:4d}   boxes: {boxes['val']}")
    if total_bad:
        print(f"malformed label lines dropped: {total_bad}")
    print(f"\ndataset yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
