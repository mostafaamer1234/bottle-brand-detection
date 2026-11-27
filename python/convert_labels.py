"""Convert the MATLAB Image Labeler ground truth (bottleGroundTruth.mat) into a YOLO dataset.

MATLAB stores boxes as [x, y, width, height] in raw pixel coordinates (EXIF rotation
not applied), so images are re-saved without their EXIF orientation tag to keep
pixels and boxes aligned. Images are downscaled to 1280 px on the long side.

    python python/convert_labels.py --out yolo_dataset
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from matio import load_from_mat
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["branded_bottle", "generic_bottle"]


def load_ground_truth(mat_path: Path) -> list[tuple[str, list[tuple[int, list[float]]]]]:
    gt = load_from_mat(str(mat_path))["gTruth"].properties["any"][0, 0]
    sources = [str(s[0][0]) if hasattr(s[0], "__len__") else str(s[0]) for s in gt["DataSource"][0, 0]["Source"]]
    rows = []
    for src, labels in zip(sources, gt["LabelData"][:, 0]):
        boxes = []
        for cls_id, cls in enumerate(CLASSES):
            arr = labels[cls]
            for box in arr.reshape(-1, 4) if arr.size else []:
                boxes.append((cls_id, [float(v) for v in box]))
        rows.append((Path(src).name, boxes))
    return rows


def convert(out: Path, val_fraction: float = 0.2, seed: int = 0, max_side: int = 1280) -> dict[str, int]:
    rows = load_ground_truth(ROOT / "labeled_dataset" / "bottleGroundTruth.mat")
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = round(len(rows) * val_fraction)
    stats = {"train": 0, "val": 0, "boxes": 0, "out_of_bounds": 0}
    for i, (name, boxes) in enumerate(rows):
        split = "val" if i < n_val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        im = Image.open(ROOT / "labeled_dataset" / "images" / name)
        w, h = im.size  # raw sensor orientation, same frame MATLAB labeled
        scale = min(1.0, max_side / max(w, h))
        im.convert("RGB").resize((round(w * scale), round(h * scale))).save(out / "images" / split / f"{Path(name).stem}.jpg", quality=92)
        lines = []
        for cls_id, (x, y, bw, bh) in boxes:
            if x + bw - 1 > w or y + bh - 1 > h:
                stats["out_of_bounds"] += 1
            cx, cy = (x - 1 + bw / 2) / w, (y - 1 + bh / 2) / h  # MATLAB pixels are 1-based
            lines.append(f"{cls_id} {min(cx, 1):.6f} {min(cy, 1):.6f} {min(bw / w, 1):.6f} {min(bh / h, 1):.6f}")
            stats["boxes"] += 1
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[split] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "yolo_dataset")
    args = ap.parse_args()
    print(convert(args.out))
