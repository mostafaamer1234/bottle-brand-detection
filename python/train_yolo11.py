"""Train and evaluate YOLO11 on the converted bottle dataset.

    python python/convert_labels.py --out yolo_dataset
    python python/train_yolo11.py --data yolo_dataset/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("yolo_dataset/data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, device=device, seed=0, deterministic=True,
                project=args.project, name="yolo11n_bottles", exist_ok=True, patience=30, plots=True, verbose=False)
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=device, split="val", plots=True)
    names = metrics.names
    result = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_AP50": {names[i]: float(ap50) for i, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50)},
        "device": device,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/yolo11n_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    onnx = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz)
    print("exported", onnx)


if __name__ == "__main__":
    main()
