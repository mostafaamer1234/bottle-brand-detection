from __future__ import annotations

from pathlib import Path

from convert_labels import CLASSES, convert, load_ground_truth

ROOT = Path(__file__).resolve().parent.parent


def test_ground_truth_has_one_box_per_image_with_both_classes():
    rows = load_ground_truth(ROOT / "labeled_dataset" / "bottleGroundTruth.mat")
    assert len(rows) == 123
    classes = {cls for _, boxes in rows for cls, _ in boxes}
    assert classes == {0, 1}


def test_conversion_is_in_bounds_and_split(tmp_path):
    stats = convert(tmp_path, max_side=320)
    assert stats["train"] + stats["val"] == 123 and stats["out_of_bounds"] == 0
    for label in (tmp_path / "labels").rglob("*.txt"):
        for line in label.read_text().splitlines():
            cls, *coords = line.split()
            assert int(cls) < len(CLASSES)
            assert all(0.0 <= float(v) <= 1.0 for v in coords)
    assert "branded_bottle" in (tmp_path / "data.yaml").read_text()
