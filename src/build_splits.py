"""Generate train/val/test split CSVs for the deepfake dataset.

Scans `<data_root>/real` and `<data_root>/fake`, groups files by source dataset
prefix (140K, DF40, forgeryNet, keggle, wild), and splits each
(source, class) bucket into 70/20/10 with a fixed random seed.

Outputs:
    <out_dir>/train.csv
    <out_dir>/val.csv
    <out_dir>/test.csv
    <out_dir>/split_summary.json

CSV columns: path,label,source
    path   - relative to <data_root>, e.g. "real/forgeryNet_0000000.jpg"
    label  - 0 (real) or 1 (fake)
    source - one of {140K, DF40, forgeryNet, keggle, wild}
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

SOURCES = ["140K", "DF40", "forgeryNet", "keggle", "wild"]
LABELS = {"real": 0, "fake": 1}
RATIOS = (0.7, 0.2, 0.1)
DEFAULT_SEED = 42


def detect_source(filename: str):
    prefix = filename.split("_", 1)[0]
    return prefix if prefix in SOURCES else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path,
                        help="Path containing real/ and fake/ subfolders")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Destination for split CSVs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        sys.exit(f"data_root does not exist: {args.data_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    buckets = {(src, cls): [] for src in SOURCES for cls in LABELS}
    skipped = 0

    for class_name in LABELS:
        class_dir = args.data_root / class_name
        if not class_dir.is_dir():
            sys.exit(f"Missing class directory: {class_dir}")
        for entry in class_dir.iterdir():
            if not entry.is_file() or entry.suffix.lower() != ".jpg":
                continue
            src = detect_source(entry.name)
            if src is None:
                skipped += 1
                continue
            buckets[(src, class_name)].append(f"{class_name}/{entry.name}")

    splits = {"train": [], "val": [], "test": []}
    summary = {}

    for (src, class_name), paths in buckets.items():
        paths.sort()  # deterministic order before shuffle
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])
        train = paths[:n_train]
        val = paths[n_train:n_train + n_val]
        test = paths[n_train + n_val:]

        label = LABELS[class_name]
        for p in train:
            splits["train"].append((p, label, src))
        for p in val:
            splits["val"].append((p, label, src))
        for p in test:
            splits["test"].append((p, label, src))

        summary[f"{src}/{class_name}"] = {
            "total": n, "train": len(train), "val": len(val), "test": len(test),
        }

    for split_name, rows in splits.items():
        rng.shuffle(rows)
        out_path = args.out_dir / f"{split_name}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "source"])
            writer.writerows(rows)
        print(f"{split_name}: {len(rows):>10d} rows -> {out_path}")

    summary_path = args.out_dir / "split_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "seed": args.seed,
            "ratios": {"train": RATIOS[0], "val": RATIOS[1], "test": RATIOS[2]},
            "sources": SOURCES,
            "skipped_files": skipped,
            "per_group": summary,
            "totals": {k: len(v) for k, v in splits.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"summary -> {summary_path}")
    if skipped:
        print(f"warning: skipped {skipped} files with unknown source prefix")


if __name__ == "__main__":
    main()
