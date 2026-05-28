"""PyTorch Dataset for the deepfake split CSVs produced by build_splits.py."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

SOURCES = ["140K", "DF40", "forgeryNet", "keggle", "wild"]
SOURCE_TO_IDX = {name: i for i, name in enumerate(SOURCES)}


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append((
                    r["path"],
                    int(r["label"]),
                    SOURCE_TO_IDX[r["source"]],
                ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rel_path, label, source_idx = self.rows[idx]
        img = Image.open(self.data_root / rel_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, source_idx
