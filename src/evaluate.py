"""Evaluate a trained checkpoint on a split (default: test).

Produces:
    <output_dir>/<split>_metrics.json   - overall + per-source metrics
    <output_dir>/<split>_confusion.png  - confusion matrix
    <output_dir>/<split>_roc.png        - ROC curve
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SOURCES, DeepfakeDataset
from model import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    probs_list, labels_list, sources_list = [], [], []
    for imgs, labels, sources in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs).squeeze(1)
        probs_list.append(torch.sigmoid(logits).float().cpu().numpy())
        labels_list.append(labels.numpy())
        sources_list.append(sources.numpy())
    return (
        np.concatenate(probs_list),
        np.concatenate(labels_list),
        np.concatenate(sources_list),
    )


def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    out = {
        "count": int(len(labels)),
        "positives": int(labels.sum()),
        "acc": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auroc"] = None
    out["confusion_matrix"] = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return out


def plot_confusion(cm, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"]); ax.set_yticklabels(["real", "fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(labels, probs, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUROC={roc_auc_score(labels, probs):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--splits_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="tf_efficientnetv2_l")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = DeepfakeDataset(args.splits_dir / f"{args.split}.csv", args.data_root, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = create_model(args.model_name, pretrained=False, num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    probs, labels, sources = predict(model, loader, device, use_amp)

    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "overall": compute_metrics(probs, labels),
        "per_source": {},
    }
    for idx, name in enumerate(SOURCES):
        mask = sources == idx
        if mask.sum() == 0:
            continue
        report["per_source"][name] = compute_metrics(probs[mask], labels[mask])

    out_json = args.output_dir / f"{args.split}_metrics.json"
    with out_json.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    cm = np.array(report["overall"]["confusion_matrix"])
    plot_confusion(cm, args.output_dir / f"{args.split}_confusion.png")
    plot_roc(labels, probs, args.output_dir / f"{args.split}_roc.png")

    print(json.dumps(report["overall"], indent=2))
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
