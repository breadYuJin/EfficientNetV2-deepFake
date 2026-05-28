"""Train EfficientNetV2-L (default) for binary deepfake classification.

Usage:
    python src/train.py \
        --data_root /path/to/dataSet \
        --splits_dir splits \
        --output_dir runs/v1

Resume:
    python src/train.py ... --resume runs/v1/checkpoints/last.pt
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DeepfakeDataset
from model import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp: bool):
    model.eval()
    probs_list, labels_list = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs).squeeze(1)
        probs_list.append(torch.sigmoid(logits).float().cpu().numpy())
        labels_list.append(labels.numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    return {"acc": acc, "auroc": auroc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--splits_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="tf_efficientnetv2_l")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed-precision training")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from scratch (skip ImageNet weights)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Debug: limit training set size")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"

    train_tf = build_transforms(args.image_size, train=True)
    eval_tf = build_transforms(args.image_size, train=False)
    train_ds = DeepfakeDataset(args.splits_dir / "train.csv", args.data_root, train_tf)
    val_ds = DeepfakeDataset(args.splits_dir / "val.csv", args.data_root, eval_tf)

    if args.max_train_samples is not None:
        train_ds.rows = train_ds.rows[: args.max_train_samples]
        val_ds.rows = val_ds.rows[: max(1, args.max_train_samples // 10)]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = create_model(args.model_name,
                         pretrained=not args.no_pretrained,
                         num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_auroc = -1.0
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc = ckpt.get("best_auroc", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} "
              f"(best_auroc={best_auroc:.4f})")

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["device"] = str(device)
    config["use_amp"] = use_amp
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    log_file = log_path.open("a")
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            running_loss = 0.0
            seen = 0
            for step, (imgs, labels, _) in enumerate(train_loader):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs).squeeze(1)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * imgs.size(0)
                seen += imgs.size(0)
                if step % args.log_every == 0:
                    print(f"[epoch {epoch} step {step}/{len(train_loader)}] "
                          f"loss={loss.item():.4f}", flush=True)
            scheduler.step()
            train_loss = running_loss / max(1, seen)
            metrics = evaluate_loader(model, val_loader, device, use_amp)
            epoch_time = time.time() - epoch_start

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": metrics["acc"],
                "val_auroc": metrics["auroc"],
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": epoch_time,
            }
            print(json.dumps(log_entry), flush=True)
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            ckpt_payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_auroc": best_auroc,
                "args": config,
            }
            torch.save(ckpt_payload, ckpt_dir / "last.pt")
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                ckpt_payload["best_auroc"] = best_auroc
                torch.save(ckpt_payload, ckpt_dir / "best.pt")
                print(f"new best val_auroc={best_auroc:.4f}")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
