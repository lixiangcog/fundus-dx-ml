#!/usr/bin/env python3
"""Train and independently benchmark the OCT layer/fluid segmenter.

The MIRAGE Duke DME package preserves the subject-wise split: subjects 6-10
for development and subjects 1-5 for the independent test set.  Subject 10 is
held out for model selection; it is never used for gradient updates.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data/reference/mirage_duke/extracted/Duke_DME"
OUTPUT_ROOT = ROOT / "models/oct-structure"
LABEL_VALUES = np.array([0, 25, 51, 76, 102, 127, 153, 178, 204, 229], dtype=np.uint8)
CLASS_NAMES = ["Background", "ILM", "NFL", "IPL", "INL", "OPL", "ISM", "OS", "BM", "Fluid"]
SEED = 20260814


def build_model() -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=10,
        channels=(24, 48, 96, 192, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


def _decode_label(array: np.ndarray) -> np.ndarray:
    distances = np.abs(array[..., None].astype(np.int16) - LABEL_VALUES.astype(np.int16))
    return np.argmin(distances, axis=-1).astype(np.int64)


class DukeDataset(Dataset):
    def __init__(self, files: list[Path], augment: bool = False):
        self.files = files
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        label = _decode_label(np.asarray(Image.open(path.parents[1] / "semseg" / path.name)))
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if self.augment:
            if random.random() < 0.5:
                image, label = np.fliplr(image).copy(), np.fliplr(label).copy()
            gamma = random.uniform(0.82, 1.22)
            image = np.power(np.clip(image, 0, 1), gamma)
            image = np.clip(image * random.uniform(0.9, 1.1) + random.uniform(-0.04, 0.04), 0, 1)
            if random.random() < 0.35:
                image = np.clip(image + np.random.normal(0, random.uniform(0.002, 0.014), image.shape), 0, 1)
        image = (image - float(image.mean())) / max(float(image.std()), 0.08)
        return torch.from_numpy(image.astype(np.float32))[None], torch.from_numpy(label), path.name


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, 10).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = (probs + one_hot).sum(dim=(0, 2, 3))
    return 1.0 - ((2 * intersection[1:] + 1.0) / (denominator[1:] + 1.0)).mean()


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    intersections = np.zeros(10, dtype=np.float64)
    pred_counts = np.zeros(10, dtype=np.float64)
    truth_counts = np.zeros(10, dtype=np.float64)
    for images, labels, _ in loader:
        prediction = model(images.to(device)).argmax(1).cpu().numpy()
        truth = labels.numpy()
        for class_id in range(10):
            pred_mask, truth_mask = prediction == class_id, truth == class_id
            intersections[class_id] += np.logical_and(pred_mask, truth_mask).sum()
            pred_counts[class_id] += pred_mask.sum()
            truth_counts[class_id] += truth_mask.sum()
    dice = (2 * intersections + 1e-9) / (pred_counts + truth_counts + 1e-9)
    iou = (intersections + 1e-9) / (pred_counts + truth_counts - intersections + 1e-9)
    return {
        "mean_dice_excluding_background": float(dice[1:].mean()),
        "mean_layer_dice": float(dice[1:9].mean()),
        "fluid_dice": float(dice[9]),
        "per_class": {
            CLASS_NAMES[i]: {"dice": float(dice[i]), "iou": float(iou[i]), "pixels": int(truth_counts[i])}
            for i in range(10)
        },
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = sorted((DATA_ROOT / "train/bscan").glob("*.png"))
    train_files = [path for path in train_all if not path.name.startswith("Subject_10_")]
    val_files = [path for path in train_all if path.name.startswith("Subject_10_")]
    test_files = sorted((DATA_ROOT / "test/bscan").glob("*.png"))
    if not train_files or not val_files or not test_files:
        raise RuntimeError(f"Duke DME split is incomplete under {DATA_ROOT}")

    train_loader = DataLoader(DukeDataset(train_files, True), batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(DukeDataset(val_files), batch_size=2, num_workers=0)
    test_loader = DataLoader(DukeDataset(test_files), batch_size=2, num_workers=0)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)
    counts = np.zeros(10, dtype=np.float64)
    for path in train_files:
        counts += np.bincount(_decode_label(np.asarray(Image.open(path.parents[1] / "semseg" / path.name))).ravel(), minlength=10)
    class_weights = torch.from_numpy(np.clip(np.sqrt(counts.sum() / np.maximum(counts, 1)), 1, 8).astype(np.float32)).to(device)
    class_weights[0] = 0.35

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    for epoch in range(1, 121):
        model.train()
        losses = []
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = 0.55 * F.cross_entropy(logits, labels, weight=class_weights) + 0.45 * soft_dice_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        validation = evaluate(model, val_loader, device)
        score = validation["mean_dice_excluding_background"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": validation, "classes": CLASS_NAMES}, OUTPUT_ROOT / "duke_unet_v1.pth")
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"epoch": epoch, "loss": round(float(np.mean(losses)), 5), "val_dice": round(score, 5), "best": round(best_score, 5)}, ensure_ascii=False), flush=True)

    checkpoint = torch.load(OUTPUT_ROOT / "duke_unet_v1.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    benchmark = {
        "model": "MONAI 2D residual U-Net",
        "seed": SEED,
        "development_subjects": [6, 7, 8, 9],
        "selection_subject": [10],
        "independent_test_subjects": [1, 2, 3, 4, 5],
        "train_images": len(train_files),
        "validation_images": len(val_files),
        "test_images": len(test_files),
        "best_epoch": best_epoch,
        "validation": checkpoint["validation"],
        "test": evaluate(model, test_loader, device),
    }
    (OUTPUT_ROOT / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    tracked_benchmark = ROOT / "benchmarks/oct_structure_v1.json"
    tracked_benchmark.parent.mkdir(parents=True, exist_ok=True)
    tracked_benchmark.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(benchmark, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
