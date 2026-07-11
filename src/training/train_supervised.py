"""
Training script for SCENARIO 1:
CNN (ResNet-18) trained in a supervised manner using the CORRECT CIFAR-10 labels

HOW TO RUN
From the project root:
    python -m src.training.train_supervised --config configs/config_supervised.yaml

Use -m so that Python treats src as a package
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.dataset import CIFAR10SupervisedDataset
from src.models.resnet import get_resnet18
from src.utils.evaluation import evaluate_model
from src.utils.metrics import AverageMeter, accuracy
from src.utils.seed import set_seed
from src.utils.stats import compute_or_load_mean_std
from src.utils.visualization import plot_confusion_matrix, plot_training_curves

DEFAULT_CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    """
    Read the configuration (config_supervised.yaml)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_supervised.yaml",
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class _TransformWrapper(torch.utils.data.Dataset):
    """
    Set a random seed to make the experiment reproducible
    This class wraps an existing subset and applies its own
    transform (one for training with augmentation, another for validation
    without augmentation)
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # The base dataset (CIFAR10SupervisedDataset) returns a float tensor
        # (C, H, W) with values in the [0, 1] range when no transform is passed.
        # We convert it back to a uint8 NumPy array (H, W, C) because the
        # torchvision transforms used below (RandomCrop, RandomHorizontalFlip,
        # etc) are designed for PIL or NumPy-style images, not for tensors
        # that have already been normalized.
        image = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        image = self.transform(image)
        return image, label


def build_transforms(mean, std):
    """
    Creates the training (with data augmentation) and
    validation/test (without augmentation) transformations, using
    the mean/std automatically computed
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, eval_transform


def build_train_val_dataloaders(cfg, train_transform, eval_transform):
    """Creates the training and validation DataLoaders (split from the training set)"""
    full_dataset = CIFAR10SupervisedDataset(
        images_path=cfg["data"]["images_path"],
        labels_path=cfg["data"]["labels_path"],
        transform=None,
    )

    val_size = int(len(full_dataset) * cfg["data"]["val_split"])
    train_size = len(full_dataset) - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
    )

    train_ds = _TransformWrapper(train_subset, train_transform)
    val_ds = _TransformWrapper(val_subset, eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


def build_test_dataloader(cfg, eval_transform):
    """
    Creates the DataLoader for the actual TEST set (test_images.npy),
    completely separate from the training/validation split above.
    It is never mixed with them and is only used after training
    has finished.
    """
    test_dataset = CIFAR10SupervisedDataset(
        images_path=cfg["data"]["test_images_path"],
        labels_path=cfg["data"]["test_labels_path"],
        transform=None,
    )
    test_ds = _TransformWrapper(test_dataset, eval_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """
    Train for one epoch while measuring loss and accuracy
    """
    model.train()  # Enable training mode (affects layers such as BatchNorm)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()  # Clear gradients from the previous batch

        # Enable automatic mixed precision where it is safe to do so
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # The GradScaler prevents float16 gradients from underflowing,
        # which is a common issue when using mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()  # Switch to evaluation mode (disables training behavior such as dropout)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():

    # Read the configuration (config_supervised.yaml)
    args = parse_args()
    cfg = load_config(args.config)

    # Set a random seed to make the experiment reproducible
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: No GPU detected. Training on CPU will be very slow.")

    class_names = cfg.get("model", {}).get("class_names", DEFAULT_CIFAR10_CLASS_NAMES)

    checkpoints_dir = Path(cfg["output"]["checkpoints_dir"])
    logs_dir = Path(cfg["output"]["logs_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Automatically compute mean/std using only the training set
    mean, std = compute_or_load_mean_std(
        images_path=cfg["data"]["images_path"],
        cache_path=cfg["output"]["stats_cache_path"],
    )

    train_transform, eval_transform = build_transforms(mean, std)
    train_loader, val_loader = build_train_val_dataloaders(cfg, train_transform, eval_transform)
    test_loader = build_test_dataloader(cfg, eval_transform)

    print(
        f"Samples -> train: {len(train_loader.dataset)} | "
        f"validation: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}"
    )

    # Create the model (see src/models/resnet.py)
    model = get_resnet18(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    # New torch.amp API (torch.cuda.amp.GradScaler is deprecated): the
    # device type ("cuda") is now passed explicitly as the first argument.
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    log_path = logs_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"]
        )

    best_val_acc = 0.0

    # --- Early stopping setup ---
    # epochs_without_improvement counts how many epochs in a row have
    # passed without a "real" improvement (bigger than min_delta) in
    # validation accuracy. If it reaches "patience", we stop training
    # early: this saves time once the model has basically converged,
    # without hurting the final result (we always keep the best
    # checkpoint seen so far, regardless of when training stops).
    es_cfg = cfg["training"].get("early_stopping", {})
    early_stopping_enabled = es_cfg.get("enabled", False)
    patience = es_cfg.get("patience", 10)
    min_delta = es_cfg.get("min_delta", 0.0)
    epochs_without_improvement = 0

    # Train for N epochs, measuring loss and accuracy on both the training and validation sets
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
            f"lr={current_lr:.5f} | {elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed]
            )

        # Save the best model (highest validation accuracy)
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "config": cfg,
                    "mean": mean,
                    "std": std,
                },
                checkpoints_dir / "best_model.pt",
            )
            print(f"  -> New best model saved (val_acc={val_acc:.2f}%)")

            # Only reset the "no improvement" counter if the improvement
            # is meaningful (bigger than min_delta). A tiny +0.01% still
            # updates the checkpoint above, but doesn't count as real
            # progress for early stopping purposes.
            if improvement >= min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if early_stopping_enabled and epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping: no meaningful improvement in validation "
                f"accuracy for {patience} epochs in a row (best so far: "
                f"{best_val_acc:.2f}%). Stopping at epoch {epoch}/"
                f"{cfg['training']['epochs']}."
            )
            break

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        checkpoints_dir / "last_model.pt",
    )
    print(f"\nTraining finished at epoch {epoch}/{cfg['training']['epochs']}. "
          f"Best validation accuracy: {best_val_acc:.2f}%")

    # Plot the training curves
    plot_training_curves(log_path, figures_dir / "training_curves.png")

    # Final evaluation on the test set using the best saved model
    print("\nEvaluating the best model on the test set...")
    best_checkpoint = torch.load(checkpoints_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    results = evaluate_model(model, test_loader, device, class_names)

    print(f"Test accuracy: {results['accuracy']:.2f}%")
    print(results["classification_report_text"])

    plot_confusion_matrix(
        results["confusion_matrix"], class_names, figures_dir / "confusion_matrix.png"
    )

    with open(figures_dir / "classification_report.txt", "w") as f:
        f.write(results["classification_report_text"])

    metrics_summary = {
        "test_accuracy": results["accuracy"],
        "best_val_accuracy": best_val_acc,
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "class_names": class_names,
        "mean": mean,
        "std": std,
    }
    with open(logs_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nEverything has been saved to:\n  {checkpoints_dir}\n  {logs_dir}\n  {figures_dir}")


if __name__ == "__main__":
    main()