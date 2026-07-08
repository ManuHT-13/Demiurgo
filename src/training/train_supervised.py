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
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.dataset import CIFAR10SupervisedDataset
from src.models.resnet import get_resnet18
from src.utils.metrics import AverageMeter, accuracy
from src.utils.seed import set_seed

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

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        """
        The base dataset (CIFAR10SupervisedDataset) returns a float tensor 
        (C, H, W) with values ​​in the [0, 1] range when no transform is passed 
        We convert it back to a uint8 NumPy array (H, W, C) because the torchvision 
        transforms used below (RandomCrop, RandomHorizontalFlip, etc) are designed 
        for PIL or NumPy-style images, not for tensors that have already been normalized
        """
        image, label = self.subset[idx]
        image = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        image = self.transform(image)
        return image, label

def build_dataloaders(cfg):
    """
    Create the training and validation DataLoaders

    We use data augmentation only on the training set
    It is not applied during validation: we want to measure actual performance on
    "clean" images, not on altered versions

    Subtracting the mean and dividing by the standard deviation ("normalizing") helps
    training converge faster and more stably
    """

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

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
    val_ds = _TransformWrapper(val_subset, val_transform)

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

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """
    Train while measuring loss and accuracy
    """
       
    model.train()   # Enable training mode (affects layers such as BatchNorm)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()  # Clear gradients from the previous batch
        
        # Enable automatic mixed precision where it is safe to do so
        with torch.cuda.amp.autocast(enabled=use_amp):
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

    # Load the data using dataset.py and split it into training and validation sets
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Training samples: {len(train_loader.dataset)} | "
      f"Validation samples: {len(val_loader.dataset)}")

    # Create the model resnet.py
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoints_dir = Path(cfg["output"]["checkpoints_dir"])
    logs_dir = Path(cfg["output"]["logs_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"]
        )

    best_val_acc = 0.0

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
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "config": cfg,
                },
                checkpoints_dir / "best_model.pt",
            )
            print(f"  -> New best model saved (val_acc={val_acc:.2f}%)")

    torch.save(
        {"epoch": cfg["training"]["epochs"], "model_state_dict": model.state_dict()},
        checkpoints_dir / "last_model.pt",
    )
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()