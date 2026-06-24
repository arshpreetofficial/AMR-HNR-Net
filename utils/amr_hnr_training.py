

from pathlib import Path
import csv
import random
import json
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(gpu_id: int = 0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        ensure_dir(self.path.parent)

        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row):
        row = {k: row.get(k, "") for k in self.fieldnames}

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    best_metric=None,
):
    path = Path(path)
    ensure_dir(path.parent)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"], strict=True)

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_parameters(model)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
