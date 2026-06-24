"""
Neuroimaging data utilities for AMR-HNR-Net.

Expected CSV format:
subject_id,image_path,label

Labels:
0 = CN
1 = MCI
2 = AD
"""

from pathlib import Path
from typing import Dict, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_manifest(csv_path):
    df = pd.read_csv(csv_path)

    required = {"subject_id", "image_path", "label"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df


def load_nifti_volume(path):
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Install nibabel using: pip install nibabel")

    img = nib.load(str(path))
    volume = img.get_fdata(dtype=np.float32)

    return volume.astype(np.float32)


def normalize_volume(volume, mode="zscore"):
    volume = volume.astype(np.float32)

    if mode == "none":
        return volume

    mask = volume != 0

    if mask.sum() == 0:
        mask = np.ones_like(volume, dtype=bool)

    if mode == "zscore":
        mean = volume[mask].mean()
        std = volume[mask].std() + 1e-6
        return (volume - mean) / std

    if mode == "minmax":
        min_v = volume[mask].min()
        max_v = volume[mask].max()
        return (volume - min_v) / (max_v - min_v + 1e-6)

    raise ValueError("Normalization must be zscore, minmax, or none.")


def resize_3d_tensor(x, size, mode="trilinear"):
    """
    x shape: [C, D, H, W]
    """

    x = x.unsqueeze(0)

    if mode == "nearest":
        x = F.interpolate(x, size=tuple(size), mode="nearest")
    else:
        x = F.interpolate(
            x,
            size=tuple(size),
            mode=mode,
            align_corners=False,
        )

    return x.squeeze(0)


class AMRHNRMRIDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_root="",
        input_size=(96, 96, 96),
        normalization="zscore",
    ):
        self.df = read_manifest(csv_path)
        self.data_root = Path(data_root)
        self.input_size = tuple(input_size)
        self.normalization = normalization

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_path):
        image_path = Path(image_path)

        if image_path.is_absolute():
            return image_path

        return self.data_root / image_path

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self._resolve_path(row["image_path"])

        if str(image_path).endswith(".npy"):
            volume = np.load(image_path).astype(np.float32)
        else:
            volume = load_nifti_volume(image_path)

        volume = normalize_volume(volume, self.normalization)

        volume = torch.from_numpy(volume).float().unsqueeze(0)
        volume = resize_3d_tensor(volume, self.input_size, mode="trilinear")

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "image": volume,
            "label": label,
            "subject_id": str(row["subject_id"]),
            "image_path": str(image_path),
        }


def load_aal_atlas_labels(atlas_path, input_size):
    """
    Load AAL-90 atlas registered to MNI152 space.
    """

    atlas_path = Path(atlas_path)

    if str(atlas_path).endswith(".npy"):
        atlas = np.load(atlas_path)
    else:
        atlas = load_nifti_volume(atlas_path)

    atlas = torch.from_numpy(atlas.astype(np.int64)).long()
    atlas = atlas.unsqueeze(0).float()

    atlas = resize_3d_tensor(
        atlas,
        input_size,
        mode="nearest",
    )

    return atlas.squeeze(0).long()


@torch.no_grad()
def classification_metrics(logits, labels, num_classes=3):
    preds = torch.argmax(logits, dim=1)

    cm = torch.zeros(
        num_classes,
        num_classes,
        device=logits.device,
        dtype=torch.float32,
    )

    for true, pred in zip(labels, preds):
        cm[true.long(), pred.long()] += 1

    eps = 1e-7

    accuracy = torch.trace(cm) / (cm.sum() + eps)

    precision_list = []
    sensitivity_list = []
    specificity_list = []
    f1_list = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp + eps)
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)

        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_list.append(f1)

    return {
        "accuracy": float(accuracy.cpu()),
        "precision": float(torch.stack(precision_list).mean().cpu()),
        "sensitivity": float(torch.stack(sensitivity_list).mean().cpu()),
        "specificity": float(torch.stack(specificity_list).mean().cpu()),
        "f1": float(torch.stack(f1_list).mean().cpu()),
    }


def save_attention_outputs(save_dir, subject_ids, slice_attention, probabilities):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    slice_attention = slice_attention.detach().cpu().numpy()
    probabilities = probabilities.detach().cpu().numpy()

    for i, sid in enumerate(subject_ids):
        np.save(save_dir / f"{sid}_slice_attention.npy", slice_attention[i])
        np.save(save_dir / f"{sid}_probabilities.npy", probabilities[i])
