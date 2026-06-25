"""
Subject-level cross-validation for AMR-HNR-Net.

This prevents slice-level or subject-level data leakage.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from amr_hnr_training_utils import ensure_dir, seed_everything


def build_parser():
    parser = argparse.ArgumentParser("AMR-HNR-Net Subject-Level Cross Validation")

    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/cross_validation")

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser


def create_folds(manifest_csv, output_dir, num_folds, seed):
    df = pd.read_csv(manifest_csv)

    subject_df = df.groupby("subject_id").first().reset_index()

    subjects = subject_df["subject_id"].values
    labels = subject_df["label"].values

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_dir = ensure_dir(Path(output_dir) / "fold_manifests")
    fold_paths = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])

        train_df = df[df["subject_id"].isin(train_subjects)]
        val_df = df[df["subject_id"].isin(val_subjects)]

        train_csv = fold_dir / f"fold_{fold_idx}_train.csv"
        val_csv = fold_dir / f"fold_{fold_idx}_val.csv"

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        fold_paths.append((train_csv, val_csv))

    return fold_paths


def main():
    args = build_parser().parse_args()

    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)

    fold_paths = create_folds(
        args.manifest_csv,
        output_dir,
        args.num_folds,
        args.seed,
    )

    for fold_idx, (train_csv, val_csv) in enumerate(fold_paths):
        fold_output = output_dir / f"fold_{fold_idx}"

        command = [
            sys.executable,
            "train_amr_hnr_net.py",
            "--train_csv", str(train_csv),
            "--val_csv", str(val_csv),
            "--data_root", args.data_root,
            "--atlas_path", args.atlas_path,
            "--output_dir", str(fold_output),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--gpu_id", str(args.gpu_id),
        ]

        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
