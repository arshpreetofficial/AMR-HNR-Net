"""
External validation script for AMR-HNR-Net.

Used for zero-shot testing on external cohorts such as OASIS or AIBL.
No fine-tuning is performed.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from amr_hnr_training_utils import ensure_dir, get_device, load_checkpoint, seed_everything
from neuroimaging_data_utils import (
    AMRHNRMRIDataset,
    classification_metrics,
    load_aal_atlas_labels,
    save_attention_outputs,
)

from modals import AMRHNRNet


def parse_size(value):
    return tuple(int(v.strip()) for v in value.split(","))


def build_parser():
    parser = argparse.ArgumentParser("External Validation for AMR-HNR-Net")

    parser.add_argument("--external_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/external_validation")

    parser.add_argument("--input_size", type=str, default="96,96,96")

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_rois", type=int, default=90)

    parser.add_argument("--feature_channels", type=int, default=32)
    parser.add_argument("--graph_hidden", type=int, default=128)
    parser.add_argument("--graph_out", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_attention", action="store_true")

    return parser


@torch.no_grad()
def main():
    args = build_parser().parse_args()

    seed_everything(args.seed)

    device = get_device(args.gpu_id)
    input_size = parse_size(args.input_size)

    output_dir = ensure_dir(args.output_dir)
    attention_dir = ensure_dir(output_dir / "slice_attention_outputs")

    dataset = AMRHNRMRIDataset(
        args.external_csv,
        data_root=args.data_root,
        input_size=input_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    atlas_labels = load_aal_atlas_labels(
        args.atlas_path,
        input_size,
    ).to(device)

    model = AMRHNRNet(
        in_channels=1,
        num_classes=args.num_classes,
        feature_channels=args.feature_channels,
        num_rois=args.num_rois,
        graph_hidden=args.graph_hidden,
        graph_out=args.graph_out,
        classifier_hidden=args.graph_out,
        dropout=args.dropout,
    ).to(device)

    load_checkpoint(
        args.checkpoint_path,
        model,
        map_location=device,
    )

    model.eval()

    all_logits = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        subject_ids = batch["subject_id"]

        outputs = model(images, atlas_labels)

        all_logits.append(outputs["logits"])
        all_labels.append(labels)

        if args.save_attention:
            save_attention_outputs(
                attention_dir,
                subject_ids,
                outputs["slice_attention"],
                outputs["probabilities"],
            )

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    metrics = classification_metrics(
        logits,
        labels,
        num_classes=args.num_classes,
    )

    result_file = output_dir / "external_validation_metrics.txt"

    with result_file.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            line = f"{key}: {value:.6f}"
            print(line)
            f.write(line + "\n")

    print(f"Saved external validation results to: {result_file}")


if __name__ == "__main__":
    main()
