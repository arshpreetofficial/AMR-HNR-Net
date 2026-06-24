

import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from amr_hnr_objective import AMRHNRObjective, inverse_frequency_weights
from amr_hnr_training_utils import (
    AverageMeter,
    CSVLogger,
    EarlyStopping,
    ensure_dir,
    get_device,
    print_model_summary,
    save_checkpoint,
    seed_everything,
)
from neuroimaging_data_utils import (
    AMRHNRMRIDataset,
    classification_metrics,
    load_aal_atlas_labels,
)

from modals import AMRHNRNet


def parse_size(value):
    return tuple(int(v.strip()) for v in value.split(","))


def build_parser():
    parser = argparse.ArgumentParser("Train AMR-HNR-Net")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/amr_hnr")

    parser.add_argument("--input_size", type=str, default="96,96,96")

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_rois", type=int, default=90)
    parser.add_argument("--class_counts", type=str, default="320,380,250")

    parser.add_argument("--feature_channels", type=int, default=32)
    parser.add_argument("--graph_hidden", type=int, default=128)
    parser.add_argument("--graph_out", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lambda_graph", type=float, default=0.01)
    parser.add_argument("--eta_l2", type=float, default=1e-5)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    return parser


def train_one_epoch(model, loader, atlas_labels, criterion, optimizer, device, scaler=None):
    model.train()

    loss_meter = AverageMeter()
    all_logits = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                outputs = model(images, atlas_labels)
                loss_dict = criterion(outputs, labels, model=model)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images, atlas_labels)
            loss_dict = criterion(outputs, labels, model=model)
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        all_logits.append(outputs["logits"].detach())
        all_labels.append(labels.detach())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    metrics = classification_metrics(logits, labels, num_classes=3)
    metrics["loss"] = loss_meter.avg

    return metrics


@torch.no_grad()
def evaluate(model, loader, atlas_labels, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    all_logits = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images, atlas_labels)
        loss_dict = criterion(outputs, labels, model=None)

        loss_meter.update(loss_dict["total_loss"].item(), images.size(0))

        all_logits.append(outputs["logits"])
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    metrics = classification_metrics(logits, labels, num_classes=3)
    metrics["loss"] = loss_meter.avg

    return metrics


def main():
    args = build_parser().parse_args()

    seed_everything(args.seed)

    device = get_device(args.gpu_id)
    input_size = parse_size(args.input_size)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")

    train_dataset = AMRHNRMRIDataset(
        args.train_csv,
        data_root=args.data_root,
        input_size=input_size,
    )

    val_dataset = AMRHNRMRIDataset(
        args.val_csv,
        data_root=args.data_root,
        input_size=input_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    atlas_labels = load_aal_atlas_labels(args.atlas_path, input_size).to(device)

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

    print_model_summary(model)

    class_counts = [int(x.strip()) for x in args.class_counts.split(",")]
    class_weights = inverse_frequency_weights(class_counts)

    criterion = AMRHNRObjective(
        gamma=args.gamma,
        class_weights=class_weights,
        lambda_graph=args.lambda_graph,
        eta_l2=args.eta_l2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = GradScaler() if args.amp and torch.cuda.is_available() else None

    stopper = EarlyStopping(patience=30, mode="max")

    logger = CSVLogger(
        log_dir / "training_log.csv",
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_f1",
            "val_loss",
            "val_accuracy",
            "val_f1",
            "val_sensitivity",
            "val_specificity",
        ],
    )

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            atlas_labels,
            criterion,
            optimizer,
            device,
            scaler,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            atlas_labels,
            criterion,
            device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
            }
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            save_checkpoint(
                checkpoint_dir / "best_amr_hnr_net.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_f1,
            )

        if stopper.step(val_metrics["f1"]):
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
