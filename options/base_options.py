import argparse
import os

from .option_utils import (
    str2bool,
    parse_tuple,
    ensure_dir,
    save_options,
    set_random_seed,
    get_device,
    create_experiment_name,
)


class BaseOptions:
    """
    Base options for AMR-HNR-Net.

    These options define dataset paths, preprocessing settings,
    model configuration, and general runtime parameters.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # ------------------------------------------------------------------
        # Basic experiment information
        # ------------------------------------------------------------------
        parser.add_argument(
            "--project_name",
            type=str,
            default="AMR_HNR_Net",
            help="Project name.",
        )

        parser.add_argument(
            "--experiment_name",
            type=str,
            default=None,
            help="Experiment name. If None, timestamp-based name will be created.",
        )

        parser.add_argument(
            "--output_dir",
            type=str,
            default="./outputs",
            help="Directory to save logs, checkpoints and results.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed.",
        )

        # ------------------------------------------------------------------
        # Dataset paths
        # ------------------------------------------------------------------
        parser.add_argument(
            "--data_root",
            type=str,
            default="./data",
            help="Root directory containing preprocessed MRI data.",
        )

        parser.add_argument(
            "--train_csv",
            type=str,
            default="./manifests/train.csv",
            help="CSV file for training subjects.",
        )

        parser.add_argument(
            "--val_csv",
            type=str,
            default="./manifests/val.csv",
            help="CSV file for validation subjects.",
        )

        parser.add_argument(
            "--test_csv",
            type=str,
            default="./manifests/test.csv",
            help="CSV file for test/external cohort subjects.",
        )

        parser.add_argument(
            "--external_csv",
            type=str,
            default=None,
            help="Optional CSV for external testing such as OASIS/AIBL.",
        )

        parser.add_argument(
            "--image_column",
            type=str,
            default="image_path",
            help="CSV column containing MRI image path.",
        )

        parser.add_argument(
            "--label_column",
            type=str,
            default="label",
            help="CSV column containing class label.",
        )

        parser.add_argument(
            "--subject_column",
            type=str,
            default="subject_id",
            help="CSV column containing subject ID for leakage-free split.",
        )

        # ------------------------------------------------------------------
        # Class information
        # ------------------------------------------------------------------
        parser.add_argument(
            "--num_classes",
            type=int,
            default=3,
            help="Number of diagnostic classes: CN, MCI, AD.",
        )

        parser.add_argument(
            "--class_names",
            type=str,
            default="CN,MCI,AD",
            help="Comma-separated class names.",
        )

        parser.add_argument(
            "--class_counts",
            type=str,
            default="320,380,250",
            help="Class counts for CN, MCI, AD based on cohort distribution.",
        )

        # ------------------------------------------------------------------
        # Preprocessing and atlas settings
        # ------------------------------------------------------------------
        parser.add_argument(
            "--input_size",
            type=parse_tuple,
            default=(96, 96, 96),
            help="Input MRI volume size as D,H,W.",
        )

        parser.add_argument(
            "--in_channels",
            type=int,
            default=1,
            help="Input channels. T1-weighted sMRI is single-channel.",
        )

        parser.add_argument(
            "--atlas_path",
            type=str,
            default="./data/atlas/AAL90_MNI152.nii.gz",
            help="AAL atlas label map registered to MNI152 space.",
        )

        parser.add_argument(
            "--num_rois",
            type=int,
            default=90,
            help="Number of AAL atlas ROIs. Paper uses AAL-90.",
        )

        parser.add_argument(
            "--use_mni152",
            type=str2bool,
            default=True,
            help="Whether images are registered to MNI152 space.",
        )

        parser.add_argument(
            "--intensity_normalization",
            type=str,
            default="zscore",
            choices=["zscore", "minmax", "none"],
            help="MRI intensity normalization method.",
        )

        # ------------------------------------------------------------------
        # AMR-HNR-Net architecture settings
        # ------------------------------------------------------------------
        parser.add_argument(
            "--feature_channels",
            type=int,
            default=32,
            help="Intermediate feature channels C in ARL/MRL stage.",
        )

        parser.add_argument(
            "--downsample_factor",
            type=int,
            default=2,
            help="Spatial downsampling factor s for global branch.",
        )

        parser.add_argument(
            "--num_res_blocks",
            type=int,
            default=2,
            help="Number of residual 3D blocks in global/local branches.",
        )

        parser.add_argument(
            "--group_norm_groups",
            type=int,
            default=8,
            help="Number of groups for Group Normalization.",
        )

        parser.add_argument(
            "--craf_reduction_ratio",
            type=int,
            default=8,
            help="Bottleneck ratio r in CRAF channel attention.",
        )

        parser.add_argument(
            "--slice_attention_dim",
            type=int,
            default=128,
            help="Hidden dimension for slice-level attention.",
        )

        parser.add_argument(
            "--graph_hidden",
            type=int,
            default=128,
            help="Hidden dimension of GCN.",
        )

        parser.add_argument(
            "--graph_out",
            type=int,
            default=128,
            help="Output dimension of graph reasoning network.",
        )

        parser.add_argument(
            "--classifier_hidden",
            type=int,
            default=128,
            help="Hidden dimension of final classifier.",
        )

        parser.add_argument(
            "--dropout",
            type=float,
            default=0.3,
            help="Dropout rate.",
        )

        # ------------------------------------------------------------------
        # Runtime
        # ------------------------------------------------------------------
        parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            help="GPU ID.",
        )

        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of data loading workers.",
        )

        parser.add_argument(
            "--pin_memory",
            type=str2bool,
            default=True,
            help="Use pin_memory in DataLoader.",
        )

        parser.add_argument(
            "--use_amp",
            type=str2bool,
            default=True,
            help="Use automatic mixed precision training.",
        )

        parser.add_argument(
            "--verbose",
            type=str2bool,
            default=True,
            help="Print detailed options.",
        )

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        options, _ = parser.parse_known_args()
        self.parser = parser

        return options

    def print_options(self, options):
        message = ""
        message += "\n---------------- AMR-HNR-Net Options ----------------\n"

        for key, value in sorted(vars(options).items()):
            message += f"{key:30}: {value}\n"

        message += "-----------------------------------------------------\n"

        print(message)

        option_save_path = os.path.join(options.experiment_dir, "options.txt")
        with open(option_save_path, "w", encoding="utf-8") as f:
            f.write(message)

    def parse(self):
        options = self.gather_options()

        if options.experiment_name is None:
            options.experiment_name = create_experiment_name(options.project_name)

        options.experiment_dir = os.path.join(
            options.output_dir,
            options.experiment_name,
        )

        options.checkpoint_dir = os.path.join(options.experiment_dir, "checkpoints")
        options.log_dir = os.path.join(options.experiment_dir, "logs")
        options.result_dir = os.path.join(options.experiment_dir, "results")

        ensure_dir(options.experiment_dir)
        ensure_dir(options.checkpoint_dir)
        ensure_dir(options.log_dir)
        ensure_dir(options.result_dir)

        options.class_names = [x.strip() for x in options.class_names.split(",")]
        options.class_counts = [int(x.strip()) for x in options.class_counts.split(",")]

        options.device = get_device(options.gpu_id)

        set_random_seed(options.seed)

        save_options(
            options,
            os.path.join(options.experiment_dir, "options.json"),
        )

        if options.verbose:
            self.print_options(options)

        self.options = options
        return self.options
