from .base_options import BaseOptions
from .option_utils import str2bool, compute_inverse_class_weights


class TrainOptions(BaseOptions):
    """
    Training options for AMR-HNR-Net.

    Includes:
    - ADNI subject-level cross-validation
    - external zero-shot evaluation
    - focal loss
    - graph Dirichlet regularization
    - Adam optimizer
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # ------------------------------------------------------------------
        # Training protocol
        # ------------------------------------------------------------------
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "test", "external"],
            help="Run mode.",
        )

        parser.add_argument(
            "--cross_validation",
            type=str2bool,
            default=True,
            help="Use stratified subject-level cross-validation.",
        )

        parser.add_argument(
            "--num_folds",
            type=int,
            default=5,
            help="Number of cross-validation folds.",
        )

        parser.add_argument(
            "--fold_index",
            type=int,
            default=0,
            help="Fold index to run.",
        )

        parser.add_argument(
            "--subject_level_split",
            type=str2bool,
            default=True,
            help="Keep all scans from same subject in same fold.",
        )

        parser.add_argument(
            "--external_validation",
            type=str2bool,
            default=True,
            help="Evaluate external datasets such as OASIS/AIBL without fine-tuning.",
        )

        # ------------------------------------------------------------------
        # Optimization
        # ------------------------------------------------------------------
        parser.add_argument(
            "--epochs",
            type=int,
            default=300,
            help="Maximum number of training epochs.",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=2,
            help="Batch size for 3D MRI training.",
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="Initial learning rate.",
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-5,
            help="L2 weight decay.",
        )

        parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "adamw", "sgd"],
            help="Optimizer type.",
        )

        parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            choices=["cosine", "step", "plateau", "none"],
            help="Learning rate scheduler.",
        )

        parser.add_argument(
            "--early_stopping",
            type=str2bool,
            default=True,
            help="Use early stopping.",
        )

        parser.add_argument(
            "--patience",
            type=int,
            default=30,
            help="Early stopping patience.",
        )

        parser.add_argument(
            "--min_delta",
            type=float,
            default=1e-4,
            help="Minimum validation improvement for early stopping.",
        )

        # ------------------------------------------------------------------
        # Loss function settings
        # ------------------------------------------------------------------
        parser.add_argument(
            "--loss_name",
            type=str,
            default="amr_hnr_loss",
            choices=["focal", "cross_entropy", "amr_hnr_loss"],
            help="Loss function.",
        )

        parser.add_argument(
            "--focal_gamma",
            type=float,
            default=2.0,
            help="Focal loss focusing parameter gamma.",
        )

        parser.add_argument(
            "--use_class_weights",
            type=str2bool,
            default=True,
            help="Use inverse-frequency class weights.",
        )

        parser.add_argument(
            "--lambda_graph",
            type=float,
            default=0.01,
            help="Weight for graph Dirichlet regularization.",
        )

        parser.add_argument(
            "--eta_l2",
            type=float,
            default=1e-5,
            help="Weight for L2 regularization inside composite loss.",
        )

        # ------------------------------------------------------------------
        # Checkpoint and logging
        # ------------------------------------------------------------------
        parser.add_argument(
            "--save_freq",
            type=int,
            default=10,
            help="Save checkpoint every N epochs.",
        )

        parser.add_argument(
            "--eval_freq",
            type=int,
            default=1,
            help="Evaluate every N epochs.",
        )

        parser.add_argument(
            "--resume",
            type=str2bool,
            default=False,
            help="Resume training from checkpoint.",
        )

        parser.add_argument(
            "--resume_path",
            type=str,
            default=None,
            help="Checkpoint path for resume.",
        )

        parser.add_argument(
            "--save_best_only",
            type=str2bool,
            default=True,
            help="Save only best validation checkpoint.",
        )

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        parser.add_argument(
            "--primary_metric",
            type=str,
            default="accuracy",
            choices=["accuracy", "f1", "auc", "loss"],
            help="Metric used to select best model.",
        )

        parser.add_argument(
            "--compute_auc",
            type=str2bool,
            default=True,
            help="Compute AUC for classification.",
        )

        parser.add_argument(
            "--compute_confusion_matrix",
            type=str2bool,
            default=True,
            help="Save confusion matrix.",
        )

        parser.add_argument(
            "--save_attention_maps",
            type=str2bool,
            default=True,
            help="Save slice attention weights and ROI graph outputs.",
        )

        return parser

    def parse(self):
        options = super().parse()

        if options.use_class_weights:
            options.class_weights = compute_inverse_class_weights(options.class_counts)
        else:
            options.class_weights = None

        return options
