import torch
import torch.nn as nn
import torch.nn.functional as F

from .arl_mrl import AdaptiveRepresentationLearning
from .hnr_reasoning import HierarchicalNeuroanatomicalReasoning


class AMRHNRNet(nn.Module):
    """
    AMR-HNR-Net according to the paper:

    Adaptive Multi-Resolution Neuroanatomical Representation Learning
    through Hierarchical Neuroanatomical Reasoning.

    Pipeline:
        Input 3D sMRI volume
            ↓
        Stage 1: Adaptive Representation Learning
            - Multi-Resolution Learning
            - Cross-Resolution Attention Fusion
            - Group Normalization
            ↓
        Stage 2: Hierarchical Neuroanatomical Reasoning
            - Slice-Level Attention
            - AAL Atlas ROI Aggregation
            - GCN Topological Reasoning
            ↓
        Classification Layer
            CN / MCI / AD

    Input:
        x:            [B, 1, D, H, W]
        atlas_labels: [D, H, W] or [B, D, H, W]

    Output:
        dictionary with logits, probabilities, attention, ROI and graph features.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        feature_channels: int = 32,
        num_rois: int = 90,
        graph_hidden: int = 128,
        graph_out: int = 128,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.arl = AdaptiveRepresentationLearning(
            in_channels=in_channels,
            feature_channels=feature_channels,
            downsample_factor=2,
            groups=8,
            reduction_ratio=8,
        )

        self.hnr = HierarchicalNeuroanatomicalReasoning(
            feature_channels=feature_channels,
            num_rois=num_rois,
            graph_hidden=graph_hidden,
            graph_out=graph_out,
            attention_dim=classifier_hidden,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(graph_out, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x, atlas_labels):
        normalized_features = self.arl(x)

        global_graph_feature, aux = self.hnr(normalized_features, atlas_labels)

        logits = self.classifier(global_graph_feature)

        probabilities = F.softmax(logits, dim=1)

        outputs = {
            "logits": logits,
            "probabilities": probabilities,
            "global_graph_feature": global_graph_feature,
            "normalized_features": normalized_features,
            "attended_features": aux["attended_features"],
            "slice_attention": aux["slice_attention"],
            "roi_features": aux["roi_features"],
            "graph_features": aux["graph_features"],
            "adjacency": aux["adjacency"],
        }

        return outputs


if __name__ == "__main__":
    model = AMRHNRNet(
        in_channels=1,
        num_classes=3,
        feature_channels=16,
        num_rois=90,
    )

    x = torch.randn(2, 1, 64, 64, 64)
    atlas_labels = torch.randint(1, 91, (64, 64, 64))

    outputs = model(x, atlas_labels)

    print("Logits:", outputs["logits"].shape)
    print("Probabilities:", outputs["probabilities"].shape)
    print("Slice Attention:", outputs["slice_attention"].shape)
    print("ROI Features:", outputs["roi_features"].shape)
    print("Graph Features:", outputs["graph_features"].shape)
