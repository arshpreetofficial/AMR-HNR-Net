import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_layers import GraphReasoningNetwork


class SliceLevelAttention(nn.Module):
    """
    Slice-level attention according to the HNR stage.

    Paper equations:
        vd = GAP(sd)
        ed = w_att^T LeakyReLU(W_att vd + b_att)
        alpha_d = softmax(ed)
        F_hat(:, d, :, :) = alpha_d * sd

    Input:
        x: [B, C, D, H, W]

    Output:
        weighted_volume: [B, C, D, H, W]
        alpha:           [B, D]
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()

        self.projection = nn.Linear(channels, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        b, c, d, h, w = x.shape

        slice_descriptors = torch.mean(x, dim=(3, 4))       # [B, C, D]
        slice_descriptors = slice_descriptors.permute(0, 2, 1)  # [B, D, C]

        attention_hidden = self.activation(self.projection(slice_descriptors))
        attention_scores = self.score_layer(attention_hidden).squeeze(-1)

        alpha = F.softmax(attention_scores, dim=1)

        weighted_volume = x * alpha.view(b, 1, d, 1, 1)

        return weighted_volume, alpha


class AALRegionAggregation(nn.Module):
    """
    Atlas-guided region-level anatomical aggregation.

    Paper:
        AAL atlas with M = 90 ROIs.
        r_k = masked global pooling over ROI k.

    Input:
        x: [B, C, D, H, W]
        atlas_labels: [D, H, W] with labels 1 to M
                      or [B, D, H, W]

    Output:
        roi_features: [B, M, C]
    """

    def __init__(self, num_rois: int = 90, eps: float = 1e-6):
        super().__init__()
        self.num_rois = num_rois
        self.eps = eps

    def forward(self, x, atlas_labels):
        b, c, d, h, w = x.shape

        if atlas_labels.dim() == 3:
            atlas_labels = atlas_labels.unsqueeze(0).repeat(b, 1, 1, 1)

        if atlas_labels.dim() != 4:
            raise ValueError("atlas_labels must be [D,H,W] or [B,D,H,W]")

        atlas_labels = atlas_labels.to(device=x.device)

        if atlas_labels.shape[-3:] != (d, h, w):
            atlas_labels_float = atlas_labels.float().unsqueeze(1)
            atlas_labels_float = F.interpolate(
                atlas_labels_float,
                size=(d, h, w),
                mode="nearest",
            )
            atlas_labels = atlas_labels_float.squeeze(1).long()

        roi_feature_list = []

        for roi_id in range(1, self.num_rois + 1):
            mask = (atlas_labels == roi_id).float()  # [B, D, H, W]
            mask_sum = mask.sum(dim=(1, 2, 3), keepdim=True) + self.eps

            mask = mask.unsqueeze(1)  # [B, 1, D, H, W]

            pooled = (x * mask).sum(dim=(2, 3, 4)) / mask_sum.view(b, 1)
            roi_feature_list.append(pooled)

        roi_features = torch.stack(roi_feature_list, dim=1)  # [B, M, C]

        return roi_features


class HierarchicalNeuroanatomicalReasoning(nn.Module):
    """
    Stage 2 of AMR-HNR-Net.

    Components:
        1. Slice-level attention
        2. AAL atlas-guided ROI aggregation
        3. GCN-based graph-level topological reasoning
        4. Global node pooling

    Input:
        features:     [B, C, D, H, W]
        atlas_labels: [D, H, W] or [B, D, H, W]

    Output:
        global_graph_feature: [B, graph_out]
        auxiliary dictionary
    """

    def __init__(
        self,
        feature_channels: int = 32,
        num_rois: int = 90,
        graph_hidden: int = 128,
        graph_out: int = 128,
        attention_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_rois = num_rois

        self.slice_attention = SliceLevelAttention(
            channels=feature_channels,
            attention_dim=attention_dim,
        )

        self.region_aggregation = AALRegionAggregation(
            num_rois=num_rois,
        )

        self.graph_network = GraphReasoningNetwork(
            in_features=feature_channels,
            hidden_features=graph_hidden,
            out_features=graph_out,
            dropout=dropout,
        )

        adjacency = torch.eye(num_rois)
        self.learnable_adjacency = nn.Parameter(adjacency)

    def get_adjacency(self):
        adjacency = torch.sigmoid(self.learnable_adjacency)
        adjacency = 0.5 * (adjacency + adjacency.t())
        return adjacency

    def forward(self, features, atlas_labels):
        attended_features, slice_attention_weights = self.slice_attention(features)

        roi_features = self.region_aggregation(attended_features, atlas_labels)

        adjacency = self.get_adjacency()

        graph_features = self.graph_network(roi_features, adjacency)

        global_graph_feature = torch.mean(graph_features, dim=1)

        aux = {
            "attended_features": attended_features,
            "slice_attention": slice_attention_weights,
            "roi_features": roi_features,
            "graph_features": graph_features,
            "adjacency": adjacency,
        }

        return global_graph_feature, aux
