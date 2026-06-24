import torch
import torch.nn as nn
import torch.nn.functional as F

from .craf import CrossResolutionAttentionFusion


class Residual3DBlock(nn.Module):
    """
    3D residual block used in both global and local branches.
    This follows the paper's idea of residual 3D convolutional feature extraction.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()

        group_count = min(groups, channels)
        if channels % group_count != 0:
            group_count = 1

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(group_count, channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(group_count, channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ConvStem3D(nn.Module):
    """
    Initial 3D convolution stem.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, groups: int = 8):
        super().__init__()

        group_count = min(groups, out_channels)
        if out_channels % group_count != 0:
            group_count = 1

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class MultiResolutionLearning(nn.Module):
    """
    Multi-Resolution Learning module from the paper.

    Global branch:
        Low-resolution macro-structural stream.

    Local branch:
        High-resolution micro-textural stream.

    Input:
        x: [B, 1, D, H, W]

    Output:
        global_feature: [B, C, D/s, H/s, W/s]
        local_feature:  [B, C, D, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        downsample_factor: int = 2,
        num_res_blocks: int = 2,
        groups: int = 8,
    ):
        super().__init__()

        self.downsample_factor = downsample_factor
        self.global_downsample = nn.AvgPool3d(
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )

        self.global_stem = ConvStem3D(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=5,
            padding=2,
            groups=groups,
        )

        self.local_stem = ConvStem3D(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
        )

        self.global_blocks = nn.Sequential(
            *[Residual3DBlock(feature_channels, groups=groups) for _ in range(num_res_blocks)]
        )

        self.local_blocks = nn.Sequential(
            *[Residual3DBlock(feature_channels, groups=groups) for _ in range(num_res_blocks)]
        )

    def forward(self, x):
        x_global = self.global_downsample(x)

        global_feature = self.global_stem(x_global)
        global_feature = self.global_blocks(global_feature)

        local_feature = self.local_stem(x)
        local_feature = self.local_blocks(local_feature)

        return global_feature, local_feature


class AdaptiveRepresentationLearning(nn.Module):
    """
    Stage 1 of AMR-HNR-Net.

    Paper components:
        1. Multi-Resolution Learning
        2. Cross-Resolution Attention Fusion
        3. Group Normalization

    Input:
        x: [B, 1, D, H, W]

    Output:
        normalized_features: [B, C, D, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        downsample_factor: int = 2,
        groups: int = 8,
        reduction_ratio: int = 8,
    ):
        super().__init__()

        self.mrl = MultiResolutionLearning(
            in_channels=in_channels,
            feature_channels=feature_channels,
            downsample_factor=downsample_factor,
            groups=groups,
        )

        self.craf = CrossResolutionAttentionFusion(
            channels=feature_channels,
            reduction_ratio=reduction_ratio,
        )

        group_count = min(groups, feature_channels)
        if feature_channels % group_count != 0:
            group_count = 1

        self.group_norm = nn.GroupNorm(group_count, feature_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        global_feature, local_feature = self.mrl(x)

        fused_feature = self.craf(global_feature, local_feature)

        normalized_feature = self.group_norm(fused_feature)
        normalized_feature = self.activation(normalized_feature)

        return normalized_feature
