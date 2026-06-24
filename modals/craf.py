import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossResolutionAttentionFusion(nn.Module):
    
    Cross-Resolution Attention Fusion module according to the paper.

    Equation implemented:

        F_global' = T(F_global)

        wc = sigmoid(MLP(GAP(F_global')) + MLP(GAP(F_local)))

        S = sigmoid(C3D([F_global' ; F_local]))

        F_fused = wc * (F_global' + S * F_local)

    Input:
        global_feature: [B, C, D/s, H/s, W/s]
        local_feature:  [B, C, D, H, W]

    Output:
        fused_feature:  [B, C, D, H, W]
    

    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()

        hidden_channels = max(channels // reduction_ratio, 4)

        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels * 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, global_feature, local_feature):
        if global_feature.shape[-3:] != local_feature.shape[-3:]:
            global_feature = F.interpolate(
                global_feature,
                size=local_feature.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        global_descriptor = torch.mean(global_feature, dim=(2, 3, 4))
        local_descriptor = torch.mean(local_feature, dim=(2, 3, 4))

        channel_attention = self.shared_mlp(global_descriptor) + self.shared_mlp(local_descriptor)
        channel_attention = torch.sigmoid(channel_attention)
        channel_attention = channel_attention.view(
            local_feature.size(0),
            local_feature.size(1),
            1,
            1,
            1,
        )

        spatial_input = torch.cat([global_feature, local_feature], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)

        fused_feature = channel_attention * (global_feature + spatial_attention * local_feature)

        return fused_feature
