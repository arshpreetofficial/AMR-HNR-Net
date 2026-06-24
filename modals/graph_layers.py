import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGraphConvolution(nn.Module):
    """
    Spectral graph convolution used in HNR stage.

    Equation:
        H(l+1) = ReLU(D^-1/2 A_hat D^-1/2 H(l) Theta(l))

    Input:
        node_features: [B, M, Fin]
        adjacency:     [M, M]

    Output:
        output:        [B, M, Fout]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @staticmethod
    def normalize_adjacency(adjacency: torch.Tensor):
        m = adjacency.size(0)

        identity = torch.eye(m, device=adjacency.device, dtype=adjacency.dtype)
        adjacency_hat = adjacency + identity

        degree = torch.sum(adjacency_hat, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)

        adjacency_norm = degree_inv_sqrt @ adjacency_hat @ degree_inv_sqrt

        return adjacency_norm

    def forward(self, node_features, adjacency):
        adjacency_norm = self.normalize_adjacency(adjacency)

        support = self.linear(node_features)

        output = torch.einsum("ij,bjf->bif", adjacency_norm, support)

        return output


class GraphReasoningNetwork(nn.Module):
    """
    GCN-based topological reasoning block.

    Input:
        roi_features: [B, M, C]
        adjacency:    [M, M]

    Output:
        graph_features: [B, M, Cout]
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        out_features: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gcn1 = SpectralGraphConvolution(in_features, hidden_features)
        self.gcn2 = SpectralGraphConvolution(hidden_features, out_features)

        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, roi_features, adjacency):
        x = self.gcn1(roi_features, adjacency)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)

        x = self.gcn2(x, adjacency)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)

        return x
