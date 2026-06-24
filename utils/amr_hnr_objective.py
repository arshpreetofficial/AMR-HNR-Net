

from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.gamma = gamma

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        labels_onehot = F.one_hot(labels.long(), num_classes=logits.size(1)).float()
        focal_weight = torch.pow(1.0 - probs, self.gamma)

        loss = -labels_onehot * focal_weight * log_probs

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device).view(1, -1)
            loss = loss * weights

        return loss.sum(dim=1).mean()


class GraphDirichletRegularization(nn.Module):
    """
    Graph regularization:
    L_graph = 1/2 Tr(H^T L H)
    """

    def forward(self, graph_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.to(graph_features.device)
        adjacency = 0.5 * (adjacency + adjacency.t())
        adjacency = torch.clamp(adjacency, min=0.0)

        degree = torch.diag(adjacency.sum(dim=1))
        laplacian = degree - adjacency

        energy = torch.einsum(
            "bmc,mn,bnc->b",
            graph_features,
            laplacian,
            graph_features,
        )

        return 0.5 * energy.mean()


def l2_parameter_penalty(model: nn.Module) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        return torch.tensor(0.0)

    penalty = torch.zeros((), device=params[0].device)

    for p in params:
        penalty = penalty + torch.sum(p ** 2)

    return penalty


class AMRHNRObjective(nn.Module):
    """
    Complete AMR-HNR-Net loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        lambda_graph: float = 0.01,
        eta_l2: float = 1e-5,
    ):
        super().__init__()

        self.focal_loss = MultiClassFocalLoss(
            gamma=gamma,
            class_weights=class_weights,
        )

        self.graph_regularizer = GraphDirichletRegularization()

        self.lambda_graph = lambda_graph
        self.eta_l2 = eta_l2

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:

        logits = outputs["logits"]
        graph_features = outputs["graph_features"]
        adjacency = outputs["adjacency"]

        focal = self.focal_loss(logits, labels)
        graph_reg = self.graph_regularizer(graph_features, adjacency)

        if model is not None:
            l2_loss = l2_parameter_penalty(model)
        else:
            l2_loss = torch.zeros((), device=logits.device)

        total = focal + self.lambda_graph * graph_reg + self.eta_l2 * l2_loss

        return {
            "total_loss": total,
            "focal_loss": focal.detach(),
            "graph_regularization_loss": graph_reg.detach(),
            "l2_loss": l2_loss.detach(),
        }


def inverse_frequency_weights(class_counts):
    """
    Example:
    CN=320, MCI=380, AD=250
    """

    counts = torch.tensor(class_counts, dtype=torch.float32)

    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()

    return weights.tolist()
