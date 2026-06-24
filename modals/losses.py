import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss used in the paper.

    L_focal = - sum_c kappa_c (1 - y_hat_c)^gamma y_c log(y_hat_c)
    """

    def __init__(self, gamma: float = 2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)

        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()

        focal_weight = torch.pow(1.0 - prob, self.gamma)

        loss = -labels_one_hot * focal_weight * log_prob

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device).view(1, -1)
            loss = loss * weights

        loss = loss.sum(dim=1).mean()

        return loss


class GraphDirichletRegularization(nn.Module):
    """
    Graph Dirichlet energy regularization.

    L_graph = 1/2 Tr(H^T L H)
    where L = D - A.

    Input:
        graph_features: [B, M, F]
        adjacency:      [M, M]
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph_features, adjacency):
        degree = torch.diag(torch.sum(adjacency, dim=1))
        laplacian = degree - adjacency

        reg = torch.einsum("bmi,ij,bmj->b", graph_features, laplacian, graph_features)
        reg = 0.5 * reg.mean()

        return reg


class AMRHNRLoss(nn.Module):
    """
    Complete objective function according to the paper.

    L_total = L_focal + lambda * L_graph_reg + eta * L2
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights=None,
        lambda_graph: float = 0.01,
        eta_l2: float = 1e-5,
    ):
        super().__init__()

        self.focal_loss = FocalLoss(
            gamma=gamma,
            class_weights=class_weights,
        )

        self.graph_regularizer = GraphDirichletRegularization()

        self.lambda_graph = lambda_graph
        self.eta_l2 = eta_l2

    def forward(self, outputs, labels, model=None):
        logits = outputs["logits"]
        graph_features = outputs["graph_features"]
        adjacency = outputs["adjacency"]

        classification_loss = self.focal_loss(logits, labels)

        graph_loss = self.graph_regularizer(graph_features, adjacency)

        l2_loss = torch.tensor(0.0, device=logits.device)
        if model is not None:
            for param in model.parameters():
                if param.requires_grad:
                    l2_loss = l2_loss + torch.sum(param ** 2)

        total_loss = classification_loss + self.lambda_graph * graph_loss + self.eta_l2 * l2_loss

        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "graph_regularization_loss": graph_loss,
            "l2_loss": l2_loss,
        }
