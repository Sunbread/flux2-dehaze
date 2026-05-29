from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(
    model_pred: torch.Tensor,
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Flow Matching MSE Loss.

    z_t = (1 - sigma) * x_clean + sigma * noise
    target = noise - x_clean
    """
    target = noise - clean_latent
    return F.mse_loss(model_pred.float(), target.float())
