'''
Custom loss functions for clip-graph
'''

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_contrastive_loss(
    logits: torch.Tensor,
    sims: Optional[torch.Tensor] = None,
    sim_weights: Optional[str] = 'identity',
    alpha: Optional[float] = 0.1
) -> torch.Tensor:
    # we need a square matrix
    assert len(logits.shape) == 2 and logits.shape[0] == logits.shape[1]

    if sims is not None:
        assert sims.device == logits.device
        assert sims.shape == logits.shape
        assert alpha is not None
        assert sim_weights is not None
        assert sim_weights in ('identity', 'exp', 'exp_thick_tail')

    N, device = logits.shape[0], logits.device

    if sims is None:
        row_targets = torch.arange(N, device=device)
        col_targets = row_targets
    else:
        # label smoothing, but instead of the uniform distribution, use the
        # distribution based on similarity over nodes

        if sim_weights == 'identity':
            weights = torch.ones((N,), device=device)
        elif sim_weights == 'exp':
            weights = (-1 * torch.arange(N, device=device)).exp()
        else:  # sim_weights == 'exp_thick_tail'
            weights = (-1 * 1/(N ** 0.5) * torch.arange(N, device=device)).exp()

        weights = weights.unsqueeze(0).expand(N, -1)
        sort_inds = torch.argsort(sims, dim=1, descending=True)
        sims = (sims.gather(1, sort_inds) * weights)
        sims = sims.gather(1, sort_inds.argsort(1))

        row_reg_dist = sims - sims.min(dim=1).values.unsqueeze(1).expand(-1, N)
        row_reg_dist = F.normalize(row_reg_dist, p=1, dim=1)

        col_reg_dist = sims.T - sims.T.min(dim=1).values.unsqueeze(1).expand(-1, N)
        col_reg_dist = F.normalize(col_reg_dist, p=1, dim=1)

        row_targets = alpha * row_reg_dist + (1 - alpha) * torch.eye(N, device=device)
        col_targets = alpha * col_reg_dist + (1 - alpha) * torch.eye(N, device=device)

    return 0.5 * (
        F.cross_entropy(logits, row_targets) +
        F.cross_entropy(logits.T, col_targets)
    )


def paired_cosine_embedding_loss(
    x0: torch.Tensor,
    x1: torch.Tensor,

    **kwargs: Any
) -> torch.Tensor:
    assert x0.device == x1.device
    assert x0.shape[0] == x1.shape[0]

    n = x0.shape[0]
    device = x0.device

    if n == 1:
        return F.cosine_embedding_loss(x0, x1, torch.ones(1, device=device))

    neg_indices = torch.combinations(torch.arange(n, device=device))

    nrepeats = int(neg_indices.shape[0] / n)
    x0 = torch.cat([
        x0.repeat(nrepeats, 1),
        x0[neg_indices[:, 0], ...]
    ], dim=0)

    x1 = torch.cat([
        x1.repeat(nrepeats, 1),
        x1[neg_indices[:, 1], ...]
    ], dim=0)

    y = torch.cat([
        torch.ones(n * nrepeats, device=device),
        -1 * torch.ones(neg_indices.shape[0], device=device)
    ], dim=0)

    return F.cosine_embedding_loss(x0, x1, y, **kwargs)
