from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class IBFTLossOutput:
    loss: torch.Tensor
    kl_loss: torch.Tensor
    z_ce_loss: torch.Tensor
    num_tokens: int


class VariationalBottleneck(nn.Module):
    """Gaussian bottleneck for IB-FT hidden-state regularization."""

    def __init__(self, hidden_size: int, z_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        z_dim = z_dim or max(1, hidden_size // 4)
        self.to_stats = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 * z_dim),
        )
        self.to_hidden = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.to_stats(hidden)
        mean, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-10.0, max=10.0)
        if self.training:
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            z = mean
        return self.to_hidden(z), mean, logvar


def prediction_positions(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> tuple[torch.Tensor, torch.Tensor]:
    valid = labels.ne(ignore_index).clone()
    valid[:, 0] = False
    batch_idx, label_pos = valid.nonzero(as_tuple=True)
    query_pos = label_pos - 1
    targets = labels[batch_idx, label_pos]
    return torch.stack([batch_idx, query_pos], dim=1), targets


def select_hidden_layer(hidden_states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
    if not hidden_states:
        raise ValueError("IB-FT requires output_hidden_states=True")
    idx = layer if layer >= 0 else len(hidden_states) + layer
    if idx < 0 or idx >= len(hidden_states):
        raise ValueError(f"invalid layer={layer}; got {len(hidden_states)} hidden-state tensors")
    return hidden_states[idx]


def compute_ibft_loss(
    *,
    hidden_states: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    lm_head: nn.Module,
    bottleneck: VariationalBottleneck,
    layer: int = -1,
    beta: float = 1.0,
    max_tokens: int = 2048,
    ignore_index: int = IGNORE_INDEX,
) -> IBFTLossOutput:
    hidden = select_hidden_layer(hidden_states, layer)
    positions, targets = prediction_positions(labels, ignore_index=ignore_index)
    if targets.numel() == 0:
        zero = hidden.new_zeros(())
        return IBFTLossOutput(zero, zero, zero, 0)

    if max_tokens > 0 and targets.numel() > max_tokens:
        choice = torch.randperm(targets.numel(), device=targets.device)[:max_tokens]
        positions = positions[choice]
        targets = targets[choice]

    h = hidden[positions[:, 0], positions[:, 1]]
    z_hidden, mean, logvar = bottleneck(h)
    kl_loss = (-0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)).mean()
    z_ce_loss = F.cross_entropy(lm_head(z_hidden).float(), targets, reduction="mean")
    return IBFTLossOutput(
        loss=kl_loss + beta * z_ce_loss,
        kl_loss=kl_loss,
        z_ce_loss=z_ce_loss,
        num_tokens=int(targets.numel()),
    )

