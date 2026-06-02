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
    """Gaussian bottleneck used by the IB-FT benchmark baseline."""

    def __init__(self, hidden_size: int, z_dim: int, dropout: float = 0.0):
        super().__init__()
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
            eps = torch.randn_like(mean)
            z = mean + eps * torch.exp(0.5 * logvar)
        else:
            z = mean
        hidden_z = self.to_hidden(z)
        return hidden_z, mean, logvar


def _select_hidden_layer(hidden_states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
    if not hidden_states:
        raise ValueError("IB-FT requires output_hidden_states=True, but model returned no hidden states.")
    idx = layer if layer >= 0 else len(hidden_states) + layer
    if idx < 0 or idx >= len(hidden_states):
        raise ValueError(
            f"Invalid ib_layer={layer}; model returned {len(hidden_states)} hidden-state tensors."
        )
    return hidden_states[idx]


def _prediction_positions(labels: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hidden query positions q and target labels y for valid next-token labels."""
    valid = labels.ne(ignore_index)
    valid[:, 0] = False
    batch_idx, label_pos = valid.nonzero(as_tuple=True)
    query_pos = label_pos - 1
    targets = labels[batch_idx, label_pos]
    return torch.stack([batch_idx, query_pos], dim=1), targets


def compute_ibft_loss(
    *,
    hidden_states: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    lm_head: nn.Module,
    bottleneck: VariationalBottleneck,
    layer: int = 20,
    beta: float = 1.0,
    max_tokens: int = 2048,
    ignore_index: int = IGNORE_INDEX,
) -> IBFTLossOutput:
    """Compute the IB-FT auxiliary loss on sampled supervised token positions.

    Standard causal LM loss is still computed by the model.  This auxiliary loss
    uses the hidden state at position q to predict the supervised token at q+1.
    """
    selected_hidden = _select_hidden_layer(hidden_states, layer)
    positions, targets = _prediction_positions(labels, ignore_index)

    num_tokens = int(targets.numel())
    if num_tokens == 0:
        zero = selected_hidden.new_zeros(())
        return IBFTLossOutput(loss=zero, kl_loss=zero, z_ce_loss=zero, num_tokens=0)

    if max_tokens > 0 and num_tokens > max_tokens:
        choice = torch.randperm(num_tokens, device=targets.device)[:max_tokens]
        positions = positions[choice]
        targets = targets[choice]

    h = selected_hidden[positions[:, 0], positions[:, 1]]
    hidden_z, mean, logvar = bottleneck(h)

    kl_per_token = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)
    kl_loss = kl_per_token.mean()

    logits_z = lm_head(hidden_z)
    z_ce_loss = F.cross_entropy(logits_z.float(), targets, reduction="mean")

    loss = kl_loss + beta * z_ce_loss
    return IBFTLossOutput(
        loss=loss,
        kl_loss=kl_loss,
        z_ce_loss=z_ce_loss,
        num_tokens=int(targets.numel()),
    )
