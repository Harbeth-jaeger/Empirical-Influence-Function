from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

def _unwrap_module(m):
    while hasattr(m, "module"):
        m = m.module
    return m



def _unwrap_to_decoder_stack(model):
    """
    Walk through HF / PEFT / DDP wrappers and return the module that owns `.layers` (ecoderLayer list).
    """
    m = model
    # DDP / DeepSpeed wrappers
    if hasattr(m, "module") and not hasattr(m, "layers"):
        m = m.module
    while True:
        if hasattr(m, "layers"):
            return m
        if hasattr(m, "model"):
            m = m.model
            continue
        if hasattr(m, "base_model"):
            m = m.base_model
            continue
        raise RuntimeError(f"Cannot locate decoder layer stack on {type(model).__name__}.")



def build_contribution_matrix(
    model,
    last_hidden_in: Tensor,   # [B, T, D]   input to the last decoder layer
    attn_probs: Tensor,       # [B, H, T, T] Step 1
) -> Tensor:
    """
        Strict Kobayashi/ALTI contribution c_{i,j} = ||T_i(x_j)||_2 for the last layer.
    """
    decoder = _unwrap_to_decoder_stack(model)
    layer = _unwrap_module(decoder.layers[-1]) # last decoder block
    self_attn = layer.self_attn

    B, T, D = last_hidden_in.shape
    H = attn_probs.size(1)
    device = last_hidden_in.device
    dtype = last_hidden_in.dtype
    compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32

    head_dim = getattr(self_attn, "head_dim", None) or (
        self_attn.q_proj.weight.shape[0] // H
    ) 
    v_out = self_attn.v_proj.weight.shape[0]
    num_kv_heads = v_out // head_dim
    assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
    n_rep = H // num_kv_heads

    gamma = layer.input_layernorm.weight.to(device).float()           # [D]
    gamma_x = last_hidden_in.float() * gamma                          # [B, T, D]
    v_w = self_attn.v_proj.weight.to(device).float()                  # [num_kv_heads*head_dim, D]
    v_b = self_attn.v_proj.bias
    v_proj = gamma_x @ v_w.t()                                        # [B, T, num_kv_heads*head_dim]
    if v_b is not None:
        v_proj = v_proj + v_b.to(device).float()
    v_states = v_proj.view(B, T, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    if n_rep > 1:
        v_states = (
            v_states.unsqueeze(2)
            .expand(B, num_kv_heads, n_rep, T, head_dim)
            .reshape(B, H, T, head_dim)
        )

    o_w = self_attn.o_proj.weight.to(device).float()    
    o_w_by_head = o_w.view(D, H, head_dim)
    transformed = torch.einsum("bhsd,ohd->bhso", v_states, o_w_by_head)

    #  Σ_h A^h_{i,j} · (W_O^h W_V^h x_j)
    attn_f = attn_probs.float()                                       # [B, H, T, T]
    T_ij = torch.einsum("bhij,bhjo->bijo", attn_f, transformed)       # [B, T, T, D]

    # ignore the diagonals (j == i branch)
    diag_idx = torch.arange(T, device=device)
    T_ij[:, diag_idx, diag_idx, :] = (
        T_ij[:, diag_idx, diag_idx, :] + last_hidden_in.float()[:, diag_idx, :]
    )

    # divide by σ_i = RMS(x_i)  (query-side) 
    eps_rms = getattr(layer.input_layernorm, "variance_epsilon", 1e-6)
    sigma_i = last_hidden_in.float().pow(2).mean(dim=-1).add(eps_rms).sqrt()  # [B, T]
    T_ij = T_ij / sigma_i.unsqueeze(-1).unsqueeze(-1).clamp_min(1e-12)

    # Take L2 norm for easier optimization!
    c = T_ij.norm(dim=-1, p=2)                                        # [B, T, T]
    return c.to(dtype)


@dataclass
class SaliencyDiagnostics:
    loss: Tensor
    avg_C: float
    avg_N: float
    avg_ratio: float
    n_queries: int
    n_samples: int
    floor_eps: float = 0.0
    batch_floor_eps: float = 0.0
    floor_eps_mode: str = "fixed"
    floor_eps_step: int = 0
    floor_eps_warmup_steps: int = 0
    floor_logit_eps: float = 0.0
    floor_eps_kind: str = "saliency"
    loss_type: str = "softmax_margin"


def canonical_saliency_loss_type(loss_type: str | None) -> str:
    """Canonical public saliency-loss names.

    `softmax_margin` is the former `infonce_floor` implementation: log(C+eps)/tau
    with a negative floor. Keep old aliases readable so previous commands still run.
    """
    value = (loss_type or "softmax_margin").strip().lower().replace("-", "_")
    if value in {"", "default", "softmax_margin", "softmax_margin_loss", "infonce", "nll", "floor", "infonce_floor"}:
        return "softmax_margin"
    if value in {"softmax", "softmax_loss"}:
        return "softmax"
    if value in {"margin_bce", "bce_margin", "decoupled_bce", "per_edge_bce"}:
        return "margin_bce"
    if value in {"ranknet", "pairwise", "pairwise_ranknet", "rank_net"}:
        return "ranknet"
    if value in {"contrastive", "triplet", "pair_hinge", "pairwise_hinge"}:
        return "contrastive"
    raise ValueError(f"Unsupported saliency loss_type={loss_type!r}; expected softmax_margin, softmax, margin_bce, ranknet, or contrastive.")


def saliency_loss_display_name(loss_type: str | None) -> str:
    kind = canonical_saliency_loss_type(loss_type)
    if kind == "softmax":
        return "softmax loss"
    if kind == "margin_bce":
        return "decoupled margin-BCE loss"
    if kind == "ranknet":
        return "pairwise RankNet loss"
    if kind == "contrastive":
        return "pairwise contrastive (triplet hinge) loss"
    return "softmax-margin loss"


def flatten_annot_pairs(
    annot_pairs_batch: list[Tensor] | Tensor,
    device,
) -> Tensor:
    """
    Flatten annotation edges to [B, 3] of int64 columns
    (batch_idx, pos_a, pos_b). Accepts:

      • list[Tensor]   — per-sample [Number of pairs, 2] tensors (the dataset collator form)
      • Tensor [B, 3]  — already flat (batch_idx, pos_a, pos_b)
      • Tensor [B, M, 2] — padded with -1 for empty slots
    """
    if isinstance(annot_pairs_batch, torch.Tensor):
        t = annot_pairs_batch.to(device=device, dtype=torch.long)
        if t.dim() == 2 and t.size(-1) == 3:
            return t
        if t.dim() == 3 and t.size(-1) == 2:
            B_, M, _ = t.shape
            batch_idx = torch.arange(B_, device=device).unsqueeze(1).expand(B_, M)
            flat = torch.stack([batch_idx, t[..., 0], t[..., 1]], dim=-1).reshape(-1, 3)
            keep = (flat[:, 1] >= 0) & (flat[:, 2] >= 0)
            return flat[keep]
        raise ValueError(f"Unsupported tensor shape {tuple(t.shape)} for annot_pairs.")

    # list[Tensor]
    chunks = []
    for b, pairs in enumerate(annot_pairs_batch):
        if pairs is None or pairs.numel() == 0:
            continue
        p = pairs.to(device=device, dtype=torch.long)
        bid = torch.full((p.size(0),), b, device=device, dtype=torch.long)
        chunks.append(torch.stack([bid, p[:, 0], p[:, 1]], dim=-1))
    if not chunks:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    return torch.cat(chunks, dim=0)


def compute_saliency_loss(
        C: Tensor,  # [B, T, T]
        annot_pairs: list[Tensor] | Tensor,  # flat [B, 3] or list of [Number of edges, 2]
        *,
        alpha: float,
        eps: float,
        floor_eps: float = 0.0,
        floor_logit_eps: float | None = None,
        loss_type: str = "softmax_margin",
) -> SaliencyDiagnostics:
    """
    Multi-positive NLL over causal sources, fully batched across samples.

    `annot_pairs` may be either
      • a flat tensor [B, 3] with columns (batch_idx, pos_a, pos_b), or
      • a list of [Number of edges, 2] tensors (for batch size = 1, one sample only).

    Cbar_i and Nbar_i are retained as diagnostics. loss_type selects either
    the softmax-margin log-saliency/floor objective or raw saliency softmax loss.

    Returns SaliencyDiagnostics.
    If no valid query rows across the batch, loss=0.
    """
    B, T, _ = C.shape
    device = C.device
    dtype = C.dtype

    # Normalize input to a flat [B, 3] tensor
    flat = flatten_annot_pairs(annot_pairs, device=device)
    if flat.numel() == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0, avg_N=0.0, avg_ratio=0.0,
            n_queries=0, n_samples=0,
        )

    batch_ids = flat[:, 0]
    p0 = flat[:, 1]
    p1 = flat[:, 2]

    # Swap so source ≤ query (respect causal direction).
    src_all = torch.minimum(p0, p1)
    qry_all = torch.maximum(p0, p1)

    # Filter out-of-range indices
    keep = (
            (src_all < T) &
            (qry_all < T) & (qry_all > src_all) &
            (batch_ids >= 0) & (batch_ids < B)
    )
    batch_ids = batch_ids[keep]
    src_all = src_all[keep]
    qry_all = qry_all[keep]

    if batch_ids.numel() == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0, avg_N=0.0, avg_ratio=0.0,
            n_queries=0, n_samples=0,
        )

    # Group edges by (batch, query) → unique rows Q
    keys = batch_ids * T + qry_all  # [B]
    unique_keys, inv = torch.unique(keys, return_inverse=True)  # inv: [B] → [0..Q)
    Q = unique_keys.numel()
    row_batch = unique_keys // T  # [Q]
    row_qry = unique_keys % T  # [Q]

    # A_adj[q, s] counts how many annotated edges of row q point to source s.
    A_adj = torch.zeros(Q, T, device=device, dtype=dtype)
    A_adj.index_put_(
        (inv, src_all),
        torch.ones_like(src_all, dtype=dtype),
        accumulate=True,
    )
    A_adj_bin = (A_adj > 0).to(dtype)  # binary mask, [Q, T]

    # Causal Mask
    src_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    qry_col = row_qry.unsqueeze(1)  # [Q, 1]
    M_causal = (
            (src_idx <= qry_col) &
            (src_idx != qry_col)
    ).to(dtype)  # [Q, T]

    # Gather c_{i,j} rows for each unique (b, q)
    C_rows = C[row_batch, row_qry, :]  # [Q, T]

    # C̄_i and N̄_i with the standard split
    annot_mask = M_causal * A_adj_bin  # [Q, T]
    non_annot_mask = M_causal * (1.0 - A_adj_bin)  # [Q, T]

    den_C = annot_mask.sum(dim=-1).clamp_min(eps)
    den_N = non_annot_mask.sum(dim=-1).clamp_min(eps)
    C_bar = (annot_mask * C_rows).sum(dim=-1) / den_C            # [Q] 仅诊断用
    N_bar = (non_annot_mask * C_rows).sum(dim=-1) / den_N        # [Q] 干扰基线

    # Only train queries that have at least one positive AND at least one negative.
    I_mask = (annot_mask.sum(dim=-1) > 0) & (non_annot_mask.sum(dim=-1) > 0)
    if not bool(I_mask.any()):
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0, avg_N=0.0, avg_ratio=0.0,
            n_queries=0, n_samples=int(batch_ids.unique().numel()),
        )
        
    # V2 loss: Hinge loss
    # C_rows_I = C_rows[I_mask]                                    # [Qi, T]
    # annot_I = annot_mask[I_mask]                                 # [Qi, T]
    # N_bar_I = N_bar[I_mask]                                      # [Qi]

    # # 逐标注边 hinge:每条标注边都要高过 alpha * N̄ 的相对 margin
    # target = alpha * N_bar_I.unsqueeze(-1)                       # [Qi, 1]
    # margin = F.relu(target - C_rows_I) * annot_I                 # [Qi, T] 只在标注边上罚
    # loss = margin.sum(dim=-1) / annot_I.sum(dim=-1).clamp_min(eps)
    # loss = loss.mean()

    # C_bar_I = C_bar[I_mask]

    # with torch.no_grad():
    #     ratio = C_bar_I / (N_bar_I + eps)
    #     diag = SaliencyDiagnostics(
    #         loss=loss,
    #         avg_C=float(C_bar_I.mean()),
    #         avg_N=float(N_bar_I.mean()),
    #         avg_ratio=float(ratio.mean()),
    #         n_queries=int(C_bar_I.numel()),
    #         n_samples=int(row_batch[I_mask].unique().numel()),
    #     )

    C_rows_I = C_rows[I_mask]                                    # [Qi, T]
    annot_I = annot_mask[I_mask]                                  # [Qi, T]
    nonannot_I = non_annot_mask[I_mask]                           # [Qi, T]
    causal_I = (annot_I + nonannot_I).clamp_max(1.0)               # [Qi, T]
    loss_kind = canonical_saliency_loss_type(loss_type)

    effective_floor_eps = max(float(floor_eps), 0.0)
    batch_floor_eps = None
    effective_floor_logit = 0.0
    floor_eps_kind = "saliency"
    eps_f = float(eps)

    if loss_kind == "softmax":
        # Softmax loss from the experiment note:
        # l_qs = C_qs / tau, p_qs = softmax over all causal sources M_q,
        # L_q = - mean_{s in A_q} log p_qs. Positives and negatives compete
        # in the same denominator, so no margin or negative floor is used.
        tau = max(float(alpha), eps_f)
        logits_I = C_rows_I.float() / tau
        neg_inf = torch.finfo(logits_I.dtype).min
        logits_I = logits_I.masked_fill(causal_I == 0, neg_inf)
        log_probs = F.log_softmax(logits_I, dim=-1)
        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_nll = (-log_probs) * annot_I
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
        effective_floor_eps = 0.0
        floor_eps_kind = "disabled"
    elif loss_kind == "softmax_margin":
        # Legacy V4 loss: multi-positive NLL with a negative-logit floor.
        # For every positive source s, its denominator is exp(l_qs) plus
        # floored negative causal sources; positives do not compete with each other.
        floor_eps_f = float(effective_floor_eps)
        tau = max(float(alpha), eps_f)
        scores_I = torch.log(C_rows_I.float() + eps_f)             # [Qi, T]
        logits_I = scores_I / tau                                  # [Qi, T]

        neg_inf = torch.finfo(logits_I.dtype).min
        if floor_logit_eps is not None:
            floor = logits_I.new_tensor(float(floor_logit_eps))
            effective_floor_logit = float(floor_logit_eps)
            floor_eps_kind = "logit"
        else:
            floor = torch.log(logits_I.new_tensor(floor_eps_f + eps_f)) / tau
            effective_floor_logit = float(floor.detach().cpu())
            floor_eps_kind = "saliency"
        neg_logits = torch.maximum(logits_I, floor).masked_fill(nonannot_I == 0, neg_inf)
        neg_logsumexp = torch.logsumexp(neg_logits, dim=-1)        # [Qi]

        n_pos = annot_I.sum(dim=-1).clamp_min(eps)                 # [Qi]
        pos_log_denom = torch.logaddexp(neg_logsumexp.unsqueeze(-1), logits_I)
        pos_nll = (pos_log_denom - logits_I) * annot_I             # [Qi, T]
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
    else:
        raise ValueError(f"Unsupported saliency loss_type={loss_type!r}; expected softmax_margin or softmax.")

    C_bar_I = C_bar[I_mask]
    N_bar_I = N_bar[I_mask]

    with torch.no_grad():
        ratio = C_bar_I / (N_bar_I + eps)
        diag = SaliencyDiagnostics(
            loss=loss,
            avg_C=float(C_bar_I.mean()),
            avg_N=float(N_bar_I.mean()),
            avg_ratio=float(ratio.mean()),
            n_queries=int(C_bar_I.numel()),
            n_samples=int(row_batch[I_mask].unique().numel()),
            floor_eps=float(effective_floor_eps),
            batch_floor_eps=float(batch_floor_eps or 0.0),
            floor_eps_mode="disabled" if loss_kind == "softmax" else "fixed_logit" if floor_eps_kind == "logit" else "fixed",
            floor_logit_eps=float(effective_floor_logit),
            floor_eps_kind=floor_eps_kind,
            loss_type=loss_kind,
        )
    return diag


def build_contribution_rows(
    model,
    last_hidden_in: Tensor,   # [B, T, D]
    attn_probs: Tensor,       # [B, H, T, T]
    row_batch: Tensor,        # [Q]
    row_qry: Tensor,          # [Q]
    *,
    source_chunk_size: int = 16,
) -> Tensor:
    """Compute only selected query rows C[b, q, :] of the contribution matrix.

    This is mathematically equivalent to gathering rows from
    build_contribution_matrix(...), but avoids materializing the full
    [B, T, T, D] contribution tensor. It is the training path for the
    InfoNCE saliency loss.
    """
    decoder = _unwrap_to_decoder_stack(model)
    layer = _unwrap_module(decoder.layers[-1])
    self_attn = layer.self_attn

    B, T, D = last_hidden_in.shape
    H = attn_probs.size(1)
    device = last_hidden_in.device
    dtype = last_hidden_in.dtype
    compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32

    head_dim = getattr(self_attn, "head_dim", None) or (
        self_attn.q_proj.weight.shape[0] // H
    )
    v_out = self_attn.v_proj.weight.shape[0]
    num_kv_heads = v_out // head_dim
    assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
    n_rep = H // num_kv_heads

    gamma = layer.input_layernorm.weight.to(device=device, dtype=compute_dtype)
    gamma_x = last_hidden_in.to(compute_dtype) * gamma
    v_w = self_attn.v_proj.weight.to(device=device, dtype=compute_dtype)
    v_b = self_attn.v_proj.bias
    v_proj = gamma_x @ v_w.t()
    if v_b is not None:
        v_proj = v_proj + v_b.to(device=device, dtype=compute_dtype)
    v_states = v_proj.view(B, T, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    if n_rep > 1:
        v_states = (
            v_states.unsqueeze(2)
            .expand(B, num_kv_heads, n_rep, T, head_dim)
            .reshape(B, H, T, head_dim)
        )

    o_w = self_attn.o_proj.weight.to(device=device, dtype=compute_dtype)
    o_w_by_head = o_w.view(D, H, head_dim)

    row_batch = row_batch.to(device=device, dtype=torch.long)
    row_qry = row_qry.to(device=device, dtype=torch.long)
    Q = row_qry.numel()
    if Q == 0:
        return torch.empty((0, T), device=device, dtype=dtype)

    eps_rms = getattr(layer.input_layernorm, "variance_epsilon", 1e-6)
    query_hidden_f = last_hidden_in.float()[row_batch, row_qry, :]     # [Q, D]
    query_hidden = last_hidden_in.to(compute_dtype)[row_batch, row_qry, :]
    sigma_q = query_hidden_f.pow(2).mean(dim=-1).add(eps_rms).sqrt().clamp_min(1e-12)

    attn_f = attn_probs.to(compute_dtype)
    chunks = []
    for start in range(0, T, source_chunk_size):
        stop = min(start + source_chunk_size, T)
        attn_chunk = attn_f[row_batch, :, row_qry, start:stop]         # [Q, H, S]
        S = stop - start
        contrib = torch.zeros((Q, S, D), device=device, dtype=compute_dtype)
        # Keep the [Q, H, S, D] transformed value projection from becoming
        # the peak allocation on dense annotation batches.
        head_chunk_size = 1
        for h_start in range(0, H, head_chunk_size):
            h_stop = min(h_start + head_chunk_size, H)
            attn_part = attn_chunk[:, h_start:h_stop, :]               # [Q, h, S]
            v_part = v_states[row_batch, h_start:h_stop, start:stop, :] # [Q, h, S, head_dim]
            o_part = o_w_by_head[:, h_start:h_stop, :]                 # [D, h, head_dim]
            transformed_part = torch.einsum("qhsd,ohd->qhso", v_part, o_part)
            contrib = contrib + torch.einsum("qhs,qhso->qso", attn_part, transformed_part)

        diag_mask = (row_qry >= start) & (row_qry < stop)
        if bool(diag_mask.any()):
            local = row_qry[diag_mask] - start
            contrib[diag_mask, local, :] = contrib[diag_mask, local, :] + query_hidden[diag_mask]

        contrib = contrib / sigma_q.to(compute_dtype).view(Q, 1, 1)
        chunks.append(contrib.float().norm(dim=-1, p=2).to(dtype))

    return torch.cat(chunks, dim=1)


def _annotation_rows_from_pairs(
    annot_pairs: list[Tensor] | Tensor,
    *,
    B: int,
    T: int,
    device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    flat = flatten_annot_pairs(annot_pairs, device=device)
    if flat.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty, empty

    batch_ids = flat[:, 0]
    p0 = flat[:, 1]
    p1 = flat[:, 2]
    src_all = torch.minimum(p0, p1)
    qry_all = torch.maximum(p0, p1)
    keep = (
        (src_all < T) &
        (qry_all < T) & (qry_all > src_all) &
        (batch_ids >= 0) & (batch_ids < B)
    )
    batch_ids = batch_ids[keep]
    src_all = src_all[keep]
    qry_all = qry_all[keep]
    if batch_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty, empty

    keys = batch_ids * T + qry_all
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    row_batch = unique_keys // T
    row_qry = unique_keys % T
    return row_batch, row_qry, src_all, inv




def estimate_causal_saliency_quantile_from_rows(
    C_rows: Tensor,
    row_qry: Tensor,
    *,
    quantile: float = 0.75,
    min_eps: float = 1e-8,
) -> float | None:
    """Estimate Q_quantile over causal source saliency for selected query rows.

    The returned value is detached and intended as a training hyperparameter
    estimate, not as a differentiable part of the objective.
    """
    if C_rows.numel() == 0 or row_qry.numel() == 0:
        return None
    with torch.no_grad():
        q = float(max(0.0, min(1.0, quantile)))
        values = []
        row_qry_list = row_qry.detach().to(device="cpu", dtype=torch.long).tolist()
        for row_idx, qry in enumerate(row_qry_list):
            if qry <= 0:
                continue
            values.append(C_rows[row_idx, :qry].detach().float().reshape(-1))
        if not values:
            return None
        flat = torch.cat(values)
        if flat.numel() == 0:
            return None
        est = torch.quantile(flat, q)
        return max(float(est.item()), float(min_eps))


def resolve_saliency_floor_eps(
    *,
    mode: str,
    fixed_floor_eps: float,
    batch_floor_eps: float | None,
    prev_floor_eps: float | None,
    ema_beta: float,
    min_eps: float,
    step: int = 0,
    warmup_steps: int = 0,
) -> float:
    """Resolve the saliency-space negative floor used by the current step.

    For ema_quantile, the current batch first produces hat_eps_t, then the
    returned eps_t is used in this same step's saliency loss:

        eps_t = 0,                                  t < T_warmup
        eps_t = hat_eps_t,                        t = T_warmup
        eps_t = beta * eps_{t-1} + (1-beta)*hat_eps_t, t > T_warmup

    ``batch_floor_eps`` is detached before this helper is called, so eps_t is a
    dynamic hyperparameter rather than a differentiable target.
    """
    mode = (mode or "fixed").strip().lower()
    fixed = max(float(fixed_floor_eps), 0.0)
    floor_min = float(min_eps)
    step_i = int(step or 0)
    warmup_i = max(int(warmup_steps or 0), 0)

    if mode == "fixed":
        return fixed

    if mode in {"batch_quantile", "ema_quantile"} and warmup_i > 0 and 0 < step_i < warmup_i:
        return 0.0

    if batch_floor_eps is None:
        if prev_floor_eps is not None:
            return max(float(prev_floor_eps), floor_min)
        return max(fixed, floor_min) if fixed > 0 else 0.0

    batch = max(float(batch_floor_eps), floor_min)
    if mode == "batch_quantile":
        return batch
    if mode == "ema_quantile":
        beta = min(max(float(ema_beta), 0.0), 0.9999)
        # At the first active floor step, initialize eps_t directly from the
        # current batch quantile. This matches eps_Twarmup = \hat{eps}_Twarmup.
        if warmup_i > 0 and step_i == warmup_i:
            return batch
        if prev_floor_eps is not None and float(prev_floor_eps) > 0:
            base = float(prev_floor_eps)
            return max(beta * base + (1.0 - beta) * batch, floor_min)
        if fixed > 0:
            return max(beta * fixed + (1.0 - beta) * batch, floor_min)
        return batch
    raise ValueError(f"Unsupported floor_eps_mode={mode!r}")

def compute_saliency_loss_from_rows(
    C_rows: Tensor,          # [Q, T]
    row_batch: Tensor,       # [Q]
    row_qry: Tensor,         # [Q]
    src_all: Tensor,         # [E]
    inv: Tensor,             # [E], edge -> row index
    *,
    alpha: float,
    eps: float,
    floor_eps: float = 0.0,
    floor_eps_mode: str = "fixed",
    floor_eps_quantile: float = 0.75,
    floor_eps_ema_beta: float = 0.95,
    prev_floor_eps: float | None = None,
    floor_eps_min: float = 1e-8,
    floor_eps_step: int = 0,
    floor_eps_warmup_steps: int = 0,
    floor_logit_eps: float | None = None,
    loss_type: str = "softmax_margin",
    margin_plus: float = 2.08,    # log(8) — desired log-saliency lower bound for annotated edges
    margin_minus: float = 0.41,   # log(1.5) — log-saliency upper bound for non-annot edges (no penalty below)
    margin_gamma: float = 2.0,    # softplus sharpness
    neg_weight: float = 0.5,      # multiplier on the negative-side mean penalty
    neg_hard_only: bool = False,  # if True, average only over negatives with r > margin_minus
    neg_sample_k: int = 0,        # if >0, randomly subsample this many negatives per query for softmax/softmax_margin (MoCo-style)
) -> SaliencyDiagnostics:
    """Saliency objective with C already row-gathered."""
    Q, T = C_rows.shape
    device = C_rows.device
    dtype = C_rows.dtype
    if Q == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0,
            avg_N=0.0,
            avg_ratio=0.0,
            n_queries=0,
            n_samples=0,
        )

    A_adj = torch.zeros(Q, T, device=device, dtype=dtype)
    A_adj.index_put_(
        (inv.to(device=device, dtype=torch.long), src_all.to(device=device, dtype=torch.long)),
        torch.ones_like(src_all, device=device, dtype=dtype),
        accumulate=True,
    )
    A_adj_bin = (A_adj > 0).to(dtype)

    src_idx = torch.arange(T, device=device).unsqueeze(0)
    qry_col = row_qry.to(device=device, dtype=torch.long).unsqueeze(1)
    M_causal = ((src_idx <= qry_col) & (src_idx != qry_col)).to(dtype)

    annot_mask = M_causal * A_adj_bin
    non_annot_mask = M_causal * (1.0 - A_adj_bin)

    den_C = annot_mask.sum(dim=-1).clamp_min(eps)
    den_N = non_annot_mask.sum(dim=-1).clamp_min(eps)
    C_bar = (annot_mask * C_rows).sum(dim=-1) / den_C
    N_bar = (non_annot_mask * C_rows).sum(dim=-1) / den_N

    I_mask = (annot_mask.sum(dim=-1) > 0) & (non_annot_mask.sum(dim=-1) > 0)
    if not bool(I_mask.any()):
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0,
            avg_N=0.0,
            avg_ratio=0.0,
            n_queries=0,
            n_samples=int(row_batch.unique().numel()),
        )

    C_rows_I = C_rows[I_mask]
    annot_I = annot_mask[I_mask]
    nonannot_I = non_annot_mask[I_mask]

    loss_kind = canonical_saliency_loss_type(loss_type)

    # Optional MoCo-style negative subsampling for softmax / softmax_margin.
    # For each query, keep at most `neg_sample_k` randomly-chosen negatives
    # active inside the saliency loss. Positives and the non-causal mask
    # are untouched. Sampling weights are uniform over current negatives so
    # every above-floor negative has an equal expected gradient share, breaking
    # the InfoNCE winner-takes-all bias. Across steps the iid resampling covers
    # the full N_q (MoCo / SimCLR style).
    if (
        int(neg_sample_k) > 0
        and loss_kind in ("softmax", "softmax_margin", "ranknet", "contrastive")
        and nonannot_I.size(0) > 0
    ):
        k = int(neg_sample_k)
        with torch.no_grad():
            neg_counts = nonannot_I.sum(dim=-1)
            # rows where we actually need to subsample (>k negatives)
            sample_rows = (neg_counts > k).nonzero(as_tuple=True)[0]
            if sample_rows.numel() > 0:
                # build per-row uniform weights on the negative positions
                weights = nonannot_I[sample_rows].to(torch.float32) + 1e-12  # [R, T]
                # zero-weight non-causal/positive positions stay near-zero; we want exactly 0
                weights = weights * nonannot_I[sample_rows].to(torch.float32)
                idx = torch.multinomial(weights, num_samples=k, replacement=False)  # [R, k]
                new_mask = torch.zeros_like(nonannot_I[sample_rows])
                new_mask.scatter_(1, idx, 1.0)
                nonannot_I = nonannot_I.clone()
                nonannot_I[sample_rows] = new_mask

    floor_mode = (floor_eps_mode or "fixed").strip().lower()
    batch_floor_eps = None
    effective_floor_eps = 0.0
    effective_floor_logit = 0.0
    floor_eps_kind = "saliency"
    eps_f = float(eps)

    if loss_kind == "softmax":
        causal_I = (annot_I + nonannot_I).clamp_max(1.0)
        tau = max(float(alpha), eps_f)
        logits_I = C_rows_I.float() / tau
        neg_inf = torch.finfo(logits_I.dtype).min
        logits_I = logits_I.masked_fill(causal_I == 0, neg_inf)
        log_probs = F.log_softmax(logits_I, dim=-1)
        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_nll = (-log_probs) * annot_I
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
        diag_floor_mode = "disabled"
        floor_eps_kind = "disabled"
    elif loss_kind == "softmax_margin":
        if floor_logit_eps is None:
            batch_floor_eps = estimate_causal_saliency_quantile_from_rows(
                C_rows_I,
                row_qry[I_mask],
                quantile=floor_eps_quantile,
                min_eps=floor_eps_min,
            ) if floor_mode != "fixed" else None
            effective_floor_eps = resolve_saliency_floor_eps(
                mode=floor_mode,
                fixed_floor_eps=floor_eps,
                batch_floor_eps=batch_floor_eps,
                prev_floor_eps=prev_floor_eps,
                ema_beta=floor_eps_ema_beta,
                min_eps=floor_eps_min,
                step=floor_eps_step,
                warmup_steps=floor_eps_warmup_steps,
            )
        else:
            batch_floor_eps = None
            effective_floor_eps = 0.0

        floor_eps_f = float(effective_floor_eps)
        tau = max(float(alpha), eps_f)
        scores_I = torch.log(C_rows_I.float() + eps_f)
        logits_I = scores_I / tau
        neg_inf = torch.finfo(logits_I.dtype).min
        if floor_logit_eps is not None:
            floor = logits_I.new_tensor(float(floor_logit_eps))
            effective_floor_logit = float(floor_logit_eps)
            floor_eps_kind = "logit"
            diag_floor_mode = "fixed_logit"
        else:
            floor = torch.log(logits_I.new_tensor(floor_eps_f + eps_f)) / tau
            effective_floor_logit = float(floor.detach().cpu())
            floor_eps_kind = "saliency"
        neg_logits = torch.maximum(logits_I, floor).masked_fill(nonannot_I == 0, neg_inf)
        neg_logsumexp = torch.logsumexp(neg_logits, dim=-1)

        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_log_denom = torch.logaddexp(neg_logsumexp.unsqueeze(-1), logits_I)
        pos_nll = (pos_log_denom - logits_I) * annot_I
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
        if floor_logit_eps is None:
            diag_floor_mode = floor_mode
    elif loss_kind == "margin_bce":
        # Per-edge decoupled soft-margin BCE on log-saliency.
        # r_{q,s} = log(C+eps); positives pushed above margin_plus, negatives
        # pushed below margin_minus. Both penalties use softplus(gamma * delta)/gamma
        # so per-edge gradient is sigmoid-bounded in (0, 1) and saturates smoothly
        # once the edge is on the right side of its margin. No shared denominator,
        # so positives don't compete, and margins are absolute constants (set from
        # the base model once) so they cannot drift with training.
        r = torch.log(C_rows_I.float() + eps_f)              # [Qi, T]
        gamma = max(float(margin_gamma), eps_f)
        # Positive side:
        pos_arg = gamma * (float(margin_plus) - r)
        pos_term = F.softplus(pos_arg) / gamma               # [Qi, T]
        pos_term = pos_term * annot_I                        # mask to positives
        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_loss_q = pos_term.sum(dim=-1) / n_pos            # [Qi]
        # Negative side:
        neg_arg = gamma * (r - float(margin_minus))
        neg_term = F.softplus(neg_arg) / gamma               # [Qi, T]
        if neg_hard_only:
            active = nonannot_I * (r > float(margin_minus)).to(nonannot_I.dtype)
            neg_term = neg_term * active
            n_neg = active.sum(dim=-1).clamp_min(eps)
        else:
            neg_term = neg_term * nonannot_I
            n_neg = nonannot_I.sum(dim=-1).clamp_min(eps)
        neg_loss_q = neg_term.sum(dim=-1) / n_neg            # [Qi]

        loss = (pos_loss_q + float(neg_weight) * neg_loss_q).mean().to(dtype)
        diag_floor_mode = "margin_bce"
        floor_eps_kind = "absolute_margin"
        # surface margins in the existing diag fields for log readability
        effective_floor_eps = float(math.exp(float(margin_minus)) - eps_f)
        effective_floor_logit = float(margin_minus)
    elif loss_kind == "ranknet":
        # Pairwise RankNet on log-saliency. For each query q and each
        # (positive s, negative s') pair:
        #   loss_{q,s,s'} = softplus( (log C_{q,s'} - log C_{q,s}) / tau )
        # Each pair contributes an independent sigmoid-bounded gradient — no
        # shared softmax denominator, so no winner-takes-all on either side.
        # Softplus saturates as the negative falls below the positive, which
        # acts as a natural "robust floor" without any explicit threshold.
        tau = max(float(alpha), eps_f)
        scores = torch.log(C_rows_I.float() + eps_f)         # [Qi, T]
        Qi = scores.shape[0]
        per_q_losses = []
        for q in range(Qi):
            p_idx = annot_I[q].bool().nonzero(as_tuple=True)[0]
            n_idx = nonannot_I[q].bool().nonzero(as_tuple=True)[0]
            if p_idx.numel() == 0 or n_idx.numel() == 0:
                continue
            pos = scores[q, p_idx]                            # [|A|]
            neg = scores[q, n_idx]                            # [|N|]
            diff = (neg.unsqueeze(0) - pos.unsqueeze(1)) / tau  # [|A|, |N|]
            per_q_losses.append(F.softplus(diff).mean())
        if per_q_losses:
            loss = torch.stack(per_q_losses).mean().to(dtype)
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
        diag_floor_mode = "ranknet"
        floor_eps_kind = "pairwise"
        effective_floor_eps = 0.0
        effective_floor_logit = 0.0
    elif loss_kind == "contrastive":
        # Classic triplet-style pair hinge on log-saliency. For each query q and
        # each (positive s, negative s') pair:
        #   loss_{q,s,s'} = max(0, margin + (log C_{q,s'} - log C_{q,s}) / tau)
        # Margin is set in log-saliency units via `margin_plus`. Once a pair is
        # on the right side of the margin, its gradient is exactly zero — so
        # already-correct negatives stop being pushed down and already-strong
        # positives stop being pulled up. This gives a sparse, robust signal
        # and avoids both InfoNCE's winner-takes-all and RankNet's softplus
        # tail still leaking gradient to easy pairs.
        tau = max(float(alpha), eps_f)
        margin = float(margin_plus)
        scores = torch.log(C_rows_I.float() + eps_f)         # [Qi, T]
        Qi_ = scores.shape[0]
        per_q = []
        for q in range(Qi_):
            p_idx = annot_I[q].bool().nonzero(as_tuple=True)[0]
            n_idx = nonannot_I[q].bool().nonzero(as_tuple=True)[0]
            if p_idx.numel() == 0 or n_idx.numel() == 0:
                continue
            pos = scores[q, p_idx]
            neg = scores[q, n_idx]
            diff = (neg.unsqueeze(0) - pos.unsqueeze(1)) / tau  # [|A|, |N|]
            per_q.append(F.relu(margin + diff).mean())
        if per_q:
            loss = torch.stack(per_q).mean().to(dtype)
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
        diag_floor_mode = "contrastive"
        floor_eps_kind = "pair_hinge"
        effective_floor_eps = 0.0
        effective_floor_logit = float(margin)
    else:
        raise ValueError(f"Unsupported saliency loss_type={loss_type!r}; expected softmax_margin, softmax, margin_bce, ranknet, or contrastive.")

    C_bar_I = C_bar[I_mask]
    N_bar_I = N_bar[I_mask]
    with torch.no_grad():
        ratio = C_bar_I / (N_bar_I + eps)
        diag = SaliencyDiagnostics(
            loss=loss,
            avg_C=float(C_bar_I.mean()),
            avg_N=float(N_bar_I.mean()),
            avg_ratio=float(ratio.mean()),
            n_queries=int(C_bar_I.numel()),
            n_samples=int(row_batch[I_mask].unique().numel()),
            floor_eps=float(effective_floor_eps),
            batch_floor_eps=float(batch_floor_eps or 0.0),
            floor_eps_mode=diag_floor_mode,
            floor_eps_step=int(floor_eps_step or 0),
            floor_eps_warmup_steps=int(floor_eps_warmup_steps or 0),
            floor_logit_eps=float(effective_floor_logit),
            floor_eps_kind=floor_eps_kind,
            loss_type=loss_kind,
        )
    return diag


def saliency_loss_from_outputs(
        model,
        outputs,  # HF output with attentions + hidden_states
        annot_pairs: list[Tensor] | Tensor,  # flat [B, 3] or list of [Number of pairs, 2]
        *,
        alpha: float = 1.5,
        eps: float = 1e-8,
        floor_eps: float = 0.0,
        floor_eps_mode: str = "fixed",
        floor_eps_quantile: float = 0.75,
        floor_eps_ema_beta: float = 0.95,
        prev_floor_eps: float | None = None,
        floor_eps_min: float = 1e-8,
        floor_eps_step: int = 0,
        floor_eps_warmup_steps: int = 0,
        floor_logit_eps: float | None = None,
        loss_type: str = "softmax_margin",
        margin_plus: float = 2.08,
        margin_minus: float = 0.41,
        margin_gamma: float = 2.0,
        neg_weight: float = 0.5,
        neg_hard_only: bool = False,
        neg_sample_k: int = 0,
        source_chunk_size: int = 16,
) -> SaliencyDiagnostics:
    """
    Takes the model's forward output and produces the saliency diagnostics in one call.

    `outputs` must have `.attentions` (non-None) and `.hidden_states`.
    `annot_pairs` may be either a flat [B, 3] tensor (batch_idx, pos_a, pos_b)
    or a list of [Number of pairs, 2] tensors.
    """
    attn_last = outputs.attentions[-1]
    assert attn_last is not None, "outputs.attentions[-1] is None — switch to eager attention."

    # hidden_states tuple: [embedding_out, layer_1_out, ..., layer_L_out]
    # input to the last decoder layer = hidden_states[-2]
    last_hidden_in = outputs.hidden_states[-2]  # [B, T, D]
    # The saliency objective only needs the last-layer attention and the input
    # to the final decoder block. Keeping every layer hidden/attention alive
    # can add several GiB of peak memory before contribution rows are built.
    try:
        outputs.attentions = (attn_last,)
        outputs.hidden_states = (last_hidden_in,)
    except Exception:
        pass

    B, T, _ = last_hidden_in.shape
    row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs(
        annot_pairs,
        B=B,
        T=T,
        device=last_hidden_in.device,
    )
    if row_qry.numel() == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=last_hidden_in.device, dtype=last_hidden_in.dtype),
            avg_C=0.0,
            avg_N=0.0,
            avg_ratio=0.0,
            n_queries=0,
            n_samples=0,
        )

    C_rows = build_contribution_rows(
        model,
        last_hidden_in,
        attn_last,
        row_batch,
        row_qry,
        source_chunk_size=max(1, int(source_chunk_size)),
    )
    return compute_saliency_loss_from_rows(
        C_rows,
        row_batch,
        row_qry,
        src_all,
        inv,
        alpha=alpha,
        eps=eps,
        floor_eps=floor_eps,
        floor_eps_mode=floor_eps_mode,
        floor_eps_quantile=floor_eps_quantile,
        floor_eps_ema_beta=floor_eps_ema_beta,
        prev_floor_eps=prev_floor_eps,
        floor_eps_min=floor_eps_min,
        floor_eps_step=floor_eps_step,
        floor_eps_warmup_steps=floor_eps_warmup_steps,
        floor_logit_eps=floor_logit_eps,
        loss_type=loss_type,
        margin_plus=margin_plus,
        margin_minus=margin_minus,
        margin_gamma=margin_gamma,
        neg_weight=neg_weight,
        neg_hard_only=neg_hard_only,
        neg_sample_k=neg_sample_k,
    )
