from __future__ import annotations

import torch

from src.baseline.clear import apply_clear_scores, clear_confidence
from src.baseline.common import IGNORE_INDEX
from src.baseline.ibft import VariationalBottleneck, compute_ibft_loss
from src.baseline.llm_code_cleaning import apply_llm_cleaning_rewrites
from src.baseline.token_cleaning import apply_token_cleaning_from_scores, token_quality_scores
from src.baseline.xtf import apply_xtf_from_scores


def _sample(uid: str = "s1"):
    return {"uid": uid, "input_ids": [1, 2, 3, 4, 5], "labels": [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]}


def test_token_cleaning_masks_low_score_labels():
    rows, report = apply_token_cleaning_from_scores(
        [_sample()],
        {"s1": {"uid": "s1", "scores": [0, 0, 0.1, 0.9, 0.8]}},
        keep_ratio=2 / 3,
    )
    assert rows[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]
    assert report["kept_tokens"] == 2


def test_token_quality_score_is_base_minus_reference():
    assert token_quality_scores([3.0, 1.0], [1.0, 2.0]) == [2.0, -1.0]


def test_xtf_masks_high_pcp_tokens():
    rows, report = apply_xtf_from_scores(
        [_sample()],
        {"s1": {"uid": "s1", "pcp_probs": [0, 0, 0.99, 0.2, 0.1]}},
        pcp_threshold=0.95,
    )
    assert rows[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]
    assert report["masked_tokens"] == 1


def test_clear_filters_and_replaces():
    rows, report = apply_clear_scores(
        [_sample("keep"), _sample("drop"), _sample("replace")],
        {
            "keep": {"uid": "keep", "confidence": 0.9},
            "drop": {"uid": "drop", "confidence": 0.1},
            "replace": {
                "uid": "replace",
                "confidence": 0.2,
                "candidate_response": "better()",
                "candidate_confidence": 0.8,
                "candidate_better_score": 0.95,
            },
        },
        gamma=0.5,
        eta=0.8,
    )
    assert len(rows) == 2
    assert report["filtered_samples"] == 1
    assert report["replaced_samples"] == 1


def test_clear_confidence_is_clamped():
    assert clear_confidence(2.0, -1.0) == 0.5


def test_llm_cleaning_rewrite_updates_response_fields():
    sample = {"uid": "s1", "response": "old", "messages": [{"role": "assistant", "content": "old"}]}
    rows, report = apply_llm_cleaning_rewrites([sample], {"s1": {"uid": "s1", "cleaned_response": "```go\nnew\n```"}})
    assert rows[0]["response"] == "new"
    assert rows[0]["messages"][0]["content"] == "new"
    assert report["replaced_samples"] == 1


def test_ibft_loss_shapes():
    hidden = (torch.randn(1, 5, 8),)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
    lm_head = torch.nn.Linear(8, 16)
    bottleneck = VariationalBottleneck(8, z_dim=4)
    out = compute_ibft_loss(hidden_states=hidden, labels=labels, lm_head=lm_head, bottleneck=bottleneck)
    assert out.num_tokens == 3
    assert out.loss.ndim == 0
