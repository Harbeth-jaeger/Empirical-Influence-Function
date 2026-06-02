from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import tqdm
from openai import OpenAI

from scripts.benchmark.governance_common import _extract_code_block_or_raw, _safe_float


def _clear_pairwise_similarity(outputs: list[str]) -> float:
    outputs = [out.strip() for out in outputs if out and out.strip()]
    if len(outputs) < 2:
        return 0.0
    scores = [
        SequenceMatcher(None, y_i, y_j).ratio()
        for i, y_i in enumerate(outputs)
        for j, y_j in enumerate(outputs)
        if i != j
    ]
    return float(np.mean(scores)) if scores else 0.0


def _clear_target_consistency(target: str, sampled_outputs: list[str]) -> float:
    """
    CLEAR-style observed consistency: compare the dataset target response with
    multiple sampled responses for the same prompt.  Pairwise agreement among
    sampled responses alone can measure model self-consistency, but it does not
    tell whether the original training target is aligned with those outputs.
    """
    target = target.strip()
    outputs = [out.strip() for out in sampled_outputs if out and out.strip()]
    if not target or not outputs:
        return 0.0
    scores = [SequenceMatcher(None, target, out).ratio() for out in outputs]
    return float(np.mean(scores)) if scores else 0.0


def _clear_generate(
    client: OpenAI,
    model_name: str,
    prompt: str,
    language: str = "",
    n: int = 1,
    temperature: float = 0.8,
    max_tokens: int = 512,
) -> list[str]:
    system = "You are a precise code completion assistant. Return code only, no explanation."
    if language:
        system = f"You are a precise {language} code completion assistant. Return code only, no explanation."

    outputs: list[str] = []
    for _ in range(max(1, n)):
        try:
            raw = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            ).choices[0].message.content or ""
            outputs.append(_extract_code_block_or_raw(raw))
        except Exception:
            outputs.append("")
    return outputs


def _clear_self_reflection(
    client: OpenAI,
    model_name: str,
    prompt: str,
    response: str,
) -> float:
    judge_prompt = (
        "You are judging a code-completion training sample. "
        "Estimate whether the target output is correct, minimal, and directly answers the input. "
        "Return only one number in [0, 1].\n\n"
        f"[Input]\n{prompt}\n\n[Target]\n{response}"
    )
    try:
        raw = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=16,
        ).choices[0].message.content or ""
        return _safe_float(raw, default=0.5)
    except Exception:
        return 0.5


def _clear_bsdetector_confidence(
    client: OpenAI,
    model_name: str,
    prompt: str,
    response: str,
    language: str = "",
    alpha: float = 0.5,
    consistency_samples: int = 3,
    consistency_temperature: float = 0.8,
    max_tokens: int = 512,
) -> dict[str, float]:
    sampled = _clear_generate(
        client=client,
        model_name=model_name,
        prompt=prompt,
        language=language,
        n=consistency_samples,
        temperature=consistency_temperature,
        max_tokens=max_tokens,
    )
    observed = _clear_target_consistency(response, sampled)
    certainty = _clear_self_reflection(client, model_name, prompt, response)
    confidence = alpha * observed + (1.0 - alpha) * certainty
    return {
        "confidence": max(0.0, min(1.0, float(confidence))),
        "observed_consistency": float(observed),
        "self_reflection_certainty": float(certainty),
    }


def _clear_candidate_better_score(
    client: OpenAI,
    model_name: str,
    prompt: str,
    original: str,
    candidate: str,
) -> float:
    judge_prompt = (
        "You are a strict LLM-as-judge for code-completion training data. "
        "Decide whether Candidate B is significantly better than Original A for the input. "
        "Consider correctness, minimality, syntax, and whether it returns only the missing code span. "
        "Return only one number in [0, 1], where 1 means B is clearly better.\n\n"
        f"[Input]\n{prompt}\n\n[Original A]\n{original}\n\n[Candidate B]\n{candidate}"
    )
    try:
        raw = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=16,
        ).choices[0].message.content or ""
        return _safe_float(raw, default=0.0)
    except Exception:
        return 0.0


def _clear_replace_response(sample: dict[str, Any], response: str) -> None:
    sample["response"] = response
    sample["fim_completion"] = response
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            msg["content"] = response


def clear_curation_operator(
    sample: dict[str, Any],
    client: OpenAI,
    model_name: str = "deepseek-v4",
    stage: Literal["filter", "correct", "filter_correct"] = "filter",
    gamma: float = 0.5,
    eta: float = 0.8,
    alpha: float = 0.5,
    consistency_samples: int = 3,
    consistency_temperature: float = 0.8,
    candidate_model_name: str | None = None,
    candidate_temperature: float = 0.2,
    max_tokens: int = 512,
) -> dict[str, Any] | None:
    """
    CLEAR-style sample filtering/correction using BSDetector confidence.
    """
    if stage not in {"filter", "correct", "filter_correct"}:
        raise ValueError("stage must be 'filter', 'correct', or 'filter_correct'.")

    local = copy.deepcopy(sample)
    x = str(local.get("prompt", ""))
    y = str(local.get("response", ""))
    language = str(local.get("language", ""))

    if stage == "filter":
        scores = _clear_bsdetector_confidence(
            client, model_name, x, y, language, alpha,
            consistency_samples, consistency_temperature, max_tokens,
        )
        local["clear_metadata"] = {"stage": "filter", "original": scores}
        return local if scores["confidence"] > gamma else None

    original_scores = _clear_bsdetector_confidence(
        client, model_name, x, y, language, alpha,
        consistency_samples, consistency_temperature, max_tokens,
    )
    y_prime = str(local.get("candidate_y", "")).strip()
    if not y_prime:
        generated = _clear_generate(
            client=client,
            model_name=candidate_model_name or model_name,
            prompt=x,
            language=language,
            n=1,
            temperature=candidate_temperature,
            max_tokens=max_tokens,
        )
        y_prime = generated[0].strip() if generated else ""

    candidate_scores = {"confidence": 0.0, "observed_consistency": 0.0, "self_reflection_certainty": 0.0}
    better_score = 0.0
    if y_prime:
        candidate_scores = _clear_bsdetector_confidence(
            client, model_name, x, y_prime, language, alpha,
            consistency_samples, consistency_temperature, max_tokens,
        )
        better_score = _clear_candidate_better_score(client, model_name, x, y, y_prime)

    local["clear_metadata"] = {
        "stage": stage,
        "original": original_scores,
        "candidate": candidate_scores,
        "candidate_better_score": float(better_score),
        "replaced": False,
    }

    if y_prime and better_score > eta and candidate_scores["confidence"] >= original_scores["confidence"]:
        _clear_replace_response(local, y_prime)
        local["clear_metadata"]["replaced"] = True
        return local

    if stage == "correct":
        return local
    return local if original_scores["confidence"] > gamma else None


def clear_curation_dataset(
    samples: list[dict[str, Any]],
    client: OpenAI,
    model_name: str = "deepseek-v4",
    stage: Literal["filter", "correct", "filter_correct"] = "filter",
    gamma: float = 0.5,
    eta: float = 0.8,
    alpha: float = 0.5,
    consistency_samples: int = 3,
    consistency_temperature: float = 0.8,
    candidate_model_name: str | None = None,
    candidate_temperature: float = 0.2,
    max_tokens: int = 512,
    num_workers: int = 1,
) -> list[dict[str, Any]]:
    """Apply CLEAR-style curation to a list of samples."""
    def curate_one(sample: dict[str, Any]) -> dict[str, Any] | None:
        return clear_curation_operator(
            sample=sample,
            client=client,
            model_name=model_name,
            stage=stage,
            gamma=gamma,
            eta=eta,
            alpha=alpha,
            consistency_samples=consistency_samples,
            consistency_temperature=consistency_temperature,
            candidate_model_name=candidate_model_name,
            candidate_temperature=candidate_temperature,
            max_tokens=max_tokens,
        )

    out: list[dict[str, Any]] = []
    if num_workers <= 1:
        iterator = map(curate_one, samples)
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        iterator = executor.map(curate_one, samples)

    try:
        for curated in tqdm.tqdm(iterator, total=len(samples), desc="  clear.curation", mininterval=5):
            if curated is not None:
                out.append(curated)
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True)
    return out

