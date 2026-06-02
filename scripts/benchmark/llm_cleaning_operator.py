from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tqdm
from openai import OpenAI

from scripts.benchmark.governance_common import _extract_code_block_or_raw


def llm_code_cleaning_operator(
    problem_stmt: str,
    raw_code: str,
    test_cases: list[dict[str, str]],
    api_key: str,
    model_name: str = "deepseek-v4",
    max_retries_per_stage: int = 5,
) -> str:
    """
    LLM-assisted code cleaning with 3-stage prompting + oracle equivalence checks.
    """
    client = OpenAI(api_key=api_key)

    prompts = {
        "rename": "Rename the variables in the program to be descriptive, meaningful, and consistent.",
        "modularize": "Refactor the above program making it more modular with smaller and meaningful helper functions.",
        "planning": "Generate a natural language description for the following functions in the program within four lines each at the top of the code.",
    }

    current_code = raw_code

    def verify_equivalence(code: str) -> bool:
        for test in test_cases:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-c", code],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate(input=test.get("input", ""), timeout=5)
                if process.returncode != 0:
                    return False
                if stderr.strip():
                    return False
                if stdout.strip() != str(test.get("output", "")).strip():
                    return False
            except Exception:
                return False
        return True

    for stage in ["rename", "modularize", "planning"]:
        instruction = prompts[stage]
        messages = [
            {
                "role": "system",
                "content": "You are an expert software engineer. Return only Python code.",
            },
            {
                "role": "user",
                "content": f"Problem: {problem_stmt}\n\nCode:\n{current_code}\n\nTask: {instruction}",
            },
        ]

        for _ in range(max_retries_per_stage):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                )
                candidate_raw = response.choices[0].message.content or ""
                cleaned_candidate = _extract_code_block_or_raw(candidate_raw)
            except Exception:
                continue

            if cleaned_candidate and verify_equivalence(cleaned_candidate):
                current_code = cleaned_candidate
                break

    return current_code


def llm_code_cleaning_dataset(
    samples: list[dict[str, Any]],
    api_key: str,
    model_name: str = "deepseek-v4",
    prompt_key: str = "prompt",
    code_key: str = "response",
    test_cases_key: str = "test_cases",
    output_key: str = "response",
) -> list[dict[str, Any]]:
    """Apply LLM code cleaning to a list of code samples."""
    out: list[dict[str, Any]] = []
    for sample in tqdm.tqdm(samples, desc="  llm_code_cleaning.dataset", mininterval=5):
        local = copy.deepcopy(sample)
        cleaned = llm_code_cleaning_operator(
            problem_stmt=str(local.get(prompt_key, "")),
            raw_code=str(local.get(code_key, "")),
            test_cases=list(local.get(test_cases_key, [])),
            api_key=api_key,
            model_name=model_name,
        )
        local[output_key] = cleaned
        out.append(local)
    return out



