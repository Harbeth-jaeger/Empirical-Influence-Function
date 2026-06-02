from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RowMetric:
    source_dataset: str
    language: str
    judged: bool
    pass1: float
    pass10: float


def aggregate_pass(metrics: list[RowMetric]) -> dict[str, Any]:
    judged = [m for m in metrics if m.judged]

    by_dataset_language: dict[tuple[str, str], list[RowMetric]] = {}
    for m in metrics:
        by_dataset_language.setdefault((m.source_dataset, m.language), []).append(m)

    detail: list[dict[str, Any]] = []
    for (ds, lang), vals in sorted(by_dataset_language.items()):
        judged_vals = [v for v in vals if v.judged]
        n_total = len(vals)
        n_judged = len(judged_vals)
        pass1 = (sum(v.pass1 for v in judged_vals) / n_judged) if n_judged > 0 else 0.0
        pass10 = (sum(v.pass10 for v in judged_vals) / n_judged) if n_judged > 0 else 0.0
        detail.append(
            {
                "source_dataset": ds,
                "language": lang,
                "n_total": n_total,
                "n_judged": n_judged,
                "n_unsupported": n_total - n_judged,
                "pass@1": pass1,
                "pass@10": pass10,
            }
        )

    n_total_all = len(metrics)
    n_judged_all = len(judged)
    overall = {
        "n_total": n_total_all,
        "n_judged": n_judged_all,
        "n_unsupported": n_total_all - n_judged_all,
        "pass@1": (sum(v.pass1 for v in judged) / n_judged_all) if n_judged_all > 0 else 0.0,
        "pass@10": (sum(v.pass10 for v in judged) / n_judged_all) if n_judged_all > 0 else 0.0,
    }
    return {"overall": overall, "detail": detail}


def _fmt_score(value: str | float | None) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def write_benchmark_tables_markdown(path: str, baseline_name: str, detail_rows: list[dict[str, Any]]) -> None:
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for row in detail_rows:
        if int(row.get("N", 0)) <= 0:
            continue
        lookup[(str(row["Dataset"]).lower(), str(row["Language"]).lower())] = (
            str(row["Pass@1"]),
            str(row["Pass@10"]),
        )

    baseline_aliases = {
        "Ours Graphsignal": {"ours graphsignal", "ours-graphsignal", "ours_graphsignal"},
        "TokenCleaning": {"tokencleaning", "token-cleaning", "token_cleaning"},
        "XTF": {"xtf"},
        "LLM-CleanCode": {"llm-cleancode", "llm_cleancode", "llm-cleancode"},
        "CLEAR": {"clear"},
    }

    def is_current(display_name: str) -> bool:
        normalized = baseline_name.strip().lower().replace(" ", "_")
        aliases = {a.lower().replace(" ", "_") for a in baseline_aliases.get(display_name, set())}
        aliases.add(display_name.lower().replace(" ", "_"))
        return normalized in aliases

    def scores(dataset: str, language: str, display_name: str) -> tuple[str, str]:
        if not is_current(display_name):
            return "", ""
        p1, p10 = lookup.get((dataset, language.lower()), ("", ""))
        return _fmt_score(p1), _fmt_score(p10)

    lines: list[str] = []
    lines.extend(
        [
            "# Benchmark Results",
            "",
            "## HumanEval",
            "",
            "| **Competitors** | **Pass@1** | **Pass@10** |",
            "| --- | --- | --- |",
        ]
    )
    for name in ["Ours Graphsignal", "TokenCleaning", "XTF", "LLM-CleanCode", "CLEAR"]:
        p1, p10 = scores("humaneval", "python", name)
        lines.append(f"| **{name}** | {p1} | {p10} |")

    lines.extend(
        [
            "",
            "## SAFIM",
            "",
            "| **Baseline** | **Language** | **Pass@1** | **Pass@10** |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name in ["Ours Graphsignal", "XTF", "CLEAR", "LLM-CleanCode", "TokenCleaning"]:
        for idx, lang in enumerate(["Python", "Java", "C++", "C#"]):
            key = {"Python": "python", "Java": "java", "C++": "cpp", "C#": "csharp"}[lang]
            p1, p10 = scores("safim", key, name)
            label = f"**{name}**" if idx == 0 else ""
            lines.append(f"| {label} | {lang} | {p1} | {p10} |")

    lines.extend(
        [
            "",
            "## McEval",
            "",
            "| **Baseline** | **Language** | **Pass@1** | **Pass@10** |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name in ["Ours Graphsignal", "TokenCleaning", "XTF", "CLEAR", "LLM-CleanCode"]:
        for idx, lang in enumerate(["C", "C++", "C#", "Go", "Java", "Python"]):
            key = {"C": "c", "C++": "cpp", "C#": "csharp", "Go": "go", "Java": "java", "Python": "python"}[lang]
            p1, p10 = scores("mceval", key, name)
            label = f"**{name}**" if idx == 0 else ""
            lines.append(f"| {label} | {lang} | {p1} | {p10} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


