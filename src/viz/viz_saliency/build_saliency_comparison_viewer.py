#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/viz_saliency/saliency_comparison_data.json"
DEFAULT_TEMPLATE = ROOT / "src/tools/viz_saliency/saliency_comparison_viewer.html"
DEFAULT_OUTPUT = ROOT / "outputs/viz_saliency/saliency_comparison_viewer.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone saliency comparison HTML viewer.")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--template_path", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--default_scope", default="", choices=["", "prompt_code", "prompt_all", "all_causal"], help="Optional source scope selected by default in the viewer.")
    parser.add_argument("--title_suffix", default="", help="Optional suffix appended to the page title/header, e.g. ' V2'.")
    args = parser.parse_args()

    data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    template = Path(args.template_path).read_text(encoding="utf-8")
    if args.title_suffix:
        template = template.replace("Saliency Comparison Viewer", f"Saliency Comparison Viewer{args.title_suffix}")
    if args.default_scope:
        option_pattern = f'<option value="{args.default_scope}">'
        selected_pattern = f'<option value="{args.default_scope}" selected>'
        template = template.replace(' selected>', '>')
        template = template.replace(option_pattern, selected_pattern)
        if args.default_scope == "all_causal":
            old_select = '<select id="scopeSelect"><option value="prompt_code">FIM prefix/suffix code</option><option value="prompt_all">All prompt tokens</option><option value="all_causal" selected>All causal tokens</option></select>'
            new_select = '<select id="scopeSelect"><option value="all_causal" selected>All previous tokens</option><option value="prompt_code">FIM prefix/suffix code</option><option value="prompt_all">All prompt tokens</option></select>'
            template = template.replace(old_select, new_select)
    json_blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    if "__SALIENCY_DATA_JSON__" not in template:
        raise ValueError("template must contain __SALIENCY_DATA_JSON__ placeholder")
    html = template.replace("__SALIENCY_DATA_JSON__", json_blob)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"built viewer -> {out}")


if __name__ == "__main__":
    main()
