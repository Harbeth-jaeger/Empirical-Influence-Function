#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path

DEFAULT_TASKS = [
    ("humaneval_infilling_python.jsonl", "single_line"),
    ("humaneval_infilling_python.jsonl", "multi_line"),
    ("humaneval_infilling_python.jsonl", "random_span"),
    ("humaneval_infilling_python.jsonl", "random_span_light"),
    ("safim_python.jsonl", "algorithmic_block"),
    ("safim_python.jsonl", "control_flow_expression"),
    ("safim_python.jsonl", "api_function_call"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_newlines(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def mask_parts_for_target(target: str) -> tuple[str, str]:
    """Render the mask in the same syntactic slot occupied by the target span."""
    if not target:
        return "", ""

    first_line = target.splitlines(keepends=True)[0]
    leading_ws = re.match(r"[ \t]*", first_line).group(0)
    if "\n" in target:
        return leading_ws, "\n"
    return leading_ws, ""


def build_view_data(test_dir: Path, samples_per_task: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    file_cache: dict[str, list[dict]] = {}
    groups = []

    for file_name, task_type in DEFAULT_TASKS:
        rows = file_cache.setdefault(file_name, read_jsonl(test_dir / file_name))
        candidates = [row for row in rows if row.get("task_type") == task_type]
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected = shuffled[:samples_per_task]
        examples = []
        for idx, row in enumerate(selected, start=1):
            prefix = normalize_newlines(row.get("prefix"))
            suffix = normalize_newlines(row.get("suffix"))
            target = normalize_newlines(row.get("target"))
            mask_prefix, mask_suffix = mask_parts_for_target(target)
            examples.append(
                {
                    "index": idx,
                    "uid": row.get("uid"),
                    "official_task_id": row.get("official_task_id"),
                    "benchmark": row.get("benchmark"),
                    "language": row.get("language"),
                    "file": file_name,
                    "task_type": task_type,
                    "prefix": prefix,
                    "suffix": suffix,
                    "target": target,
                    "mask_prefix": mask_prefix,
                    "mask_suffix": mask_suffix,
                    "prompt_with_mask": f"{prefix}{mask_prefix}[MASK]{mask_suffix}{suffix}",
                    "filled_code": f"{prefix}{target}{suffix}",
                    "target_lines": line_count(target),
                    "target_chars": len(target),
                }
            )
        groups.append(
            {
                "file": file_name,
                "task_type": task_type,
                "num_available": len(candidates),
                "num_shown": len(examples),
                "examples": examples,
            }
        )
    return groups


def make_example(row: dict, file_name: str, task_type: str, index: int) -> dict:
    prefix = normalize_newlines(row.get("prefix"))
    suffix = normalize_newlines(row.get("suffix"))
    target = normalize_newlines(row.get("target"))
    mask_prefix, mask_suffix = mask_parts_for_target(target)
    return {
        "index": index,
        "uid": row.get("uid"),
        "official_task_id": row.get("official_task_id") or row.get("uid"),
        "benchmark": row.get("benchmark"),
        "language": row.get("language"),
        "file": file_name,
        "task_type": task_type,
        "prefix": prefix,
        "suffix": suffix,
        "target": target,
        "mask_prefix": mask_prefix,
        "mask_suffix": mask_suffix,
        "prompt_with_mask": f"{prefix}{mask_prefix}[MASK]{mask_suffix}{suffix}",
        "filled_code": f"{prefix}{target}{suffix}",
        "target_lines": line_count(target),
        "target_chars": len(target),
    }


def build_auto_view_data(test_dir: Path, samples_per_task: int, seed: int, task_types: set[str] | None) -> list[dict]:
    rng = random.Random(seed)
    groups = []
    for path in sorted(test_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        available_types = sorted({str(row.get("task_type") or "unknown") for row in rows})
        for task_type in available_types:
            if task_types is not None and task_type not in task_types:
                continue
            candidates = [row for row in rows if str(row.get("task_type") or "unknown") == task_type]
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            selected = shuffled[:samples_per_task]
            examples = [make_example(row, path.name, task_type, idx) for idx, row in enumerate(selected, start=1)]
            groups.append(
                {
                    "file": path.name,
                    "task_type": task_type,
                    "num_available": len(candidates),
                    "num_shown": len(examples),
                    "examples": examples,
                }
            )
    return groups


SAFIM_VIEWER_FILE_RE = re.compile(r"^(train|official)_safim_(python|java|cpp|csharp)\.jsonl$")


def build_paired_safim_view_data(
    test_dir: Path,
    samples_per_task: int,
    seed: int,
    task_types: set[str] | None,
    languages: set[str] | None,
) -> dict:
    rng = random.Random(seed)
    groups: dict[str, dict[str, dict[str, dict]]] = {}

    for path in sorted(test_dir.glob("*.jsonl")):
        match = SAFIM_VIEWER_FILE_RE.match(path.name)
        if not match:
            continue
        source, language = match.groups()
        if languages is not None and language not in languages:
            continue
        rows = read_jsonl(path)
        for current_task in sorted({str(row.get("task_type") or "unknown") for row in rows}):
            if task_types is not None and current_task not in task_types:
                continue
            candidates = [row for row in rows if str(row.get("task_type") or "unknown") == current_task]
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            selected = shuffled[:samples_per_task]
            examples = [make_example(row, path.name, current_task, idx) for idx, row in enumerate(selected, start=1)]
            groups.setdefault(language, {}).setdefault(current_task, {})[source] = {
                "file": path.name,
                "source": source,
                "language": language,
                "task_type": current_task,
                "num_available": len(candidates),
                "num_shown": len(examples),
                "examples": examples,
            }

    complete_groups: dict[str, dict[str, dict[str, dict]]] = {}
    for language, task_map in groups.items():
        for current_task, source_map in task_map.items():
            if "train" in source_map and "official" in source_map:
                complete_groups.setdefault(language, {})[current_task] = source_map

    return {
        "languages": sorted(complete_groups),
        "task_types": sorted({task for task_map in complete_groups.values() for task in task_map}),
        "samples_per_task": samples_per_task,
        "groups": complete_groups,
    }


def render_paired_safim_html(data_payload: dict, title: str) -> str:
    data = json.dumps(data_payload, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --bg: #f6f7f9; --panel: #ffffff; --ink: #17202a; --muted: #657181; --line: #d8dde6; --blue: #3157d5; --mask: #fff0b8; --code: #101522; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ min-height: 62px; display: flex; align-items: center; gap: 18px; padding: 12px 18px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 5; }}
    h1 {{ margin: 0; font-size: 19px; font-weight: 750; letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .controls {{ position: sticky; top: 62px; z-index: 4; background: rgba(246, 247, 249, .96); border-bottom: 1px solid var(--line); padding: 10px 18px; display: grid; gap: 10px; }}
    .controls.collapsed {{ padding: 8px 18px; }}
    .controls.collapsed .control-row {{ display: none; }}
    .controls.collapsed .compact-row {{ display: flex; }}
    .compact-row {{ display: none; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
    .compact-summary {{ color: var(--muted); font-size: 13px; }}
    .compact-summary strong {{ color: var(--ink); }}
    .control-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .label {{ width: 82px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 7px; padding: 7px 10px; cursor: pointer; font-size: 13px; }}
    button:hover {{ border-color: #9aa9c5; }}
    button.active {{ background: var(--blue); border-color: var(--blue); color: #fff; }}
    button.sample-index {{ min-width: 38px; padding: 6px 8px; }}
    .toggle-controls {{ font-weight: 700; }}
    .main {{ padding: 16px 18px 22px; }}
    .compare-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; align-items: start; }}
    .pane {{ min-width: 0; }}
    .pane-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin: 0 0 8px; }}
    .pane-title {{ font-size: 17px; font-weight: 760; }}
    .pane-sub {{ color: var(--muted); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 10px; min-height: 30px; }}
    .chip {{ border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 5px 8px; color: var(--muted); font-size: 12px; }}
    .chip strong {{ color: var(--ink); }}
    .code-wrap {{ border: 1px solid #22283a; border-radius: 8px; overflow: hidden; background: var(--code); }}
    .code-title {{ display: flex; justify-content: space-between; gap: 12px; padding: 8px 10px; border-bottom: 1px solid #252c40; color: #bec8d9; font-size: 12px; background: #151b2b; }}
    pre {{ margin: 0; padding: 14px; overflow: auto; color: #eef2ff; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 13px; line-height: 1.5; tab-size: 4; white-space: pre; max-height: 68vh; }}
    .mask {{ background: var(--mask); color: #141414; border-radius: 4px; padding: 1px 5px; font-weight: 800; }}
    .target-box {{ margin-top: 10px; border: 1px solid var(--line); background: #fff; border-radius: 8px; overflow: hidden; }}
    .target-box h2 {{ margin: 0; padding: 9px 10px; border-bottom: 1px solid var(--line); font-size: 13px; }}
    .target-box pre {{ background: #fff; color: #17202a; font-size: 13px; max-height: 180px; }}
    .empty {{ color: var(--muted); padding: 16px; background: #fff; border: 1px solid var(--line); border-radius: 8px; }}
    @media (max-width: 1100px) {{ .compare-grid {{ grid-template-columns: 1fr; }} .controls {{ top: 0; }} header {{ position: static; }} }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1><div class="meta">Choose language and subtask · left train, right official test · same sample index on both sides</div></header>
  <section id="controls" class="controls">
    <div class="compact-row"><div id="compact-summary" class="compact-summary"></div><button id="show-controls" class="toggle-controls">Show controls</button></div>
    <div class="control-row"><div class="label">Controls</div><button id="hide-controls" class="toggle-controls">Hide controls</button></div>
    <div class="control-row"><div class="label">Language</div><div id="language-controls"></div></div>
    <div class="control-row"><div class="label">Subtask</div><div id="task-controls"></div></div>
    <div class="control-row"><div class="label">Sample</div><div id="sample-controls"></div></div>
    <div class="control-row"><div class="label">View</div><div id="view-controls"></div></div>
  </section>
  <main class="main"><div id="compare" class="compare-grid"></div></main>
  <script>
    const DATA = {data};
    const VIEW_LABELS = {{prompt: "Prompt With Mask", filled: "Filled Code", prefix: "Prefix", suffix: "Suffix"}};
    let state = {{language: DATA.languages[0], task: null, sampleIndex: 0, view: "prompt", controlsCollapsed: false}};
    function escapeHtml(text) {{ return String(text ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch])); }}
    function availableTasks(language) {{ return Object.keys(DATA.groups[language] || {{}}).sort(); }}
    function currentPair() {{ const tasks = availableTasks(state.language); if (!state.task || !tasks.includes(state.task)) state.task = tasks[0]; return DATA.groups[state.language]?.[state.task] || null; }}
    function button(label, active, onClick, className="") {{ const btn = document.createElement("button"); btn.textContent = label; btn.className = className; if (active) btn.classList.add("active"); btn.addEventListener("click", onClick); return btn; }}
    function updateControlsVisibility() {{
      const controls = document.getElementById("controls");
      controls.classList.toggle("collapsed", state.controlsCollapsed);
      const summary = document.getElementById("compact-summary");
      summary.innerHTML = `<strong>${{escapeHtml(state.language)}}</strong> · ${{escapeHtml(state.task || "unknown")}} · sample <strong>${{state.sampleIndex + 1}}</strong> · ${{escapeHtml(VIEW_LABELS[state.view])}}`;
    }}
    function renderControls() {{
      document.getElementById("hide-controls").onclick = () => {{ state.controlsCollapsed = true; updateControlsVisibility(); }};
      document.getElementById("show-controls").onclick = () => {{ state.controlsCollapsed = false; updateControlsVisibility(); }};
      const langRoot = document.getElementById("language-controls"); langRoot.innerHTML = "";
      DATA.languages.forEach(lang => langRoot.appendChild(button(lang, lang === state.language, () => {{ state.language = lang; state.task = null; state.sampleIndex = 0; render(); }})));
      const taskRoot = document.getElementById("task-controls"); taskRoot.innerHTML = "";
      availableTasks(state.language).forEach(task => taskRoot.appendChild(button(task, task === state.task, () => {{ state.task = task; state.sampleIndex = 0; render(); }})));
      const pair = currentPair(); const shown = pair ? Math.min(pair.train.examples.length, pair.official.examples.length) : 0;
      if (state.sampleIndex >= shown) state.sampleIndex = Math.max(0, shown - 1);
      const sampleRoot = document.getElementById("sample-controls"); sampleRoot.innerHTML = "";
      for (let i = 0; i < shown; i++) sampleRoot.appendChild(button(String(i + 1), i === state.sampleIndex, () => {{ state.sampleIndex = i; render(); }}, "sample-index"));
      const viewRoot = document.getElementById("view-controls"); viewRoot.innerHTML = "";
      Object.entries(VIEW_LABELS).forEach(([key, label]) => viewRoot.appendChild(button(label, key === state.view, () => {{ state.view = key; render(); }})));
      updateControlsVisibility();
    }}
    function codeHtml(sample) {{
      if (state.view === "prompt") return escapeHtml(sample.prefix) + escapeHtml(sample.mask_prefix) + '<span class="mask">[MASK]</span>' + escapeHtml(sample.mask_suffix) + escapeHtml(sample.suffix);
      return escapeHtml(state.view === "filled" ? sample.filled_code : state.view === "prefix" ? sample.prefix : sample.suffix);
    }}
    function renderPane(source, group) {{
      const sample = group.examples[state.sampleIndex];
      if (!sample) return `<section class="pane"><div class="empty">No ${{source}} sample for this bucket.</div></section>`;
      return `<section class="pane"><div class="pane-head"><div class="pane-title">${{source === "train" ? "Train" : "Official Test"}}</div><div class="pane-sub">${{escapeHtml(group.file)}} · ${{group.num_shown}}/${{group.num_available}}</div></div><div class="chips"><span class="chip"><strong>lang:</strong> ${{escapeHtml(sample.language)}}</span><span class="chip"><strong>task:</strong> ${{escapeHtml(sample.task_type)}}</span><span class="chip"><strong>id:</strong> ${{escapeHtml(sample.official_task_id)}}</span><span class="chip"><strong>uid:</strong> ${{escapeHtml(sample.uid)}}</span></div><section class="code-wrap"><div class="code-title"><span>${{escapeHtml(VIEW_LABELS[state.view])}}</span><span>${{sample.target_lines}} target lines · ${{sample.target_chars}} target chars</span></div><pre>${{codeHtml(sample)}}</pre></section><section class="target-box"><h2>Target Completion</h2><pre>${{escapeHtml(sample.target)}}</pre></section></section>`;
    }}
    function renderCompare() {{ const root = document.getElementById("compare"); const pair = currentPair(); root.innerHTML = pair ? renderPane("train", pair.train) + renderPane("official", pair.official) : '<div class="empty">No paired train/official data found.</div>'; }}
    function render() {{ currentPair(); renderControls(); renderCompare(); }}
    render();
  </script>
</body>
</html>
'''


def render_html(groups: list[dict], title: str) -> str:
    data = json.dumps(groups, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657181;
      --line: #d8dde6;
      --blue: #3157d5;
      --green: #0f8f68;
      --mask: #fff0b8;
      --code: #101522;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 56px;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      height: calc(100vh - 56px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      overflow: auto;
      padding: 14px;
    }}
    main {{
      overflow: auto;
      padding: 18px;
    }}
    .group {{
      margin-bottom: 16px;
    }}
    .group-title {{
      display: grid;
      gap: 5px;
      padding: 8px 6px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .file {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-transform: none;
      letter-spacing: 0;
      color: var(--ink);
      font-size: 12px;
    }}
    button.sample {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      padding: 9px 10px;
      margin: 4px 0;
      text-align: left;
      cursor: pointer;
      display: grid;
      gap: 4px;
      min-height: 54px;
    }}
    button.sample:hover {{ border-color: #9aa9c5; }}
    button.sample.active {{
      border-color: var(--blue);
      background: #eef2ff;
    }}
    .sample-line {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
    }}
    .uid {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 5px 9px;
      color: var(--muted);
      font-size: 12px;
    }}
    .chip strong {{ color: var(--ink); }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 14px 0;
    }}
    .tab {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    .tab.active {{
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }}
    .code-wrap {{
      border: 1px solid #22283a;
      border-radius: 8px;
      overflow: hidden;
      background: var(--code);
      min-height: 300px;
    }}
    .code-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 12px;
      border-bottom: 1px solid #252c40;
      color: #bec8d9;
      font-size: 12px;
      background: #151b2b;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      overflow: auto;
      color: #eef2ff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 14px;
      line-height: 1.55;
      tab-size: 4;
      white-space: pre;
    }}
    .mask {{
      background: var(--mask);
      color: #141414;
      border-radius: 4px;
      padding: 1px 5px;
      font-weight: 700;
    }}
    .target-box {{
      margin-top: 14px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
    }}
    .target-box h2 {{
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    .target-box pre {{
      background: #fff;
      color: #17202a;
      font-size: 14px;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; height: auto; }}
      aside {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }}
      main {{ min-height: 58vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">Python subtasks · 10 examples each · prefix + [MASK] + suffix</div>
  </header>
  <div class="layout">
    <aside id="sidebar"></aside>
    <main>
      <div class="toolbar">
        <div class="chips" id="chips"></div>
      </div>
      <div class="tabs">
        <button class="tab active" data-view="prompt">Prompt With Mask</button>
        <button class="tab" data-view="filled">Filled Code</button>
        <button class="tab" data-view="prefix">Prefix</button>
        <button class="tab" data-view="suffix">Suffix</button>
      </div>
      <section class="code-wrap">
        <div class="code-title">
          <span id="code-title">Prompt With Mask</span>
          <span id="code-stats"></span>
        </div>
        <pre id="code"></pre>
      </section>
      <section class="target-box">
        <h2>Target Completion</h2>
        <pre id="target"></pre>
      </section>
    </main>
  </div>
  <script>
    const DATA = {data};
    let current = DATA[0].examples[0];
    let currentView = "prompt";

    function escapeHtml(text) {{
      return String(text ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[ch]));
    }}

    function renderCode() {{
      const code = document.getElementById("code");
      let title = "Prompt With Mask";
      let content = "";
      if (currentView === "prompt") {{
        title = "Prompt With Mask";
        content = escapeHtml(current.prefix)
          + escapeHtml(current.mask_prefix)
          + '<span class="mask">[MASK]</span>'
          + escapeHtml(current.mask_suffix)
          + escapeHtml(current.suffix);
        code.innerHTML = content;
      }} else if (currentView === "filled") {{
        title = "Filled Code";
        code.textContent = current.filled_code;
      }} else if (currentView === "prefix") {{
        title = "Prefix";
        code.textContent = current.prefix;
      }} else {{
        title = "Suffix";
        code.textContent = current.suffix;
      }}
      document.getElementById("code-title").textContent = title;
      document.getElementById("code-stats").textContent =
        `${{current.target_lines}} target lines · ${{current.target_chars}} target chars`;
      document.getElementById("target").textContent = current.target;
      document.getElementById("chips").innerHTML = [
        ["file", current.file],
        ["task", current.task_type],
        ["id", current.official_task_id],
        ["uid", current.uid],
      ].map(([k, v]) => `<span class="chip"><strong>${{escapeHtml(k)}}:</strong> ${{escapeHtml(v)}}</span>`).join("");
    }}

    function renderSidebar() {{
      const root = document.getElementById("sidebar");
      root.innerHTML = "";
      DATA.forEach(group => {{
        const section = document.createElement("section");
        section.className = "group";
        section.innerHTML = `
          <div class="group-title">
            <div>${{escapeHtml(group.task_type)}} · ${{group.num_shown}}/${{group.num_available}}</div>
            <div class="file">${{escapeHtml(group.file)}}</div>
          </div>`;
        group.examples.forEach(sample => {{
          const btn = document.createElement("button");
          btn.className = "sample";
          btn.innerHTML = `
            <div class="sample-line">
              <strong>#${{sample.index}}</strong>
              <span>${{sample.target_lines}} lines · ${{sample.target_chars}} chars</span>
            </div>
            <div class="uid">${{escapeHtml(sample.official_task_id)}}</div>`;
          btn.addEventListener("click", () => {{
            current = sample;
            document.querySelectorAll("button.sample").forEach(x => x.classList.remove("active"));
            btn.classList.add("active");
            renderCode();
          }});
          section.appendChild(btn);
        }});
        root.appendChild(section);
      }});
      const first = document.querySelector("button.sample");
      if (first) first.classList.add("active");
    }}

    document.querySelectorAll(".tab").forEach(btn => {{
      btn.addEventListener("click", () => {{
        currentView = btn.dataset.view;
        document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
        btn.classList.add("active");
        renderCode();
      }});
    }});

    renderSidebar();
    renderCode();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight static viewer for Python FIM benchmark subtasks.")
    parser.add_argument("--test-dir", type=Path, default=Path("/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/external/benchmark/test_data"))
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark/data_analysis/python_subtask_viewer.html"))
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--title", default="Python FIM Benchmark Subtask Viewer")
    parser.add_argument("--auto-groups", action="store_true", help="Group every JSONL in --test-dir by its task_type field.")
    parser.add_argument("--task-types", nargs="*", default=None, help="Optional task_type allowlist for --auto-groups or --paired-safim.")
    parser.add_argument("--languages", nargs="*", default=None, help="Optional language allowlist for --paired-safim.")
    parser.add_argument("--paired-safim", action="store_true", help="Render train-vs-official paired SAFIM viewer from train_safim_*.jsonl and official_safim_*.jsonl.")
    args = parser.parse_args()

    task_types = set(args.task_types) if args.task_types else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.paired_safim:
        languages = set(args.languages) if args.languages else None
        payload = build_paired_safim_view_data(args.test_dir, args.samples_per_task, args.seed, task_types, languages)
        args.out.write_text(render_paired_safim_html(payload, args.title), encoding="utf-8")
        pairs = sum(len(task_map) for task_map in payload["groups"].values())
        samples = pairs * args.samples_per_task * 2
        print(json.dumps({"out": str(args.out), "pairs": pairs, "samples": samples}, ensure_ascii=False, indent=2))
    else:
        if args.auto_groups:
            groups = build_auto_view_data(args.test_dir, args.samples_per_task, args.seed, task_types)
        else:
            groups = build_view_data(args.test_dir, args.samples_per_task, args.seed)
        args.out.write_text(render_html(groups, args.title), encoding="utf-8")
        total = sum(group["num_shown"] for group in groups)
        print(json.dumps({"out": str(args.out), "groups": len(groups), "samples": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
