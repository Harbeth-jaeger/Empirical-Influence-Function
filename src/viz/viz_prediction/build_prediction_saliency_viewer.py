#!/usr/bin/env python3
"""Build a standalone prediction/saliency comparison viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/viz_prediction/prediction_saliency_50.json"
DEFAULT_HTML = ROOT / "outputs/viz_prediction/base_vs_sft_prediction_saliency_viewer.html"


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>训练前后test样本预测效果对比</title>
<style>
:root {
  --blue:#3159d8;
  --ink:#0f172a;
  --muted:#657085;
  --line:#d8deeb;
  --bg:#f7f9fc;
  --code:#101423;
  --target:#f6c642;
  --orange:#ff7a1a;
  --green:#22c55e;
}
* { box-sizing:border-box; }
html, body { height:100%; margin:0; overflow:hidden; }
body {
  display:grid;
  grid-template-rows:auto auto minmax(0, 1fr);
  background:var(--bg);
  color:var(--ink);
  font:13px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.topbar {
  height:58px;
  display:flex;
  align-items:center;
  gap:18px;
  padding:0 22px;
  background:#fff;
  border-bottom:1px solid var(--line);
}
h1 { margin:0; color:var(--blue); font-size:22px; white-space:nowrap; }
.meta { color:var(--muted); font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.controls {
  min-height:62px;
  display:flex;
  align-items:end;
  gap:18px;
  padding:11px 22px 13px;
  background:#fff;
  border-bottom:1px solid var(--line);
}
.control { min-width:0; }
.sample { width:min(760px, 58vw); }
.topk { width:110px; }
label {
  display:block;
  margin:0 0 4px;
  color:var(--muted);
  font-size:10px;
  font-weight:800;
  letter-spacing:.08em;
  text-transform:uppercase;
}
select, input {
  width:100%;
  height:32px;
  border:1px solid var(--line);
  border-radius:7px;
  background:#fff;
  color:var(--ink);
  padding:0 10px;
  font:inherit;
}
.main {
  min-height:0;
  display:grid;
  grid-template-rows:minmax(0, 1fr) 210px;
  gap:14px;
  padding:14px;
}
.pred-grid {
  min-height:0;
  display:grid;
  grid-template-columns:minmax(0, 1fr) minmax(0, 1fr);
  gap:14px;
}
.bottom-grid {
  min-height:0;
  display:grid;
  grid-template-columns:minmax(0, 1.15fr) minmax(0, .85fr);
  gap:14px;
}
.panel {
  min-height:0;
  display:grid;
  grid-template-rows:auto minmax(0, 1fr);
  overflow:hidden;
  border:1px solid var(--line);
  border-radius:8px;
  background:#fff;
}
.head {
  min-height:58px;
  padding:12px 16px 9px;
  border-bottom:1px solid var(--line);
}
.head h2 { margin:0 0 2px; font-size:18px; }
.sub { color:var(--muted); font-size:12px; }
.code {
  min-height:0;
  overflow:auto;
  background:var(--code);
  color:#dbe5ff;
  padding:20px 26px 30px;
  white-space:pre-wrap;
  font:13px/1.62 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.gt {
  min-height:0;
  overflow:auto;
  padding:16px 20px;
  background:#0f172a;
  color:#e6edf7;
  white-space:pre-wrap;
  font:13px/1.62 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.salbox {
  min-height:0;
  overflow:auto;
  padding:16px 18px;
}
.chips { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.chip {
  display:inline-flex;
  align-items:center;
  max-width:100%;
  padding:4px 9px;
  border-radius:999px;
  border:1px solid var(--orange);
  color:#9a3412;
  background:#fff7ed;
  font:13px/1.2 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.selected {
  display:inline-block;
  margin:0 0 12px;
  padding:5px 9px;
  border-radius:999px;
  background:#fff8c5;
  color:#92400e;
  font-weight:800;
}
.hint { margin-top:12px; color:var(--muted); font-size:12px; }
.tok {
  display:inline;
  padding:0 1px;
  border-radius:3px;
  cursor:default;
}
.tok.prefix, .tok.suffix { color:#d6def4; }
.tok.completion { color:#d8ffe7; background:rgba(34,197,94,.08); }
.ours .tok.completion { cursor:pointer; }
.tok.target { background:var(--target); color:#111827; padding:0 3px; }
.tok.sal { border:1.5px solid var(--orange); padding:0 3px; color:#fff; }
.empty { color:var(--muted); }
@media (max-width: 900px) {
  html, body { overflow:auto; }
  body { display:block; }
  .topbar { height:auto; min-height:58px; flex-wrap:wrap; padding:12px 16px; }
  h1 { white-space:normal; }
  .controls { flex-wrap:wrap; padding:12px 16px; }
  .sample, .topk { width:100%; }
  .main { display:block; }
  .pred-grid, .bottom-grid { grid-template-columns:1fr; margin-bottom:14px; }
  .panel { min-height:360px; margin-bottom:14px; }
}
</style>
</head>
<body>
<header class="topbar">
  <h1>训练前后test样本预测效果对比</h1>
  <div class="meta">展示 Base 与 SFT/Ours 在同一测试样本上的预测变化，并查看 Ours 关键 token 的 saliency top-5</div>
</header>
<section class="controls">
  <div class="control sample"><label>Sample</label><select id="sampleSelect"></select></div>
  <div class="control topk"><label>Top K</label><input id="topK" type="number" min="1" max="20" value="5"></div>
</section>
<main class="main">
  <section class="pred-grid">
    <div class="panel">
      <div class="head"><h2>Base prediction</h2><div class="sub" id="baseSub"></div></div>
      <div class="code base" id="baseCode"></div>
    </div>
    <div class="panel">
      <div class="head"><h2>Ours/SFT prediction</h2><div class="sub" id="oursSub"></div></div>
      <div class="code ours" id="oursCode"></div>
    </div>
  </section>
  <section class="bottom-grid">
    <div class="panel">
      <div class="head"><h2>GT completion</h2><div class="sub" id="gtSub"></div></div>
      <div class="gt" id="gtBox"></div>
    </div>
    <div class="panel">
      <div class="head"><h2>Top-5 saliency source tokens</h2><div class="sub">点击 Ours/SFT prediction 中的 completion token 查看</div></div>
      <div class="salbox" id="salBox"></div>
    </div>
  </section>
</main>
<script>
const DATA = __VIEWER_DATA__;
const state = { sampleId: "", step: 0 };
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function samples(){ return (DATA.samples || []).filter(s => !s.skipped); }
function current(){ return samples().find(s => s.sample_id === state.sampleId) || samples()[0]; }
function target(){
  const s = current();
  const idx = s?.ours?.targets_by_step?.[String(state.step)];
  return idx === undefined ? null : s.ours.targets[String(idx)];
}
function displayToken(t){ return t?.display ?? t?.text ?? ""; }
function tokenText(t){ return String(t?.text ?? t?.display ?? "").replace(/\\n/g, "\n").replace(/\\t/g, "\t"); }
function salRows(){
  const t = target();
  const k = Number($("topK").value || 5);
  return (t?.sources || []).slice(0, k);
}
function salSet(){ return new Set(salRows().map(r => Number(r.idx))); }
function fillSamples(){
  const rows = samples();
  $("sampleSelect").innerHTML = rows.map(s => {
    const label = `#${s.filtered_index} · ${s.language} · ${s.raw_id || s.uid}`;
    return `<option value="${esc(s.sample_id)}">${esc(label)}</option>`;
  }).join("");
  if (!rows.find(s => s.sample_id === state.sampleId)) state.sampleId = rows[0]?.sample_id || "";
  $("sampleSelect").value = state.sampleId;
}
function chooseInitialStep(){
  const s = current();
  const steps = Object.keys(s?.ours?.targets_by_step || {}).map(Number).sort((a,b)=>a-b);
  state.step = steps[Math.min(8, Math.max(0, steps.length - 1))] ?? 0;
}
function tokenClass(t, model){
  const cls = ["tok", t.region || ""];
  const tgt = target();
  if (model === "ours" && tgt && Number(t.idx) === Number(tgt.target_idx)) cls.push("target");
  if (model === "ours" && salSet().has(Number(t.idx))) cls.push("sal");
  return cls.join(" ");
}
function renderCode(model, el){
  const s = current();
  const toks = model === "base" ? (s.base?.tokens || []) : (s.ours?.tokens || []);
  el.innerHTML = toks.map(t => {
    const clickable = model === "ours" && t.region === "completion" && t.step !== undefined ? ` data-step="${t.step}"` : "";
    return `<span class="${tokenClass(t, model)}" data-idx="${t.idx}"${clickable} title="#${t.idx} ${esc(t.region)} ${esc(displayToken(t))}">${esc(tokenText(t))}</span>`;
  }).join("");
  if (model === "ours") {
    el.querySelectorAll("[data-step]").forEach(n => n.onclick = () => {
      state.step = Number(n.dataset.step);
      renderAll(false);
    });
  }
}
function renderBottom(){
  const s = current();
  const t = target();
  $("gtBox").textContent = s?.ground_truth || "";
  $("gtSub").textContent = s ? `#${s.filtered_index} · ${s.uid}` : "";
  const rows = salRows();
  const selected = t ? `<div class="selected">selected: ${esc(displayToken(t))}</div>` : "";
  const chips = rows.length ? `<div class="chips">${rows.map(r => `<span class="chip">${esc(displayToken(r))}</span>`).join("")}</div>` : '<span class="empty">No saliency for selected token</span>';
  $("salBox").innerHTML = selected + chips + '<div class="hint">橙色方框同步标出 Ours/SFT prediction 中对应的 source tokens。</div>';
}
function renderDetails(){
  const s = current();
  $("baseSub").textContent = s ? `#${s.filtered_index} · before SFT` : "";
  const t = target();
  $("oursSub").textContent = t ? `after SFT · selected step ${state.step}, target #${t.target_idx}` : "after SFT";
}
function renderAll(reset=true){
  if (reset) chooseInitialStep();
  renderCode("base", $("baseCode"));
  renderCode("ours", $("oursCode"));
  renderDetails();
  renderBottom();
}
$("sampleSelect").onchange = () => { state.sampleId = $("sampleSelect").value; renderAll(true); };
$("topK").onchange = () => renderAll(false);
fillSamples();
renderAll(true);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--output_path", default=str(DEFAULT_HTML))
    args = parser.parse_args()
    data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    html = HTML.replace("__VIEWER_DATA__", json.dumps(data, ensure_ascii=False))
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
