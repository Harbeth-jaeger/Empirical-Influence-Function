#!/usr/bin/env python3
'Build a standalone two-model prediction/saliency viewer.'

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>双模型预测与 saliency 对比</title>
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
  --red:#ef4444;
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
  min-height:58px;
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
  min-height:72px;
  display:flex;
  align-items:end;
  gap:18px;
  padding:11px 22px 13px;
  background:#fff;
  border-bottom:1px solid var(--line);
}
.control { min-width:0; }
.category { width:280px; }
.sample { width:min(760px, 48vw); }
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
  grid-template-rows:minmax(0, 1fr) 218px;
  gap:14px;
  padding:14px;
}
.pred-grid, .bottom-grid {
  min-height:0;
  display:grid;
  grid-template-columns:minmax(0, 1fr) minmax(0, 1fr);
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
  min-height:64px;
  padding:12px 16px 9px;
  border-bottom:1px solid var(--line);
}
.head h2 { margin:0 0 3px; font-size:18px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.sub { color:var(--muted); font-size:12px; }
.badge {
  display:inline-flex;
  align-items:center;
  padding:2px 8px;
  border-radius:999px;
  font-size:11px;
  font-weight:800;
  line-height:1.35;
}
.pass { background:#dcfce7; color:#15803d; }
.fail { background:#fee2e2; color:#b91c1c; }
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
.tok.completion { color:#d8ffe7; background:rgba(34,197,94,.08); cursor:pointer; }
.tok.completion:hover { outline:1px solid rgba(246,196,66,.75); }
.tok.target { background:var(--target); color:#111827; padding:0 3px; }
.tok.sal { border:1.5px solid var(--orange); padding:0 3px; color:#fff; }
.empty { color:var(--muted); }
@media (max-width: 900px) {
  html, body { overflow:auto; }
  body { display:block; }
  .topbar { height:auto; min-height:58px; flex-wrap:wrap; padding:12px 16px; }
  h1 { white-space:normal; }
  .controls { flex-wrap:wrap; padding:12px 16px; }
  .category, .sample, .topk { width:100%; }
  .main { display:block; }
  .pred-grid, .bottom-grid { grid-template-columns:1fr; margin-bottom:14px; }
  .panel { min-height:360px; margin-bottom:14px; }
}
</style>
</head>
<body>
<header class="topbar">
  <h1 id="title">双模型预测与 saliency 对比</h1>
  <div class="meta" id="subtitle"></div>
</header>
<section class="controls">
  <div class="control category"><label>Category</label><select id="categorySelect"></select></div>
  <div class="control sample"><label>Sample</label><select id="sampleSelect"></select></div>
  <div class="control topk"><label>Top K</label><input id="topK" type="number" min="1" max="30" value="10"></div>
</section>
<main class="main">
  <section class="pred-grid">
    <div class="panel">
      <div class="head"><h2 id="modelATitle"></h2><div class="sub" id="modelASub"></div></div>
      <div class="code" id="modelACode"></div>
    </div>
    <div class="panel">
      <div class="head"><h2 id="modelBTitle"></h2><div class="sub" id="modelBSub"></div></div>
      <div class="code" id="modelBCode"></div>
    </div>
  </section>
  <section class="bottom-grid">
    <div class="panel">
      <div class="head"><h2>GT completion</h2><div class="sub" id="gtSub"></div></div>
      <div class="gt" id="gtBox"></div>
    </div>
    <div class="panel">
      <div class="head"><h2>Selected token saliency</h2><div class="sub">两个模型的 completion token 都可点击；橙色方框标出当前模型 top-k source tokens。</div></div>
      <div class="salbox" id="salBox"></div>
    </div>
  </section>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const $ = (id) => document.getElementById(id);
const state = { category: 'all', sampleId: null, selected: { model_a: null, model_b: null }, selectedModel: 'model_b' };

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function codeText(s) {
  return esc(s).replace(/\\n/g, '\n').replace(/\\t/g, '\t');
}
function badge(label, ok) {
  return `<span class="badge ${ok ? 'pass' : 'fail'}">${label} ${ok ? 'PASS' : 'FAIL'}</span>`;
}
function categories() {
  const cats = new Set();
  for (const s of DATA.samples || []) for (const c of (s.categories || [])) cats.add(c);
  return ['all', ...Array.from(cats).sort()];
}
function categoryLabel(c) {
  if (c === 'all') return `全部样本 (${(DATA.samples || []).length})`;
  const label = (DATA.categories || {})[c] || c;
  const n = (DATA.samples || []).filter(s => (s.categories || []).includes(c)).length;
  return `${label} (${n})`;
}
function filteredSamples() {
  const samples = DATA.samples || [];
  if (state.category === 'all') return samples;
  return samples.filter(s => (s.categories || []).includes(state.category));
}
function sampleLabel(s) {
  const cats = (s.categories || []).join(', ');
  return `#${s.filtered_index ?? '?'} · ${s.language || ''} · ${s.raw_id || s.uid || ''} · ${cats}`;
}
function currentSample() {
  const samples = filteredSamples();
  return samples.find(s => s.sample_id === state.sampleId) || samples[0] || null;
}
function salSourceSet(modelData, selectedIdx, topK) {
  if (selectedIdx == null || !modelData || !modelData.targets) return new Set();
  const t = modelData.targets[String(selectedIdx)];
  if (!t) return new Set();
  return new Set((t.sources || []).slice(0, topK).map(x => Number(x.idx)));
}
function renderCode(el, sample, key) {
  const modelData = sample.models?.[key] || {};
  if (modelData.skipped) {
    el.innerHTML = `<span class="empty">skipped: ${esc(modelData.reason || '')}</span>`;
    return;
  }
  const topK = Number($('topK').value || DATA.top_k || 10);
  const selectedIdx = state.selected[key];
  const salSet = salSourceSet(modelData, selectedIdx, topK);
  const chunks = [];
  for (const tok of modelData.tokens || []) {
    const idx = Number(tok.idx);
    const cls = ['tok', tok.region || ''];
    if (idx === selectedIdx) cls.push('target');
    if (salSet.has(idx)) cls.push('sal');
    const attrs = [`class="${cls.join(' ')}"`, `data-idx="${idx}"`, `data-model="${key}"`];
    if (tok.region === 'completion') attrs.push('title="click to inspect saliency"');
    chunks.push(`<span ${attrs.join(' ')}>${codeText(tok.display ?? tok.text ?? '')}</span>`);
  }
  el.innerHTML = chunks.join('');
  el.querySelectorAll('.tok.completion').forEach(node => {
    node.addEventListener('click', () => {
      state.selectedModel = node.dataset.model;
      state.selected[node.dataset.model] = Number(node.dataset.idx);
      renderAll(false);
    });
  });
}
function renderSal(sample) {
  const key = state.selectedModel;
  const modelName = key === 'model_a' ? DATA.model_a_name : DATA.model_b_name;
  const modelData = sample.models?.[key] || {};
  const idx = state.selected[key];
  const target = idx == null ? null : modelData.targets?.[String(idx)];
  const box = $('salBox');
  if (!target) {
    box.innerHTML = `<div class="empty">点击左/右任一模型的 completion token 查看 saliency top-k。</div>`;
    return;
  }
  const topK = Number($('topK').value || DATA.top_k || 10);
  const chips = (target.sources || []).slice(0, topK).map(src => {
    const val = Number(src.value || 0);
    return `<span class="chip">#${src.rank} ${esc(src.display)} · ${val.toExponential(2)}</span>`;
  }).join('');
  box.innerHTML = `
    <div class="selected">${esc(modelName)} · step ${target.step} · #${target.target_idx} · ${esc(target.display)}</div>
    <div class="chips">${chips || '<span class="empty">No source tokens</span>'}</div>
    <div class="hint">当前 completion source: ${esc(modelData.completion_source || DATA.completion_source || 'prediction')}</div>
  `;
}
function ensureSelections(sample) {
  for (const key of ['model_a', 'model_b']) {
    const md = sample.models?.[key] || {};
    const generated = md.generated_token_indices || [];
    if (!generated.includes(state.selected[key])) state.selected[key] = generated[0] ?? null;
  }
}
function renderAll(resetSample) {
  if (resetSample) {
    const samples = filteredSamples();
    state.sampleId = samples[0]?.sample_id || null;
    state.selected = { model_a: null, model_b: null };
  }
  const sample = currentSample();
  if (!sample) return;
  state.sampleId = sample.sample_id;
  ensureSelections(sample);

  $('title').textContent = DATA.title || '双模型预测与 saliency 对比';
  $('subtitle').textContent = DATA.subtitle || '';
  $('topK').value = $('topK').value || DATA.top_k || 10;
  $('gtSub').textContent = `#${sample.filtered_index ?? '?'} · ${sample.raw_id || sample.uid || ''}`;
  $('gtBox').textContent = sample.ground_truth || '';

  for (const [key, prefix] of [['model_a', 'A'], ['model_b', 'B']]) {
    const md = sample.models?.[key] || {};
    const name = key === 'model_a' ? DATA.model_a_name : DATA.model_b_name;
    const title = key === 'model_a' ? $('modelATitle') : $('modelBTitle');
    const sub = key === 'model_a' ? $('modelASub') : $('modelBSub');
    title.innerHTML = `${esc(name)} ${badge('P@1', !!md.pass1)} ${badge('P@10', !!md.pass10)}`;
    sub.textContent = `${prefix} · ${sample.raw_id || sample.uid || ''}`;
  }
  renderCode($('modelACode'), sample, 'model_a');
  renderCode($('modelBCode'), sample, 'model_b');
  renderSal(sample);
}
function initControls() {
  const catSel = $('categorySelect');
  catSel.innerHTML = categories().map(c => `<option value="${esc(c)}">${esc(categoryLabel(c))}</option>`).join('');
  catSel.addEventListener('change', () => {
    state.category = catSel.value;
    populateSamples(true);
  });
  $('topK').value = DATA.top_k || 10;
  $('topK').addEventListener('input', () => renderAll(false));
  populateSamples(true);
}
function populateSamples(reset) {
  const sampleSel = $('sampleSelect');
  const samples = filteredSamples();
  sampleSel.innerHTML = samples.map(s => `<option value="${esc(s.sample_id)}">${esc(sampleLabel(s))}</option>`).join('');
  if (reset) state.sampleId = samples[0]?.sample_id || null;
  sampleSel.value = state.sampleId || '';
  sampleSel.onchange = () => {
    state.sampleId = sampleSel.value;
    state.selected = { model_a: null, model_b: null };
    renderAll(false);
  };
  renderAll(false);
}
initControls();
</script>
</body>
</html>
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    data_path = Path(args.data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", blob)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
