#!/usr/bin/env python3
"""Build the Base-vs-Ours annotation-alignment saliency viewer.

The input JSON is produced from the all-causal saliency data and annotation
edges.  This script only renders a standalone HTML file, with all data embedded
so the viewer can be copied or served as a single artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/visual_saliency/base_vs_ours_annotation_saliency_data_v2.json"
DEFAULT_HTML = ROOT / "outputs/visual_saliency/base_vs_ours_annotation_saliency_viewer_v2.html"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>训练前后saliency分布对比及与annotation对齐状况</title>
<style>
:root {
  --blue: #3159d8;
  --ink: #0f172a;
  --muted: #657085;
  --line: #d8deeb;
  --panel: #ffffff;
  --code-bg: #101423;
  --target: #f6c642;
  --orange: #ff7a1a;
  --green: #22c55e;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; overflow: hidden; }
body {
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  color: var(--ink);
  font: 12px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f9fc;
}
.topbar {
  height: 52px;
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 0 18px;
  background: #fff;
  border-bottom: 1px solid var(--line);
}
.topbar h1 { margin: 0; color: var(--blue); font-size: 20px; line-height: 1.1; white-space: nowrap; }
.topbar .meta { color: var(--muted); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.controlsbar {
  min-height: 72px;
  display: flex;
  flex-wrap: wrap;
  gap: 14px 24px;
  align-items: end;
  padding: 12px 28px 14px;
  background: #fff;
  border-bottom: 1px solid var(--line);
}
.control { min-width: 0; flex: 0 0 auto; }
.control.language { width: 180px; }
.control.sample { width: clamp(360px, 34vw, 560px); }
.control.topk { width: 120px; }
label {
  display: block;
  color: var(--muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin: 0 0 4px;
}
select, input[type="number"] {
  width: 100%;
  min-width: 0;
  height: 30px;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 0 9px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  overflow: hidden;
  text-overflow: ellipsis;
}
.filter-group {
  min-height: 30px;
  display: flex;
  flex: 1 1 380px;
  align-items: center;
  gap: 20px;
  padding-bottom: 1px;
}
.check {
  min-height: 30px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ink);
  font-size: 12px;
  line-height: 1.1;
  white-space: nowrap;
  justify-self: start;
}
.check input { width: 14px; height: 14px; }
.layout {
  height: auto;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
  padding: 16px;
  min-height: 0;
  background: #f7f9fc;
}
.pane {
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) minmax(92px, auto);
  background: var(--code-bg);
  color: #dbe5ff;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.pane-head {
  background: #fff;
  color: var(--ink);
  padding: 14px 18px 10px;
  border-bottom: 1px solid var(--line);
}
.pane-head h2 { margin: 0 0 2px; font-size: 18px; }
.sub { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px; color: var(--muted); font-size: 12px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 4px; vertical-align: -1px; margin-right: 5px; }
.code {
  min-height: 0;
  overflow: auto;
  padding: 22px 28px 36px;
  font: 12.5px/1.62 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  white-space: pre-wrap;
}
.summary {
  min-height: 0;
  overflow: auto;
  background: #fff;
  color: var(--ink);
  border-top: 1px solid var(--line);
  padding: 13px 18px 15px;
}
.summary h3 { margin: 0 0 7px; color: var(--blue); font-size: 13px; }
.summary-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 7px; }
.summary-label { color: var(--muted); font-weight: 800; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; margin-right: 2px; }
.chip {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 999px;
  background: #eef3ff;
  color: var(--blue);
  font: 12px/1.25 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.chip.salchip { border: 1px solid var(--orange); background: #fff7ed; color: #9a3412; }
.chip.annchip { text-decoration-line: underline; text-decoration-style: double; text-decoration-color: var(--green); text-decoration-thickness: 1.5px; text-underline-offset: 3px; background: #f0fdf4; color: #166534; }
.empty { color: var(--muted); }
.tok {
  display: inline;
  margin: 0;
  padding: 0 1px;
  border-radius: 3px;
  background: transparent;
  color: #dce6ff;
  cursor: default;
}
.tok.prompt, .tok.prompt_code, .tok.prompt_other { color: #cfd8ee; }
.tok.completion { color: #d9ffe9; }
.tok.clickable-target { cursor: pointer; background: rgba(246, 198, 66, .16); text-decoration: underline; text-decoration-style: dashed; text-decoration-color: var(--target); text-underline-offset: 3px; }
.tok.clickable-target:hover { background: rgba(246, 198, 66, .28); }
.tok.target { background: var(--target); color: #1f2937; padding: 0 3px; }
.tok.sal { border: 1.5px solid var(--orange); padding: 0 3px; color: #ffffff; }
.tok.ann { text-decoration-line: underline; text-decoration-style: double; text-decoration-color: var(--green); text-decoration-thickness: 1.8px; text-underline-offset: 3px; }
.tok.hidden { display: none; }
@media (max-width: 1200px) {
  .controlsbar { align-items: end; }
  .control.language { width: 150px; }
  .control.sample { width: min(100%, 520px); }
  .filter-group { flex-basis: 100%; }
  .check { min-height: 24px; }
}
@media (max-width: 820px) {
  html, body { overflow: auto; }
  body { display: block; }
  .topbar { height: auto; min-height: 52px; flex-wrap: wrap; padding: 10px 16px; }
  .topbar h1 { white-space: normal; }
  .controlsbar { padding: 12px 16px; }
  .control.language,
  .control.sample,
  .control.topk { width: 100%; }
  .filter-group { flex-basis: 100%; flex-wrap: wrap; }
  .layout { height: auto; grid-template-columns: 1fr; }
  .pane { min-height: 620px; }
}
</style>
</head>
<body>
<header class="topbar">
  <h1>训练前后saliency分布对比及与annotation对齐状况</h1>
  <div class="meta">验证模型训练后是否朝着我们预期的方向发展</div>
</header>
<section class="controlsbar">
  <div class="control language"><label>Language</label><select id="languageSelect"></select></div>
  <div class="control sample"><label>Sample</label><select id="sampleSelect"></select></div>
  <select id="stepSelect" style="display:none"></select>
  <div class="control topk"><label>Top K</label><input id="topKInput" type="number" min="1" max="20" value="20"></div>
  <div class="filter-group">
    <label class="check"><input id="hideSpecial" type="checkbox"> Hide special tokens</label>
    <label class="check"><input id="hideWhitespace" type="checkbox"> Hide whitespace tokens</label>
  </div>
</section>
<main class="layout">
  <section class="pane" id="basePane">
    <div class="pane-head">
      <h2 id="baseTitle">Base Qwen</h2>
      <div class="sub" id="baseSub"></div>
      <div class="legend"><span><i class="sw" style="background:#45588d"></i>prompt</span><span><i class="sw" style="background:#23744c"></i>completion</span><span><i class="sw" style="background:#ff7a1a"></i>saliency top-k</span></div>
    </div>
    <div class="code" id="baseCode"></div>
    <div class="summary" id="baseSummary"></div>
  </section>
  <section class="pane" id="oursPane">
    <div class="pane-head">
      <h2 id="oursTitle">Ours GraphSignal</h2>
      <div class="sub" id="oursSub"></div>
      <div class="legend"><span><i class="sw" style="background:#45588d"></i>prompt</span><span><i class="sw" style="background:#23744c"></i>completion</span><span><i class="sw" style="background:#ff7a1a"></i>saliency top-k</span><span><i class="sw" style="background:#22c55e"></i>annotation</span></div>
    </div>
    <div class="code" id="oursCode"></div>
    <div class="summary" id="oursSummary"></div>
  </section>
</main>
<script>
const DATA = __VIEWER_DATA__;
const DEFAULT_TOP_K = 20;
const state = { language: '', sampleId: '', step: 0 };
const $ = id => document.getElementById(id);
function esc(s){return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function samples(){return DATA.samples || [];}
function currentSample(){return samples().find(s => s.sample_id === state.sampleId) || samples()[0];}
function modelData(model){const s = currentSample(); return model === 'base' ? s.base : s.ours;}
function targetFor(model, step){const m = modelData(model); const idx = m.targets_by_step?.[String(step)]; return idx === undefined ? null : m.targets?.[String(idx)];}
function tokenText(t){
  const raw = String(t.text ?? t.display ?? '');
  return raw.replace(/\\n/g, '\n');
}
function displayToken(t){return t?.display ?? t?.text ?? t?.token ?? '';}
function salRows(model){const t = targetFor(model, state.step); const k = Number($('topKInput').value || DEFAULT_TOP_K); return (t?.sources || []).slice(0, k);}
function salSet(model){return new Set(salRows(model).map(r => Number(r.idx)));}
function annRows(){
  const s = currentSample();
  const ot = targetFor('ours', state.step);
  if (!ot) return [];
  const targetIdx = Number(ot.target_idx);
  return (s.annotation?.edges_by_dst?.[String(ot.target_idx)] || []).filter(e => Number(e.src) < targetIdx);
}
function annSet(){const rows = annRows(); const out = new Set(); rows.forEach(e => { out.add(Number(e.src)); out.add(Number(e.dst)); }); return out;}
function annSourceRows(){const seen = new Set(); const out = []; for (const e of annRows()) { const key = `${e.src}:${e.src_display || e.src_text || e.src}`; if (!seen.has(key)) { seen.add(key); out.push(e); } } return out;}
function annSourceSet(){return new Set(annSourceRows().map(e => Number(e.src)));}
function metricsFor(model){
  const t = targetFor(model, state.step);
  const k = Number($('topKInput').value || DEFAULT_TOP_K);
  if (!t) return null;
  if (t.metrics && Number(t.metrics.top_k) === k) return t.metrics;
  const ann = annSourceSet();
  const sourceRows = t.sources || [];
  const kEff = Math.min(Math.max(1, k), sourceRows.length);
  const rows = sourceRows.slice(0, kEff);
  let hits = 0, precisionSum = 0;
  rows.forEach((r, i) => {
    if (ann.has(Number(r.idx))) { hits += 1; precisionSum += hits / (i + 1); }
  });
  const denom = Math.max(1, ann.size);
  return {
    top_k: k,
    num_annotation_sources: ann.size,
    num_hits: hits,
    recall_at_k: hits / denom,
    precision_at_k: hits / Math.max(1, kEff),
    map_at_k: precisionSum / denom,
  };
}
function metricText(model){
  const m = metricsFor(model);
  if (!m) return 'recall@k / precision@k / mAP@k: no saliency';
  if (!m.num_annotation_sources) return `recall@${m.top_k}: n/a · precision@${m.top_k}: n/a · mAP@${m.top_k}: n/a · no annotation`;
  return `recall@${m.top_k}: ${Number(m.recall_at_k).toFixed(3)} · precision@${m.top_k}: ${Number(m.precision_at_k).toFixed(3)} · mAP@${m.top_k}: ${Number(m.map_at_k).toFixed(3)}`;
}
function fillControls(){
  const langs = [...new Set(samples().map(s => s.language))].sort();
  $('languageSelect').innerHTML = langs.map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join('');
  state.language = state.language || langs[0] || '';
  $('languageSelect').value = state.language;
  fillSamples();
}
function fillSamples(){
  const rows = samples().filter(s => s.language === state.language);
  $('sampleSelect').innerHTML = rows.map(s => `<option value="${esc(s.sample_id)}">${esc(s.source_dataset)} · row ${s.row_index} · ${esc(s.uid)}</option>`).join('');
  if (!rows.find(s => s.sample_id === state.sampleId)) state.sampleId = rows[0]?.sample_id || '';
  $('sampleSelect').value = state.sampleId;
  fillSteps();
}
function annCountForStep(st){
  const s = currentSample();
  const t = targetFor('ours', st);
  if (!t) return 0;
  return (s.annotation?.edges_by_dst?.[String(t.target_idx)] || []).length;
}
function fillSteps(){
  const s = currentSample();
  const steps = Object.keys(s.ours.targets_by_step || {}).map(Number).sort((a,b) => a-b);
  $('stepSelect').innerHTML = steps.map(st => {
    const t = targetFor('ours', st);
    const n = annCountForStep(st);
    const ann = n ? ` · ann ${n}` : '';
    return `<option value="${st}">step ${st} · #${t?.target_idx ?? ''} · ${esc(displayToken(t))}${ann}</option>`;
  }).join('');
  const currentValid = steps.includes(Number(state.step));
  const currentHasAnn = currentValid && annCountForStep(Number(state.step)) > 0;
  if (!currentValid || !currentHasAnn) state.step = steps.find(st => annCountForStep(st) > 0) ?? steps[0] ?? 0;
  $('stepSelect').value = String(state.step);
}
function tokenClass(t, model){
  const cls = ['tok', t.region];
  const tgt = targetFor(model, state.step);
  if (tgt && Number(t.idx) === Number(tgt.target_idx)) cls.push('target');
  if (salSet(model).has(Number(t.idx))) cls.push('sal');
  if (model === 'ours' && annSet().has(Number(t.idx))) cls.push('ann');
  if ($('hideSpecial').checked && t.is_special) cls.push('hidden');
  if ($('hideWhitespace').checked && t.is_whitespace) cls.push('hidden');
  return cls.join(' ');
}
function renderCode(model, el){
  const toks = modelData(model).tokens || [];
  const validSteps = new Set(Object.keys(modelData(model).targets_by_step || {}));
  el.innerHTML = toks.map(t => {
    const st = t.step == null ? '' : String(t.step);
    const isClickable = t.region === 'completion' && validSteps.has(st);
    const clickable = isClickable ? ` data-step="${st}"` : '';
    const cls = tokenClass(t, model) + (isClickable ? ' clickable-target' : '');
    return `<span class="${cls}" data-idx="${t.idx}"${clickable} title="#${t.idx} ${esc(t.region)} ${esc(displayToken(t))}">${esc(tokenText(t))}</span>`;
  }).join('');
  el.querySelectorAll('[data-step]').forEach(n => n.onclick = () => {
    const st = Number(n.dataset.step);
    if (!Number.isNaN(st)) {
      state.step = st;
      $('stepSelect').value = String(st);
      renderAll(false);
    }
  });
}
function salChips(model){
  const rows = salRows(model);
  const ann = annSourceSet();
  if (!rows.length) return '<span class="empty">No saliency rows</span>';
  return rows.map(r => {
    const cls = ann.has(Number(r.idx)) ? 'chip salchip annchip' : 'chip salchip';
    const tag = ann.has(Number(r.idx)) ? ' labeled' : '';
    return `<span class="${cls}" title="#${r.idx}${tag}">${esc(displayToken(r))}</span>`;
  }).join('');
}
function annChips(){
  const rows = annSourceRows();
  if (!rows.length) return '<span class="empty">No annotation edge for selected target</span>';
  return rows.slice(0, 16).map(e => `<span class="chip annchip">${esc(e.src_display || e.src_text || e.src)}</span>`).join('');
}
function renderSummaries(){
  $('baseSummary').innerHTML = `<div class="summary-row"><span class="summary-label">Base top-${$('topKInput').value}</span>${salChips('base')}</div><div class="summary-row"><span class="summary-label">Base Metrics</span><span class="chip">${esc(metricText('base'))}</span></div>`;
  $('oursSummary').innerHTML = `<div class="summary-row"><span class="summary-label">Ours top-${$('topKInput').value}</span>${salChips('ours')}</div><div class="summary-row"><span class="summary-label">SFT Metrics</span><span class="chip">${esc(metricText('ours'))}</span></div><div class="summary-row"><span class="summary-label">Annotation</span>${annChips()}</div>`;
}
function renderDetails(){
  const bt = targetFor('base', state.step), ot = targetFor('ours', state.step);
  $('baseTitle').textContent = modelData('base').name || 'Base Qwen';
  $('oursTitle').textContent = modelData('ours').name || 'SFT Model';
  $('baseSub').textContent = bt ? `step ${state.step}; target #${bt.target_idx}: ${displayToken(bt)}; ${metricText('base')}` : 'No Base target for this step';
  $('oursSub').textContent = ot ? `step ${state.step}; target #${ot.target_idx}: ${displayToken(ot)}; annotation sources: ${annSourceSet().size}; ${metricText('ours')}` : 'No SFT target for this step';
}
function renderAll(reset=true){
  if (reset) fillSteps();
  renderCode('base', $('baseCode'));
  renderCode('ours', $('oursCode'));
  renderDetails();
  renderSummaries();
}
$('languageSelect').onchange = () => { state.language = $('languageSelect').value; fillSamples(); renderAll(false); };
$('sampleSelect').onchange = () => { state.sampleId = $('sampleSelect').value; renderAll(true); };
$('stepSelect').onchange = () => { state.step = Number($('stepSelect').value); renderAll(false); };
$('topKInput').onchange = () => renderAll(false);
$('hideSpecial').onchange = () => renderAll(false);
$('hideWhitespace').onchange = () => renderAll(false);
$('topKInput').value = String(Math.max(DEFAULT_TOP_K, Number(DATA.top_k || 0) || DEFAULT_TOP_K));
fillControls();
renderAll(true);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--output_path", default=str(DEFAULT_HTML))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    html = HTML_TEMPLATE.replace("__VIEWER_DATA__", json.dumps(data, ensure_ascii=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
