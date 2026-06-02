#!/usr/bin/env python
"""Build a v2 Ours-vs-CLEAR failure viewer with optional Ours saliency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_FAILURE_DATA = "outputs/visual_failure/ours_vs_clear_humaneval1000.json"
DEFAULT_SALIENCY_DATA = "outputs/visual_failure/ours_vs_clear_humaneval1000_ours_saliency.json"
DEFAULT_OUTPUT = "outputs/visual_failure/ours_vs_clear_failure_viewer_v2.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=DEFAULT_FAILURE_DATA)
    parser.add_argument("--saliency_path", default=DEFAULT_SALIENCY_DATA)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    return parser.parse_args()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ours vs CLEAR 测试集优劣案例对比</title>
<style>
:root{--blue:#3158dc;--line:#d8dce8;--ink:#111827;--muted:#697386;--bg:#f6f7fb;--panel:#fff;--code:#0f1424;--pass:#16a34a;--fail:#dc2626;--target:#f6c642;--orange:#ff7a1a;--green:#22c55e;}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:auto minmax(0,1fr);background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif}
header{height:58px;border-bottom:1px solid var(--line);background:#fff;display:flex;align-items:center;padding:0 18px;gap:16px}
h1{font-size:20px;margin:0;color:var(--blue);font-weight:800}.sub{color:var(--muted);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.layout{min-height:0;display:grid;grid-template-columns:330px minmax(520px,1fr) 430px}.side,.right{background:#fff;overflow-y:auto;overflow-x:hidden}.side{border-right:1px solid var(--line)}.right{border-left:1px solid var(--line)}.main{overflow:auto;background:#0d1322;color:#e5edff;position:relative}
.section{padding:16px;border-bottom:1px solid var(--line)}.label{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:800;margin-bottom:8px}select,input{width:100%;height:42px;border:1px solid var(--line);border-radius:7px;padding:0 10px;background:#fff;font-size:14px}.search{margin-top:10px}
.sample{padding:11px 16px;border-bottom:1px solid var(--line);cursor:pointer}.sample:hover{background:#f1f4ff}.sample.active{background:#4164e8;color:white}.sample .top{font-weight:800;display:flex;justify-content:space-between;gap:8px}.sample .meta{font-size:12px;color:var(--muted);margin-top:4px}.sample.active .meta{color:#eaf0ff}.badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.badge{font-size:11px;border-radius:999px;padding:3px 7px;background:#eef2ff;color:#3158dc}.sample.active .badge{background:#dbe5ff;color:#17328f}
.tabs{position:sticky;top:0;background:#111827;border-bottom:1px solid #253047;padding:10px 14px;display:flex;gap:8px;z-index:2;align-items:center;flex-wrap:wrap}.btn{border:1px solid #3b4661;background:#1c263a;color:#d9e5ff;border-radius:6px;padding:7px 10px;cursor:pointer;font-weight:700}.btn.active{background:#f5b83d;color:#111827;border-color:#f5b83d}.btn.small{font-size:12px;padding:5px 8px}.model-toggle{margin-left:auto;display:flex;gap:8px}.inline-control{width:auto;height:34px;background:#162036;color:#d9e5ff;border-color:#3b4661;font-weight:700}.content{padding:18px;max-width:1180px;margin:0 auto}.card{border:1px solid #26324a;background:#111827;border-radius:8px;margin-bottom:16px;overflow:hidden}.card h2{font-size:13px;letter-spacing:.11em;text-transform:uppercase;margin:0;padding:10px 12px;border-bottom:1px solid #26324a;color:#93a4c8}.code{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.55;padding:14px;overflow:auto;max-height:420px;color:#e6efff;background:#0f1424}.code.gt{border-left:4px solid var(--pass)}.code.pred.pass{border-left:4px solid var(--pass);background:#0f1424;color:#e6efff}.code.pred.fail{border-left:4px solid var(--fail);background:#0f1424;color:#e6efff}.code.full{max-height:none}
.tok{display:inline;border-radius:3px;padding:0 1px}.tok.clickable{cursor:pointer}.tok.clickable:hover{background:#23314e}.tok.sal-target{background:var(--target);color:#1f2937;padding:0 3px}.tok.sal-src{border:1.5px solid var(--orange);color:#fff;padding:0 3px}.tok.prompt_code{color:#cfd8ee}.tok.prompt_other{color:#93a4c8}.tok.completion{color:#d9ffe9}
.summary{display:block;padding:14px}.metric{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff;margin-bottom:12px;min-width:0}.metric h3{margin:0 0 8px;font-size:15px}.pill{display:inline-flex;align-items:center;justify-content:center;min-width:48px;border-radius:999px;padding:3px 7px;font-size:11px;font-weight:900}.pass{background:#dcfce7;color:#166534}.fail{background:#fee2e2;color:#991b1b}.unknown{background:#f1f5f9;color:#475569}.statline{font-size:12px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.empty{padding:20px;color:var(--muted)}.right h2{font-size:18px;line-height:1.22;overflow-wrap:anywhere}.scorebar{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0 12px}.scorebox{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff}.scorebox h3{margin:0 0 6px;font-size:14px}.compare{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;margin-bottom:12px}.compare-head,.compare-row{display:grid;grid-template-columns:72px 1fr 1fr;align-items:center}.compare-head{background:#f8fafc;color:var(--muted);font-size:11px;letter-spacing:.11em;text-transform:uppercase;font-weight:900}.compare-head div,.compare-row div,.compare-row button{padding:8px}.compare-row{border-top:1px solid var(--line)}.compare-row.active{background:#f5f7ff}.cand-label{font-weight:800;font-size:12px;color:#111827}.result{border:0;background:transparent;text-align:left;cursor:pointer;width:100%;height:100%;font:inherit}.result:hover,.result.active{background:#eef2ff}.result.active .pill{outline:2px solid var(--blue);outline-offset:2px}
.sal-box{border:1px solid var(--line);border-radius:8px;background:#fff;margin-bottom:12px;overflow:hidden}.sal-head{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px;align-items:center}.sal-head h3{margin:0;font-size:14px}.sal-body{padding:10px 12px}.note{font-size:12px;line-height:1.45;color:var(--muted)}.lens{border-top:1px solid var(--line);padding:10px 12px;font-size:12px;line-height:1.45}.lens b{color:#111827}
@media(max-width:1180px){html,body{overflow:auto}body{display:block}.layout{display:block}.side,.right,.main{height:auto;max-height:none}.side,.right{border:0}.main{min-height:680px}.model-toggle{margin-left:0}}
</style>
</head>
<body>
<header><h1>Ours vs CLEAR 测试集优劣案例对比</h1><div class="sub" id="headerSub"></div></header>
<div class="layout">
<aside class="side"><div class="section"><div class="label">Failure Type</div><select id="categorySelect"></select><input class="search" id="searchBox" placeholder="search uid / raw_id / token text" /></div><div id="sampleList"></div></aside>
<main class="main"><div class="tabs"><button class="btn active" data-view="completion">Completion</button><button class="btn" data-view="full">Full code</button><button class="btn" data-panel="prefix">Prefix</button><button class="btn" data-panel="suffix">Suffix</button><button class="btn active" data-panel="gt">GT</button><select class="inline-control" id="scopeSelect"><option value="all_causal">all causal</option><option value="prompt_code">prompt code</option><option value="prompt_all">prompt all</option></select><div class="model-toggle"><button class="btn small active" data-model="ours">Ours</button><button class="btn small" data-model="clear">CLEAR</button></div></div><div class="content" id="mainContent"></div></main>
<aside class="right"><div class="summary" id="rightPanel"></div></aside>
</div>
<script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
<script id="saliency-data" type="application/json">__SALIENCY_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('viewer-data').textContent);
const SALIENCY = JSON.parse(document.getElementById('saliency-data').textContent);
const $ = sel => document.querySelector(sel);
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let category = 'clear_pass10_ours_fail10', selectedKey = null, currentModel = 'ours', currentCandidate = 'greedy', viewMode = 'completion';
let showPanels = new Set(['gt']);
let selectedTargetBySample = {};
function passClass(v){ return v === true ? 'pass' : (v === false ? 'fail' : 'unknown'); }
function passText(v){ return v === true ? 'PASS' : (v === false ? 'FAIL' : 'N/A'); }
function currentSample(){ return DATA.samples.find(s => s.key === selectedKey) || DATA.samples[0]; }
function filteredSamples(){ const q=$('#searchBox').value.trim().toLowerCase(); return DATA.samples.filter(s => s.categories.includes(category)).filter(s => !q || JSON.stringify([s.uid,s.raw_id,s.language,s.prefix,s.ground_truth]).toLowerCase().includes(q)); }
function renderCategories(){ const sel=$('#categorySelect'); sel.innerHTML=''; Object.entries(DATA.categories).forEach(([key,label])=>{ const n=DATA.category_counts[key]||0; const opt=document.createElement('option'); opt.value=key; opt.textContent=`${label} (${n})`; sel.appendChild(opt); }); sel.value=category; }
function renderList(){ const list=$('#sampleList'); const rows=filteredSamples(); if(!rows.length){ list.innerHTML='<div class="empty">No samples.</div>'; selectedKey=null; renderAllPanels(); return; } if(!selectedKey || !rows.some(s=>s.key===selectedKey)) selectedKey=rows[0].key; list.innerHTML=rows.map(s=>`<div class="sample ${s.key===selectedKey?'active':''}" data-key="${esc(s.key)}"><div class="top"><span>#${esc(s.filtered_index)}</span><span>${esc(s.language)}</span></div><div class="meta">${esc(s.uid || s.raw_id)} · ${esc(s.entry_point||'')}</div><div class="badges">${s.categories.slice(0,3).map(c=>`<span class="badge">${esc(DATA.categories[c]||c)}</span>`).join('')}</div></div>`).join(''); document.querySelectorAll('.sample').forEach(el=>el.onclick=()=>{selectedKey=el.dataset.key; currentCandidate='greedy'; renderAll();}); }
function candidateByLabel(model,label){ if(label==='greedy') return {label:'greedy', ...model.greedy}; const idx=Number(label.replace('cand ',''))-1; return {label, ...(model.samples[idx]||{})}; }
function codeFor(s,cand){ return viewMode==='full' ? `${s.prefix}${cand.prediction||''}${s.suffix}` : cand.prediction||''; }
function saliencySample(s){
  const rows = (SALIENCY && SALIENCY.samples) || [];
  return rows.find(x => String(x.uid||'') === String(s.uid||'')) ||
         rows.find(x => String(x.raw_id||'') === String(s.raw_id||'')) ||
         rows.find(x => Number(x.row_index) === Number(s.filtered_index));
}
function oursSaliencyResult(s){
  const row = saliencySample(s);
  if(!row) return null;
  const models = row.models || {};
  const key = Object.keys(models).find(k => k.toLowerCase().includes('ours')) || Object.keys(models)[0];
  return key ? models[key] : null;
}
function targetIds(res){ return Object.keys(res?.targets || {}).map(Number).sort((a,b)=>a-b); }
function selectedTargetId(s,res){
  const ids = targetIds(res);
  if(!ids.length) return null;
  const current = selectedTargetBySample[s.key];
  return ids.includes(Number(current)) ? Number(current) : ids[0];
}
function sourceRowsForTarget(t){
  const scope = $('#scopeSelect')?.value || 'all_causal';
  return (t?.scopes?.[scope] || t?.scopes?.all_causal || []).slice(0,5);
}
function sourceSetForTarget(t){ return new Set(sourceRowsForTarget(t).map(r => Number(r.idx))); }
function tokenText(t){ return String(t.text ?? t.display ?? '').replace(/\\n/g, '\n').replace(/\\t/g, '\t'); }
function renderGeneratedTokens(s,res,targetId){
  const target = res?.targets?.[String(targetId)];
  const srcSet = sourceSetForTarget(target);
  const ids = new Set(res?.generated_token_indices || []);
  return (res?.tokens || []).filter(t => ids.has(Number(t.idx))).map(t => {
    const cls = `${Number(t.idx)===Number(targetId)?'sal-target ':''}${srcSet.has(Number(t.idx))?'sal-src ':''}clickable`;
    return `<span class="tok completion ${cls}" data-target-idx="${esc(t.idx)}" title="#${esc(t.idx)} ${esc(t.display||t.text||'')}">${esc(tokenText(t))}</span>`;
  }).join('');
}
function canShowSaliency(){ return currentModel === 'ours' && currentCandidate === 'greedy'; }
function renderMain(){
  const s=currentSample(); if(!s){ $('#mainContent').innerHTML='<div class="empty">No sample selected.</div>'; return; }
  const model=s[currentModel]; const cand=candidateByLabel(model,currentCandidate);
  const res = oursSaliencyResult(s); const tid = selectedTargetId(s,res);
  const canAttachSaliency = canShowSaliency() && res && !res.skipped && tid !== null;
  const useSal = canAttachSaliency && viewMode==='completion';
  const predBody = useSal ? renderGeneratedTokens(s,res,tid) : esc(codeFor(s,cand));
  const salBadge = canShowSaliency()
    ? (canAttachSaliency ? '<span class="pill pass">SALIENCY READY</span>' : '<span class="pill unknown">NO SALIENCY</span>')
    : '';
  let html=`<div class="card"><h2>${esc(currentModel.toUpperCase())} · ${esc(cand.label)} · <span class="${passClass(cand.pass)} pill">${passText(cand.pass)}</span> ${salBadge}</h2><pre class="code pred ${passClass(cand.pass)} ${viewMode==='full'?'full':''}">${predBody}</pre></div>`;
  if(showPanels.has('gt')) html+=`<div class="card"><h2>Ground Truth Completion</h2><pre class="code gt">${esc(s.ground_truth)}</pre></div>`;
  if(showPanels.has('prefix')) html+=`<div class="card"><h2>FIM Prefix</h2><pre class="code">${esc(s.prefix)}</pre></div>`;
  if(showPanels.has('suffix')) html+=`<div class="card"><h2>FIM Suffix</h2><pre class="code">${esc(s.suffix)}</pre></div>`;
  $('#mainContent').innerHTML=html;
  document.querySelectorAll('[data-target-idx]').forEach(el=>el.onclick=()=>{ selectedTargetBySample[s.key]=Number(el.dataset.targetIdx); renderAllPanels(); });
}
function candidateLabels(s){ const n=Math.max((s.ours.samples||[]).length,(s.clear.samples||[]).length); const labels=['greedy']; for(let i=1;i<=n;i++) labels.push('cand '+i); return labels; }
function scoreBox(name, model){ return `<div class="scorebox"><h3>${esc(name)}</h3><span class="pill ${passClass(model.pass1)}">P@1 ${passText(model.pass1)}</span> <span class="pill ${passClass(model.pass10)}">P@10 ${passText(model.pass10)}</span></div>`; }
function resultCell(modelKey,label){ const s=currentSample(); const cand=candidateByLabel(s[modelKey],label); const active=currentModel===modelKey && currentCandidate===label; return `<button class="result ${active?'active':''}" data-modelbtn="${modelKey}" data-cand="${esc(label)}"><span class="pill ${passClass(cand.pass)}">${passText(cand.pass)}</span></button>`; }
function comparisonTable(s){ return `<div class="compare"><div class="compare-head"><div>cand</div><div>Ours</div><div>CLEAR</div></div>${candidateLabels(s).map(label=>`<div class="compare-row ${currentCandidate===label?'active':''}"><div class="cand-label">${esc(label)}</div><div>${resultCell('ours',label)}</div><div>${resultCell('clear',label)}</div></div>`).join('')}</div>`; }
function saliencyStatus(s){
  const res = oursSaliencyResult(s);
  if(!SALIENCY) return 'No saliency file embedded yet.';
  if(!res) return 'No matched saliency row for this sample.';
  if(res.skipped) return `Skipped: ${res.reason||'unknown reason'}`;
  return `${targetIds(res).length} generated tokens with ALTI top-5.`;
}
function diagnosisLens(s){
  if(s.categories.includes('ours_pass10_clear_fail10') || s.categories.includes('ours_pass1_clear_fail1')) return '<b>1) Ours exceeds CLEAR.</b> Check whether top-5 sources point to semantic anchors such as variables, loop state, branch predicates, API names, or matching delimiters. If yes, this is evidence that saliency moved in a useful direction.';
  if(s.categories.includes('clear_pass10_ours_fail10') || s.categories.includes('clear_pass1_ours_fail1')) return '<b>2) Ours does not exceed CLEAR.</b> If saliency still points to meaningful anchors, the likely issue is not attention alignment alone: inspect over-generation, missing stop behavior, wrong span boundary, or judge-sensitive formatting.';
  if(s.categories.includes('both_fail10')) return '<b>3) Both fail.</b> Use saliency to separate causes: wrong anchors suggest better graph/annotation coverage; good anchors but bad code suggest decoding, target-length, or training objective gaps.';
  return '<b>General lens.</b> Compare saliency anchors with the ground truth logic and the candidate table to decide whether the failure is retrieval of the right context, generation stability, or benchmark/judge brittleness.';
}
function renderRight(){
  const s=currentSample(); if(!s){ $('#rightPanel').innerHTML=''; return; }
  $('#rightPanel').innerHTML=`<div><h2 style="margin:0;color:var(--blue)">#${esc(s.filtered_index)} · ${esc(s.language)} · ${esc(s.uid||s.raw_id)}</h2><div class="statline">${s.categories.map(c=>DATA.categories[c]||c).join(' · ')}</div></div><div class="scorebar">${scoreBox('Ours',s.ours)}${scoreBox('CLEAR',s.clear)}</div>${comparisonTable(s)}<div class="sal-box"><div class="sal-head"><h3>Saliency status</h3><span class="pill ${oursSaliencyResult(s)?'pass':'unknown'}">${oursSaliencyResult(s)?'READY':'MISSING'}</span></div><div class="sal-body note">${esc(saliencyStatus(s))}</div><div class="lens">${diagnosisLens(s)}</div></div>`;
  document.querySelectorAll('.result[data-modelbtn]').forEach(btn=>btn.onclick=()=>{ currentModel=btn.dataset.modelbtn; currentCandidate=btn.dataset.cand; document.querySelectorAll('[data-model]').forEach(b=>b.classList.toggle('active',b.dataset.model===currentModel)); renderAllPanels(); document.querySelector('.main').scrollTop=0; });
}
function renderAllPanels(){ renderMain(); renderRight(); }
function renderAll(){ renderList(); renderAllPanels(); }
renderCategories(); $('#headerSub').textContent='分析我们的模型在测试集上的泛化性';
$('#categorySelect').onchange=e=>{category=e.target.value; selectedKey=null; currentCandidate='greedy'; renderAll();};
$('#searchBox').oninput=()=>{selectedKey=null; renderAll();};
$('#scopeSelect').onchange=()=>renderAllPanels();
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{viewMode=b.dataset.view; document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b)); renderMain();});
document.querySelectorAll('[data-model]').forEach(b=>b.onclick=()=>{currentModel=b.dataset.model; document.querySelectorAll('[data-model]').forEach(x=>x.classList.toggle('active',x===b)); renderAllPanels();});
document.querySelectorAll('[data-panel]').forEach(b=>b.onclick=()=>{const p=b.dataset.panel; if(showPanels.has(p)) showPanels.delete(p); else showPanels.add(p); b.classList.toggle('active',showPanels.has(p)); renderMain();});
renderAll();
</script>
</body>
</html>
"""


def script_safe_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</script", "<\\/script")


def main() -> None:
    args = parse_args()
    failure_data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    saliency_path = Path(args.saliency_path)
    saliency_data = json.loads(saliency_path.read_text(encoding="utf-8")) if saliency_path.exists() else None
    html = (
        HTML.replace("__VIEWER_DATA__", script_safe_json(failure_data))
        .replace("__SALIENCY_DATA__", script_safe_json(saliency_data))
    )
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    if saliency_data is None:
        print(f"note: saliency file not found: {saliency_path}")


if __name__ == "__main__":
    main()
