#!/usr/bin/env python
"""Build a standalone HTML viewer for Ours-vs-CLEAR failure analysis."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="outputs/visual_failure/ours_vs_clear_humaneval1000.json")
    p.add_argument("--output_path", default="outputs/visual_failure/ours_vs_clear_failure_viewer.html")
    return p.parse_args()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ours vs CLEAR Failure Viewer</title>
<style>
:root{--blue:#3158dc;--line:#d8dce8;--ink:#111827;--muted:#697386;--bg:#f6f7fb;--panel:#fff;--code:#0f1424;--pass:#16a34a;--fail:#dc2626;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif;height:100vh;overflow:hidden}
header{height:58px;border-bottom:1px solid var(--line);background:#fff;display:flex;align-items:center;padding:0 18px;gap:16px} h1{font-size:20px;margin:0;color:var(--blue);font-weight:800}.sub{color:var(--muted);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.layout{height:calc(100vh - 58px);display:grid;grid-template-columns:330px 1fr 390px;min-width:1120px}.side,.right{background:#fff;overflow-y:auto;overflow-x:hidden}.side{border-right:1px solid var(--line)}.right{border-left:1px solid var(--line)}.main{overflow:auto;background:#0d1322;color:#e5edff;position:relative}
.section{padding:16px;border-bottom:1px solid var(--line)}.label{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:800;margin-bottom:8px}select,input{width:100%;height:42px;border:1px solid var(--line);border-radius:7px;padding:0 10px;background:#fff;font-size:14px}.search{margin-top:10px}
.sample{padding:11px 16px;border-bottom:1px solid var(--line);cursor:pointer}.sample:hover{background:#f1f4ff}.sample.active{background:#4164e8;color:white}.sample .top{font-weight:800;display:flex;justify-content:space-between;gap:8px}.sample .meta{font-size:12px;color:var(--muted);margin-top:4px}.sample.active .meta{color:#eaf0ff}.badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.badge{font-size:11px;border-radius:999px;padding:3px 7px;background:#eef2ff;color:#3158dc}.sample.active .badge{background:#dbe5ff;color:#17328f}
.tabs{position:sticky;top:0;background:#111827;border-bottom:1px solid #253047;padding:10px 14px;display:flex;gap:8px;z-index:2;align-items:center;flex-wrap:wrap}.btn{border:1px solid #3b4661;background:#1c263a;color:#d9e5ff;border-radius:6px;padding:7px 10px;cursor:pointer;font-weight:700}.btn.active{background:#f5b83d;color:#111827;border-color:#f5b83d}.btn.small{font-size:12px;padding:5px 8px}.model-toggle{margin-left:auto;display:flex;gap:8px}.content{padding:18px;max-width:1100px;margin:0 auto}.card{border:1px solid #26324a;background:#111827;border-radius:8px;margin-bottom:16px;overflow:hidden}.card h2{font-size:13px;letter-spacing:.11em;text-transform:uppercase;margin:0;padding:10px 12px;border-bottom:1px solid #26324a;color:#93a4c8}.code{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.55;padding:14px;overflow:auto;max-height:360px;color:#e6efff}.code.gt{border-left:4px solid var(--pass)}.code.pred.pass{border-left:4px solid var(--pass)}.code.pred.fail{border-left:4px solid var(--fail)}.code.full{max-height:none}
.summary{display:block;padding:14px}.metric{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff;margin-bottom:12px;min-width:0}.metric h3{margin:0 0 8px;font-size:15px}.pill{display:inline-flex;align-items:center;justify-content:center;min-width:48px;border-radius:999px;padding:3px 7px;font-size:11px;font-weight:900}.pass{background:#dcfce7;color:#166534}.fail{background:#fee2e2;color:#991b1b}.unknown{background:#f1f5f9;color:#475569}.cand{display:grid;grid-template-columns:72px minmax(0,1fr);align-items:center;border-top:1px solid var(--line);padding:7px 0;gap:8px;cursor:pointer}.cand:first-of-type{border-top:0}.cand.active{background:#f6f8ff;margin-left:-6px;margin-right:-6px;padding-left:6px;padding-right:6px;border-radius:6px}.cand button{height:28px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer;font-weight:700;font-size:12px}.cand button:hover,.cand.active button{border-color:var(--blue);color:var(--blue)}.detail{font-size:11px;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}.statline{font-size:12px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.empty{padding:24px;color:var(--muted)}.right h2{font-size:18px;line-height:1.22;overflow-wrap:anywhere}.scorebar{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0 12px}.scorebox{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff}.scorebox h3{margin:0 0 6px;font-size:14px}.compare{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}.compare-head,.compare-row{display:grid;grid-template-columns:72px 1fr 1fr;align-items:center}.compare-head{background:#f8fafc;color:var(--muted);font-size:11px;letter-spacing:.11em;text-transform:uppercase;font-weight:900}.compare-head div,.compare-row div,.compare-row button{padding:8px}.compare-row{border-top:1px solid var(--line)}.compare-row.active{background:#f5f7ff}.cand-label{font-weight:800;font-size:12px;color:#111827}.result{border:0;background:transparent;text-align:left;cursor:pointer;width:100%;height:100%;font:inherit}.result:hover,.result.active{background:#eef2ff}.result.active .pill{outline:2px solid var(--blue);outline-offset:2px}
</style>
</head>
<body>
<header><h1>Ours vs CLEAR Failure Viewer</h1><div class="sub" id="headerSub"></div></header>
<div class="layout"><aside class="side"><div class="section"><div class="label">Failure Type</div><select id="categorySelect"></select><input class="search" id="searchBox" placeholder="search uid / raw_id / token text" /></div><div id="sampleList"></div></aside><main class="main"><div class="tabs"><button class="btn active" data-view="completion">Completion</button><button class="btn" data-view="full">Full code</button><button class="btn" data-panel="prefix">Prefix</button><button class="btn" data-panel="suffix">Suffix</button><button class="btn active" data-panel="gt">GT</button><div class="model-toggle"><button class="btn small active" data-model="ours">Ours</button><button class="btn small" data-model="clear">CLEAR</button></div></div><div class="content" id="mainContent"></div></main><aside class="right"><div class="summary" id="rightPanel"></div></aside></div>
<script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('viewer-data').textContent);
const $ = sel => document.querySelector(sel);
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let category = 'clear_pass10_ours_fail10', selectedKey = null, currentModel = 'ours', currentCandidate = 'greedy', viewMode = 'completion';
let showPanels = new Set(['gt']);
function passClass(v){ return v === true ? 'pass' : (v === false ? 'fail' : 'unknown'); }
function passText(v){ return v === true ? 'PASS' : (v === false ? 'FAIL' : 'N/A'); }
function sampleCandidates(model){ const arr=[{kind:'greedy', label:'greedy', ...model.greedy}]; (model.samples||[]).forEach((s,i)=>arr.push({kind:'sample', label:'cand '+(i+1), ...s})); return arr; }
function currentSample(){ return DATA.samples.find(s => s.key === selectedKey) || DATA.samples[0]; }
function filteredSamples(){ const q=$('#searchBox').value.trim().toLowerCase(); return DATA.samples.filter(s => s.categories.includes(category)).filter(s => !q || JSON.stringify([s.uid,s.raw_id,s.language,s.prefix,s.ground_truth]).toLowerCase().includes(q)); }
function renderCategories(){ const sel=$('#categorySelect'); sel.innerHTML=''; Object.entries(DATA.categories).forEach(([key,label])=>{ const n=DATA.category_counts[key]||0; const opt=document.createElement('option'); opt.value=key; opt.textContent=`${label} (${n})`; sel.appendChild(opt); }); sel.value=category; }
function renderList(){ const list=$('#sampleList'); const rows=filteredSamples(); if(!rows.length){ list.innerHTML='<div class="empty">No samples.</div>'; selectedKey=null; renderAllPanels(); return; } if(!selectedKey || !rows.some(s=>s.key===selectedKey)) selectedKey=rows[0].key; list.innerHTML=rows.map(s=>`<div class="sample ${s.key===selectedKey?'active':''}" data-key="${esc(s.key)}"><div class="top"><span>#${esc(s.filtered_index)}</span><span>${esc(s.language)}</span></div><div class="meta">${esc(s.uid || s.raw_id)} · ${esc(s.entry_point||'')}</div><div class="badges">${s.categories.slice(0,3).map(c=>`<span class="badge">${esc(DATA.categories[c]||c)}</span>`).join('')}</div></div>`).join(''); document.querySelectorAll('.sample').forEach(el=>el.onclick=()=>{selectedKey=el.dataset.key; currentCandidate='greedy'; renderAll();}); }
function candidateByLabel(model,label){ if(label==='greedy') return {label:'greedy', ...model.greedy}; const idx=Number(label.replace('cand ',''))-1; return {label, ...(model.samples[idx]||{})}; }
function codeFor(s,cand){ return viewMode==='full' ? `${s.prefix}${cand.prediction||''}${s.suffix}` : cand.prediction||''; }
function renderMain(){ const s=currentSample(); if(!s){ $('#mainContent').innerHTML='<div class="empty">No sample selected.</div>'; return; } const model=s[currentModel]; const cand=candidateByLabel(model,currentCandidate); let html=`<div class="card"><h2>${esc(currentModel.toUpperCase())} · ${esc(cand.label)} · <span class="${passClass(cand.pass)} pill">${passText(cand.pass)}</span></h2><pre class="code pred ${passClass(cand.pass)} ${viewMode==='full'?'full':''}">${esc(codeFor(s,cand))}</pre></div>`; if(showPanels.has('gt')) html+=`<div class="card"><h2>Ground Truth Completion</h2><pre class="code gt">${esc(s.ground_truth)}</pre></div>`; if(showPanels.has('prefix')) html+=`<div class="card"><h2>FIM Prefix</h2><pre class="code">${esc(s.prefix)}</pre></div>`; if(showPanels.has('suffix')) html+=`<div class="card"><h2>FIM Suffix</h2><pre class="code">${esc(s.suffix)}</pre></div>`; $('#mainContent').innerHTML=html; }
function candidateLabels(s){
  const n = Math.max((s.ours.samples||[]).length, (s.clear.samples||[]).length);
  const labels = ['greedy'];
  for(let i=1;i<=n;i++) labels.push('cand '+i);
  return labels;
}
function scoreBox(name, model){
  return `<div class="scorebox"><h3>${esc(name)}</h3><span class="pill ${passClass(model.pass1)}">P@1 ${passText(model.pass1)}</span> <span class="pill ${passClass(model.pass10)}">P@10 ${passText(model.pass10)}</span></div>`;
}
function resultCell(modelKey, label){
  const s=currentSample();
  const cand=candidateByLabel(s[modelKey], label);
  const active=currentModel===modelKey && currentCandidate===label;
  return `<button class="result ${active?'active':''}" data-modelbtn="${modelKey}" data-cand="${esc(label)}"><span class="pill ${passClass(cand.pass)}">${passText(cand.pass)}</span></button>`;
}
function comparisonTable(s){
  return `<div class="compare"><div class="compare-head"><div>cand</div><div>Ours</div><div>CLEAR</div></div>${candidateLabels(s).map(label=>`<div class="compare-row ${currentCandidate===label?'active':''}"><div class="cand-label">${esc(label)}</div><div>${resultCell('ours', label)}</div><div>${resultCell('clear', label)}</div></div>`).join('')}</div>`;
}
function renderRight(){
  const s=currentSample();
  if(!s){ $('#rightPanel').innerHTML=''; return; }
  $('#rightPanel').innerHTML=`<div><h2 style="margin:0;color:var(--blue)">#${esc(s.filtered_index)} · ${esc(s.language)} · ${esc(s.uid||s.raw_id)}</h2><div class="statline">${s.categories.map(c=>DATA.categories[c]||c).join(' · ')}</div></div><div class="scorebar">${scoreBox('Ours',s.ours)}${scoreBox('CLEAR',s.clear)}</div>${comparisonTable(s)}`;
  document.querySelectorAll('.result[data-modelbtn]').forEach(btn=>btn.onclick=()=>{
    currentModel=btn.dataset.modelbtn;
    currentCandidate=btn.dataset.cand;
    document.querySelectorAll('[data-model]').forEach(b=>b.classList.toggle('active',b.dataset.model===currentModel));
    renderAllPanels();
    document.querySelector('.main').scrollTop = 0;
  });
}
function renderAllPanels(){ renderMain(); renderRight(); } function renderAll(){ renderList(); renderAllPanels(); }
renderCategories(); $('#headerSub').textContent=`${DATA.num_common} samples · Ours GraphSignal vs CLEAR`;
$('#categorySelect').onchange=e=>{category=e.target.value; selectedKey=null; currentCandidate='greedy'; renderAll();}; $('#searchBox').oninput=()=>{selectedKey=null; renderAll();};
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{viewMode=b.dataset.view; document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b)); renderMain();});
document.querySelectorAll('[data-model]').forEach(b=>b.onclick=()=>{currentModel=b.dataset.model; document.querySelectorAll('[data-model]').forEach(x=>x.classList.toggle('active',x===b)); renderAllPanels();});
document.querySelectorAll('[data-panel]').forEach(b=>b.onclick=()=>{const p=b.dataset.panel; if(showPanels.has(p)) showPanels.delete(p); else showPanels.add(p); b.classList.toggle('active',showPanels.has(p)); renderMain();});
renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = Path(args.data_path).read_text(encoding='utf-8')
    html = HTML.replace('__VIEWER_DATA__', data.replace('</script', '<\\/script'))
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
