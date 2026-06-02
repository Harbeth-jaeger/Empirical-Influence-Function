import { useMemo, useState } from 'react';
// Use import.meta.glob to gracefully handle missing files without crashing Vite
const saliencyFiles = import.meta.glob([
    '../../../latest_saliency.json',
    '../../../runs/attribution/latest_saliency.json',
], { query: '?raw', eager: true }) as Record<string, any>;
const latestSaliencyJSONText = saliencyFiles['../../../runs/attribution/latest_saliency.json']?.default
    || saliencyFiles['../../../latest_saliency.json']?.default
    || "{}";

const markedCodeFiles = import.meta.glob([
    '../../../marked_code_samples.md',
    '../../../runs/attribution/marked_code_samples.md',
], { query: '?raw', eager: true }) as Record<string, any>;
const markedCodeSamplesText = markedCodeFiles['../../../runs/attribution/marked_code_samples.md']?.default
    || markedCodeFiles['../../../marked_code_samples.md']?.default
    || "";

import './App.css';
import { SwitchTokenCodeBlock } from './components/SwitchTokenCodeBlock';

/* ------------------------------------------------------------------ */
/* Data loading                                                       */
/* ------------------------------------------------------------------ */

let allSaliencies: any = {};
try {
    allSaliencies = JSON.parse(latestSaliencyJSONText);
} catch (e) {
    allSaliencies = {};
}
// New format has 'overfit_test_results'; old format stores overfit data under 'related_train_samples'
const hasNewFormat = 'overfit_test_results' in allSaliencies;
const overfitResults = hasNewFormat
    ? (allSaliencies['overfit_test_results'] ?? [])
    : (allSaliencies['related_train_samples'] ?? []);
const BOOST_COEFS = [1, 10, 100, 1000, 10000];

// Section 2: real related train samples (only present in new JSON format)
const trainSamples: any[] = hasNewFormat
    ? (allSaliencies['related_train_samples'] ?? [])
    : [];

/** Extract fenced code blocks from markdown */
function extractFencedCodeBlocks(text: string): string[] {
    const re = /```[^\n]*\n(.*?)\n```/gs;
    const blocks: string[] = [];
    let m;
    while ((m = re.exec(text)) !== null) blocks.push(m[1]);
    return blocks;
}

/** Parse <ATTN>…</ATTN> → cleaned text + char-level boolean array */
function extractAttnSpans(text: string): { cleanedText: string; spans: [number, number][] } {
    const TAG_RE = /<ATTN>(.*?)<\/ATTN>/gs;
    const spans: [number, number][] = [];
    const parts: string[] = [];
    let cursor = 0, outLen = 0, m;
    while ((m = TAG_RE.exec(text)) !== null) {
        const pre = text.slice(cursor, m.index);
        parts.push(pre);
        outLen += pre.length;
        const content = m[1];
        spans.push([outLen, outLen + content.length]);
        parts.push(content);
        outLen += content.length;
        cursor = m.index + m[0].length;
    }
    parts.push(text.slice(cursor));
    return { cleanedText: parts.join(''), spans };
}

const gptBlocks = extractFencedCodeBlocks(markedCodeSamplesText).map(b => extractAttnSpans(b));

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

function convertRawSaliencyToObject(saliency: any[]) {
    const converted: { [key: number]: number[] } = {};
    saliency.forEach((x: any) => { converted[x['index']] = x['saliency']; });
    return converted;
}

function convertTokens(tokens: string[]) {
    return tokens.map(t => t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  '));
}

/* ------------------------------------------------------------------ */
/* App                                                                */
/* ------------------------------------------------------------------ */

// (Optional) load the correlation matching JSON if available
const interventionFiles = import.meta.glob([
    '../../../correlation_matching_results.json',
    '../../../runs/attribution/correlation_matching_results.json',
], { query: '?raw', eager: true }) as Record<string, any>;
const causalInterventionJSONText = interventionFiles['../../../runs/attribution/correlation_matching_results.json']?.default
    || interventionFiles['../../../correlation_matching_results.json']?.default
    || "";
import { CausalInterventionSection } from './components/CausalIntervention';

let causalInterventionData: any = null;
try {
    causalInterventionData = JSON.parse(causalInterventionJSONText);
} catch (e) {
    console.log("No correlation matching results available or JSON parse failed.");
}

function App() {
    return (
        <div className="app-root">
            <header className="app-header">
                <h1>Attribution Analysis</h1>
            </header>
            <OverfitSection />
            <TrainSampleSection />

            {/* Third Section: New Causal Verification results */}
            {causalInterventionData && <CausalInterventionSection reportData={causalInterventionData} />}
        </div>
    );
}

/* ------------------------------------------------------------------ */
/* Section 1: Overfit Experiment                                      */
/* ------------------------------------------------------------------ */

function OverfitSection() {
    const [coefIndex, setCoefIndex] = useState(0);
    const coef = BOOST_COEFS[coefIndex] ?? '?';

    const testSample = allSaliencies['target_test_sample'];

    // Safety check if JSON was missing or malformed
    if (!testSample || !testSample['before']) {
        return (
            <section className="analysis-section">
                <div className="section-header">
                    <h2>Section 1: Overfit Experiment</h2>
                    <p className="no-data" style={{ marginTop: '16px' }}>
                        No `latest_saliency.json` found. Please run Option 1 in `NIF.py`.
                    </p>
                </div>
            </section>
        );
    }

    const testTokensBefore = useMemo(() => convertTokens(testSample['before']['full_tokens']), []);
    const testTokensAfter = useMemo(() => convertTokens(testSample['after']['full_tokens']), []);
    const testSalBefore = useMemo(() => convertRawSaliencyToObject(testSample['before']['saliency_list']), []);
    const testSalAfter = useMemo(() => convertRawSaliencyToObject(testSample['after']['saliency_list']), []);

    const overfitSample = overfitResults[coefIndex];
    const overfitTokens = useMemo(() => overfitSample ? convertTokens(overfitSample['before_original']['full_tokens']) : [], [overfitSample]);
    const overfitSal = useMemo(() => overfitSample ? convertRawSaliencyToObject(overfitSample['before_original']['saliency_list']) : {}, [overfitSample]);

    const goPrev = () => setCoefIndex(((coefIndex - 1) + BOOST_COEFS.length) % BOOST_COEFS.length);
    const goNext = () => setCoefIndex((coefIndex + 1) % BOOST_COEFS.length);

    return (
        <section className="analysis-section">
            <div className="section-header">
                <h2>Section 1: Overfit Experiment</h2>
                <p className="section-desc">
                    对同一个 test sample，使用 GPT 标注的 attention token 以不同强度（boost_coef）过拟合后，
                    观察模型 saliency 分布的变化。左侧为 test sample 的推理结果，右侧为过拟合后 test sample 的 saliency 变化。
                </p>
            </div>

            <div className="two-panel-row">
                <div className="panel" style={{ flex: '2 0 0' }}>
                    <div className="panel-title">
                        <span className="badge badge-test">TEST</span>
                        Target Test Sample
                    </div>
                    <div className="panel-columns">
                        <div className="panel-col">
                            <div className="col-label">Ground Truth</div>
                            <SwitchTokenCodeBlock tokens={testTokensBefore} salienciesByToken={testSalBefore} answerStartIndex={testSample['before']['start_index']} />
                        </div>
                        <div className="panel-col">
                            <div className="col-label">Prediction</div>
                            <SwitchTokenCodeBlock tokens={testTokensAfter} salienciesByToken={testSalAfter} answerStartIndex={testSample['after']['start_index']} />
                        </div>
                    </div>
                </div>

                <div className="panel" style={{ flex: '1 0 0' }}>
                    <div className="panel-title">
                        <span className="badge badge-overfit">OVERFIT</span>
                        <span>Overfit Result</span>
                        <span className="coef-label">boost_coef = <strong>{coef}</strong></span>
                        <span className="nav-buttons">
                            <button onClick={goPrev}>← prev</button>
                            <button onClick={goNext}>next →</button>
                        </span>
                    </div>
                    {overfitSample ? (
                        <div className="panel-columns">
                            <div className="panel-col">
                                <div className="col-label">Saliency after overfit (coef={coef})</div>
                                <SwitchTokenCodeBlock key={coefIndex} tokens={overfitTokens} salienciesByToken={overfitSal} answerStartIndex={overfitSample['before_original']['start_index']} />
                            </div>
                        </div>
                    ) : (
                        <div className="no-data">No overfit data available</div>
                    )}
                </div>
            </div>
        </section>
    );
}

/* ------------------------------------------------------------------ */
/* Section 2: Train Sample (Saliency + GPT Annotation)                */
/* ------------------------------------------------------------------ */

type LayerMode = 'both' | 'saliency' | 'gpt';

function TrainSampleSection() {
    const [sampleIndex, setSampleIndex] = useState(0);
    const [layerMode, setLayerMode] = useState<LayerMode>('both');

    // Determine max count: max of train samples from JSON and GPT blocks
    const maxCount = Math.max(trainSamples.length, gptBlocks.length);
    if (maxCount === 0) {
        return (
            <section className="analysis-section">
                <div className="section-header">
                    <h2>Section 2: Related Train Samples</h2>
                    <p className="no-data" style={{ marginTop: '16px' }}>
                        No train samples or GPT annotations (`marked_code_samples.md`) found. Please run Option 2 in `NIF.py`.
                    </p>
                </div>
            </section>
        );
    }

    const goPrev = () => setSampleIndex(((sampleIndex - 1) + maxCount) % maxCount);
    const goNext = () => setSampleIndex((sampleIndex + 1) % maxCount);

    const trainSample = sampleIndex < trainSamples.length ? trainSamples[sampleIndex] : null;
    const gptBlock = sampleIndex < gptBlocks.length ? gptBlocks[sampleIndex] : null;

    const hasSaliency = trainSample !== null;
    const hasGpt = gptBlock !== null;

    return (
        <section className="analysis-section">
            <div className="section-header">
                <h2>Section 2: Related Train Samples</h2>
                <p className="section-desc">
                    与 test sample 最相关的训练样本。<strong>黄色热力图</strong> 为模型 saliency 分数，
                    <strong>紫色标注</strong> 为 GPT 标记的关键 token（<code>&lt;ATTN&gt;</code>）。
                    可切换 "Saliency"、"GPT" 和 "Both" 三种视图进行对比。
                </p>
            </div>

            <div className="train-nav">
                <span className="badge badge-train">TRAIN</span>
                <span>Sample {sampleIndex + 1} / {maxCount}</span>
                <span className="nav-buttons">
                    <button onClick={goPrev}>← prev</button>
                    <button onClick={goNext}>next →</button>
                </span>

                <span className="layer-toggle">
                    <button className={layerMode === 'both' ? 'active' : ''} onClick={() => setLayerMode('both')}>Both</button>
                    <button className={layerMode === 'saliency' ? 'active' : ''} onClick={() => setLayerMode('saliency')}>Saliency</button>
                    <button className={layerMode === 'gpt' ? 'active' : ''} onClick={() => setLayerMode('gpt')}>GPT</button>
                </span>
            </div>

            <TrainSampleView
                key={sampleIndex}
                trainSample={trainSample}
                gptBlock={gptBlock}
                layerMode={layerMode}
                hasSaliency={hasSaliency}
                hasGpt={hasGpt}
            />
        </section>
    );
}

/* ------------------------------------------------------------------ */
/* TrainSampleView: combined saliency + GPT overlay                   */
/* ------------------------------------------------------------------ */

function TrainSampleView({
    trainSample,
    gptBlock,
    layerMode,
    hasSaliency,
    hasGpt,
}: {
    trainSample: any | null;
    gptBlock: { cleanedText: string; spans: [number, number][] } | null;
    layerMode: LayerMode;
    hasSaliency: boolean;
    hasGpt: boolean;
}) {
    const showSaliency = layerMode === 'both' || layerMode === 'saliency';
    const showGpt = layerMode === 'both' || layerMode === 'gpt';

    // If we have saliency data, render using HighlightedTokenCodeBlock with optional GPT overlay
    if (hasSaliency && trainSample) {
        const tokens = convertTokens(trainSample['before_original']['full_tokens']);
        const sal = convertRawSaliencyToObject(trainSample['before_original']['saliency_list']);
        const startIndex = trainSample['before_original']['start_index'];

        // Compute GPT annotation token indices only when GPT layer is visible
        const gptTokenIndices = (showGpt && hasGpt && gptBlock)
            ? computeGptTokenIndices(tokens, gptBlock)
            : undefined;

        return (
            <div className="train-panel-columns">
                <div className="panel-col">
                    <div className="col-label">
                        {layerMode === 'both' ? 'Model Saliency + GPT Annotation'
                            : layerMode === 'saliency' ? 'Model Saliency'
                                : 'GPT Annotation Only'}
                    </div>
                    <SwitchTokenCodeBlock
                        key={`train-${layerMode}`}
                        tokens={tokens}
                        salienciesByToken={showSaliency ? sal : {}}
                        answerStartIndex={startIndex}
                        gptAnnotationIndices={gptTokenIndices}
                    />
                </div>
            </div>
        );
    }

    // Fallback: no saliency data — show GPT only if GPT layer is on
    if (hasGpt && gptBlock && showGpt) {
        return (
            <div className="train-panel-columns">
                <div className="panel-col">
                    <div className="col-label">GPT Annotation</div>
                    <GptAnnotatedBlock cleanedText={gptBlock.cleanedText} spans={gptBlock.spans} />
                </div>
            </div>
        );
    }

    if (layerMode === 'saliency' && !hasSaliency) {
        return <div className="no-data">该样本暂无 saliency 数据。请使用 NIF_old.py 的 Option 2 重新生成 JSON。</div>;
    }

    return <div className="no-data">No data for this sample</div>;
}

/* ------------------------------------------------------------------ */
/* Compute GPT token indices from char spans                          */
/* ------------------------------------------------------------------ */

function computeGptTokenIndices(
    tokens: string[],
    gptBlock: { cleanedText: string; spans: [number, number][] }
): Set<number> {
    // Build token-to-char mapping: each token maps to a character range in the joined text
    const indices = new Set<number>();
    const fullText = tokens.join('');
    const gptText = gptBlock.cleanedText;

    // Simple approach: for each <ATTN> span in GPT text, find which tokens overlap
    // First build char offset array for tokens
    const tokenStarts: number[] = [];
    let offset = 0;
    for (const t of tokens) {
        tokenStarts.push(offset);
        offset += t.length;
    }

    // The GPT text and token text may not align perfectly (different tokenization)
    // Use fuzzy matching: find each ATTN span's text in the token stream
    for (const [spanStart, spanEnd] of gptBlock.spans) {
        const spanText = gptText.slice(spanStart, spanEnd);
        // Find this text in the full token text
        const matchIdx = fullText.indexOf(spanText);
        if (matchIdx >= 0) {
            // Mark all tokens that overlap with [matchIdx, matchIdx + spanText.length)
            const matchEnd = matchIdx + spanText.length;
            for (let ti = 0; ti < tokens.length; ti++) {
                const tStart = tokenStarts[ti];
                const tEnd = tStart + tokens[ti].length;
                if (tEnd > matchIdx && tStart < matchEnd) {
                    indices.add(ti);
                }
            }
        }
    }
    return indices;
}

/* ------------------------------------------------------------------ */
/* GPT Annotated Block (fallback, char-level rendering)               */
/* ------------------------------------------------------------------ */

function GptAnnotatedBlock({ cleanedText, spans }: { cleanedText: string; spans: [number, number][] }) {
    const elements: React.ReactNode[] = [];
    let cursor = 0;

    for (let si = 0; si < spans.length; si++) {
        const [start, end] = spans[si];
        if (cursor < start) elements.push(<span key={`u${si}`}>{cleanedText.slice(cursor, start)}</span>);
        elements.push(<span key={`m${si}`} className="attn-highlight">{cleanedText.slice(start, end)}</span>);
        cursor = end;
    }
    if (cursor < cleanedText.length) elements.push(<span key="tail">{cleanedText.slice(cursor)}</span>);

    return (
        <pre className="highlighted-token-code-block">
            <code>{elements}</code>
        </pre>
    );
}

export default App;
