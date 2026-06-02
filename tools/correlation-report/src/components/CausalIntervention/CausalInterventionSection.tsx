import { useState, useMemo } from 'react';
import { SwitchTokenCodeBlock } from '../SwitchTokenCodeBlock';

// ─── Types ────────────────────────────────────────────────────────────────────

type Annotation = 'correct' | 'incorrect' | 'ambiguous';
type AnnotationLabel = Annotation | null;

interface TestCorrelation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score: number;
}

interface TrainCorrelation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score: number;
    response_token_offset: number;
}

interface CorrelationPair {
    id: string;
    cos_sim: number;
    coarse_cos_sim: number;
    train_sample_id: number;
    test_correlation: TestCorrelation;
    train_correlation: TrainCorrelation;
    train_context: { source_context: string[]; target_context: string[] };
    annotation: AnnotationLabel;
}

interface TrainSampleDetail {
    full_tokens: string[];              // vocab-level encoding (Ġ/Ċ) for SwitchTokenCodeBlock
    answer_start_index: number;         // first response token index
    coarse_cos_sim: number;
    saliencies_by_token: Record<string, number[]>;  // "token_idx" -> saliency list
}

interface ReportData {
    experiment_meta: {
        test_sample_index: number;
        target_token_index: number;
        config?: Record<string, number>;
    };
    test_sample_baseline: {
        target_token: string;
        target_token_index: number;
        full_tokens: string[];
        top_correlations: TestCorrelation[];
    };
    correlation_pairs: CorrelationPair[];
    train_sample_details: Record<string, TrainSampleDetail>;  // str(train_idx) -> detail
}

interface Props { reportData: ReportData; }

// ─── Annotation persistence ───────────────────────────────────────────────────

const LS_KEY = 'eif_annotations_v3';
function loadAnnotations(): Record<string, Annotation> {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch { return {}; }
}
function persistAnnotations(data: Record<string, Annotation>) {
    localStorage.setItem(LS_KEY, JSON.stringify(data));
}

// ─── Visual helpers ───────────────────────────────────────────────────────────

function convertTokens(tokens: string[]) {
    return tokens.map(t => t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  '));
}

function cosSimStyle(s: number) {
    if (s > 0.6) return { bg: '#dcfce7', fg: '#15803d' };
    if (s > 0.3) return { bg: '#fef9c3', fg: '#854d0e' };
    return { bg: '#fee2e2', fg: '#b91c1c' };
}

const ANN_CFG: Record<Annotation, { emoji: string; label: string; bg: string; fg: string; border: string }> = {
    correct:   { emoji: '✓', label: 'Correct',   bg: '#dcfce7', fg: '#166534', border: '#22c55e' },
    incorrect: { emoji: '✗', label: 'Incorrect', bg: '#fee2e2', fg: '#991b1b', border: '#ef4444' },
    ambiguous: { emoji: '?', label: 'Ambiguous', bg: '#fef3c7', fg: '#92400e', border: '#f59e0b' },
};

// ─── Micro-components ─────────────────────────────────────────────────────────

function ContextChip({ tokens }: { tokens: string[] }) {
    return (
        <span style={{ fontFamily: 'monospace', fontSize: '12px', display: 'inline-flex', flexWrap: 'wrap', gap: '2px' }}>
            {tokens.map((tok, i) => {
                const marked = tok.startsWith('→[') && tok.endsWith(']←');
                const text   = (marked ? tok.slice(2, -2) : tok).replace(/\n/g, '↵').replace(/\t/g, '⇥') || '·';
                return (
                    <span key={i} style={{
                        padding: '1px 5px', borderRadius: '4px',
                        background: marked ? '#fde68a' : '#f1f5f9',
                        color:      marked ? '#78350f' : '#475569',
                        fontWeight: marked ? 700 : 400,
                        border: marked ? '1.5px solid #f59e0b' : '1px solid transparent',
                    }}>{text}</span>
                );
            })}
        </span>
    );
}

function AnnBtn({ type, current, onToggle }: { type: Annotation; current: AnnotationLabel; onToggle: () => void }) {
    const c = ANN_CFG[type];
    const active = current === type;
    return (
        <button onClick={onToggle} style={{
            padding: '4px 13px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px',
            border:      `1.5px solid ${active ? c.border : '#d1d5db'}`,
            background:  active ? c.bg  : '#fff',
            color:       active ? c.fg  : '#9ca3af',
            fontWeight:  active ? 700   : 500,
            transition:  'all 0.15s ease',
        }}>
            {c.emoji} {c.label}
        </button>
    );
}

function FilterBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
    return (
        <button onClick={onClick} style={{
            padding: '3px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer',
            border:     `1px solid ${active ? '#6366f1' : '#d1d5db'}`,
            background:  active ? '#6366f1' : '#fff',
            color:       active ? '#fff'    : '#6b7280',
            fontWeight:  active ? 700 : 500,
            transition: 'all 0.12s',
        }}>{children}</button>
    );
}

// ─── Train Sample Group ───────────────────────────────────────────────────────

function TrainSampleGroup({
    trainIdx,
    detail,
    pairs,
    annotations,
    onToggle,
    onClear,
}: {
    trainIdx: number;
    detail: TrainSampleDetail | undefined;
    pairs: CorrelationPair[];
    annotations: Record<string, Annotation>;
    onToggle: (id: string, label: Annotation) => void;
    onClear: (id: string) => void;
}) {
    const [collapsed, setCollapsed] = useState(false);

    const bestSim = Math.max(...pairs.map(p => p.cos_sim));
    const coarse  = detail?.coarse_cos_sim ?? pairs[0]?.coarse_cos_sim ?? 0;

    // Prepare SwitchTokenCodeBlock data from detail
    const displayTokens = useMemo(() =>
        detail ? convertTokens(detail.full_tokens) : [],
    [detail]);

    // saliencies_by_token is { "tokenIdx": number[] } — SwitchTokenCodeBlock wants { [number]: number[] }
    const salienciesByToken = useMemo(() => {
        if (!detail) return {};
        const out: Record<number, number[]> = {};
        Object.entries(detail.saliencies_by_token).forEach(([k, v]) => { out[parseInt(k)] = v; });
        return out;
    }, [detail]);

    return (
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '12px', overflow: 'hidden', marginBottom: '20px', boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>

            {/* ── Group header ── */}
            <div
                onClick={() => setCollapsed(c => !c)}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', cursor: 'pointer', background: '#f8fafc', borderBottom: collapsed ? 'none' : '1px solid #e2e8f0', flexWrap: 'wrap' }}
            >
                <span style={{ fontSize: '11px', color: '#818cf8', fontWeight: 700, fontFamily: 'monospace' }}>TRAIN #{trainIdx}</span>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>coarse {coarse.toFixed(4)}</span>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>{pairs.length} pairs</span>
                <span style={{ ...cosSimStyle(bestSim), fontWeight: 700, fontSize: '12px', padding: '2px 10px', borderRadius: '10px' }}>
                    best {bestSim.toFixed(4)}
                </span>
                <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: '16px' }}>{collapsed ? '▶' : '▼'}</span>
            </div>

            {!collapsed && (
                <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>

                    {/* ── Full code view with interactive saliency ── */}
                    {detail ? (
                        <div>
                            <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
                                完整训练样本 — 点击 Response Token 切换 Saliency 视图
                            </div>
                            <SwitchTokenCodeBlock
                                tokens={displayTokens}
                                salienciesByToken={salienciesByToken}
                                answerStartIndex={detail.answer_start_index}
                            />
                        </div>
                    ) : (
                        <div style={{ color: '#f59e0b', fontSize: '12px', background: '#fef3c7', padding: '8px 12px', borderRadius: '6px' }}>
                            ⚠ 该训练样本的完整 token 数据不在 JSON 中（train_sample_details 缺失）。请重新运行实验。
                        </div>
                    )}

                    {/* ── Correlation pair cards ── */}
                    <div>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
                            Correlation Pairs ({pairs.length})
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {pairs.map(pair => {
                                const ann = annotations[pair.id] ?? pair.annotation;
                                const annCfg = ann ? ANN_CFG[ann] : null;
                                const { bg, fg } = cosSimStyle(pair.cos_sim);

                                return (
                                    <div key={pair.id} style={{
                                        border: ann ? `2px solid ${annCfg!.border}` : '1px solid #e5e7eb',
                                        borderRadius: '8px', background: '#fff', padding: '12px 14px',
                                        display: 'flex', flexDirection: 'column', gap: '8px',
                                        boxShadow: ann ? `0 0 0 3px ${annCfg!.bg}` : 'none',
                                        transition: 'border-color 0.2s, box-shadow 0.2s',
                                    }}>

                                        {/* Row 1: ID + cos_sim + correlation tags + inline ann badge */}
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                            <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#c4b5fd' }}>{pair.id}</span>
                                            <span style={{ background: bg, color: fg, fontWeight: 700, fontSize: '12px', padding: '2px 9px', borderRadius: '10px', flexShrink: 0 }}>
                                                {pair.cos_sim.toFixed(4)}
                                            </span>

                                            {/* Test correlation tag */}
                                            <span style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '6px', padding: '2px 8px', fontSize: '11px', fontFamily: 'monospace', color: '#1d4ed8' }}>
                                                <span style={{ opacity: 0.65 }}>test: </span>
                                                <strong>{pair.test_correlation.source_token.trim() || '·'}</strong>
                                                <span style={{ margin: '0 3px', opacity: 0.55 }}>→</span>
                                                <strong>{pair.test_correlation.target_token.trim() || '·'}</strong>
                                            </span>

                                            <span style={{ color: '#c4b5fd', fontSize: '15px' }}>⇔</span>

                                            {/* Train correlation tag */}
                                            <span style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '6px', padding: '2px 8px', fontSize: '11px', fontFamily: 'monospace', color: '#92400e' }}>
                                                <span style={{ opacity: 0.65 }}>train: </span>
                                                <strong>{pair.train_correlation.source_token.trim() || '·'}</strong>
                                                <span style={{ margin: '0 3px', opacity: 0.55 }}>→</span>
                                                <strong>{pair.train_correlation.target_token.trim() || '·'}</strong>
                                                <span style={{ marginLeft: '4px', opacity: 0.5, fontSize: '10px' }}>+{pair.train_correlation.response_token_offset}</span>
                                            </span>

                                            {ann && (
                                                <span style={{ marginLeft: 'auto', background: annCfg!.bg, color: annCfg!.fg, border: `1px solid ${annCfg!.border}`, borderRadius: '6px', padding: '2px 9px', fontSize: '11px', fontWeight: 700 }}>
                                                    {annCfg!.emoji} {annCfg!.label}
                                                </span>
                                            )}
                                        </div>

                                        {/* Row 2: Context windows */}
                                        <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', background: '#f8fafc', borderRadius: '6px', padding: '7px 10px', border: '1px solid #f1f5f9' }}>
                                            <div style={{ flex: '1 1 160px' }}>
                                                <div style={{ fontSize: '10px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', marginBottom: '4px' }}>Source Context</div>
                                                <ContextChip tokens={pair.train_context.source_context} />
                                            </div>
                                            <div style={{ width: '1px', background: '#e2e8f0' }} />
                                            <div style={{ flex: '1 1 160px' }}>
                                                <div style={{ fontSize: '10px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', marginBottom: '4px' }}>Target Context</div>
                                                <ContextChip tokens={pair.train_context.target_context} />
                                            </div>
                                        </div>

                                        {/* Row 3: Annotation buttons */}
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                            <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase' }}>标注:</span>
                                            {(['correct', 'incorrect', 'ambiguous'] as Annotation[]).map(label => (
                                                <AnnBtn key={label} type={label} current={ann} onToggle={() => onToggle(pair.id, label)} />
                                            ))}
                                            {ann && (
                                                <button onClick={() => onClear(pair.id)} style={{ fontSize: '11px', color: '#9ca3af', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>
                                                    clear
                                                </button>
                                            )}
                                            <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#c4b5fd', fontFamily: 'monospace' }}>
                                                src_sal {pair.train_correlation.saliency_score.toFixed(4)} · test_sal {pair.test_correlation.saliency_score.toFixed(4)}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function CausalInterventionSection({ reportData }: Props) {
    if (!reportData?.experiment_meta) return null;

    const baseline    = reportData.test_sample_baseline;
    const allPairs    = reportData.correlation_pairs    || [];
    const details     = reportData.train_sample_details || {};
    const testCorrs   = baseline.top_correlations       || [];

    // ── Annotation state ──────────────────────────────────────────────────────
    const [annotations, setAnnotations] = useState<Record<string, Annotation>>(loadAnnotations);

    const toggleAnnotation = (id: string, label: Annotation) => {
        setAnnotations(prev => {
            const next = { ...prev };
            if (next[id] === label) delete next[id]; else next[id] = label;
            persistAnnotations(next);
            return next;
        });
    };
    const clearAnnotation = (id: string) => {
        setAnnotations(prev => { const next = { ...prev }; delete next[id]; persistAnnotations(next); return next; });
    };

    // ── Filters ───────────────────────────────────────────────────────────────
    const [threshold,       setThreshold]       = useState(0.0);
    const [testSrcFilter,   setTestSrcFilter]   = useState<number | 'all'>('all');
    const [annFilter,       setAnnFilter]       = useState<'all' | 'unannotated' | Annotation>('all');

    const effectiveAnn = (pair: CorrelationPair): AnnotationLabel =>
        annotations[pair.id] !== undefined ? annotations[pair.id] : pair.annotation;

    const displayPairs = useMemo(() => allPairs.filter(p => {
        if (p.cos_sim < threshold) return false;
        if (testSrcFilter !== 'all' && p.test_correlation.source_token_index !== testSrcFilter) return false;
        const ann = effectiveAnn(p);
        if (annFilter === 'unannotated') return !ann;
        if (annFilter !== 'all') return ann === annFilter;
        return true;
    }), [allPairs, threshold, testSrcFilter, annFilter, annotations]);

    // ── Group by train_sample_id, sort groups by best cos_sim ─────────────────
    const groups = useMemo(() => {
        const map = new Map<number, CorrelationPair[]>();
        displayPairs.forEach(p => {
            if (!map.has(p.train_sample_id)) map.set(p.train_sample_id, []);
            map.get(p.train_sample_id)!.push(p);
        });
        return Array.from(map.entries())
            .map(([id, pairs]) => ({ id, pairs, bestSim: Math.max(...pairs.map(p => p.cos_sim)) }))
            .sort((a, b) => b.bestSim - a.bestSim);
    }, [displayPairs]);

    // ── Stats ─────────────────────────────────────────────────────────────────
    const stats = useMemo(() => {
        const above = allPairs.filter(p => p.cos_sim >= threshold).length;
        const cnt = { correct: 0, incorrect: 0, ambiguous: 0, total: 0 };
        allPairs.forEach(p => { const a = effectiveAnn(p); if (a) { cnt[a]++; cnt.total++; } });
        return { total: allPairs.length, above, ...cnt };
    }, [allPairs, threshold, annotations]);

    // ── Export ────────────────────────────────────────────────────────────────
    const exportJSON = () => {
        const out = { ...reportData, correlation_pairs: allPairs.map(p => ({ ...p, annotation: effectiveAnn(p) })) };
        const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }));
        Object.assign(document.createElement('a'), { href: url, download: 'correlation_annotated.json' }).click();
        URL.revokeObjectURL(url);
    };

    // ─────────────────────────────────────────────────────────────────────────
    return (
        <section className="analysis-section" style={{ marginTop: '40px', borderTop: '2px dashed #c7d2fe', paddingTop: '40px' }}>

            {/* ── Header ── */}
            <div className="section-header">
                <h2>Section 3: Correlation Pair Annotation</h2>
                <p className="section-desc">
                    <strong>标注目标:</strong> 对每个 train correlation (source→target)，在训练样本自身的代码语境下判断该关联是否有语义依据。<br />
                    <strong>完整训练代码</strong>展示在每个样本组内，可点击 Response Token 切换 Saliency 热力图视角。<br />
                    <strong>Target Token (test):</strong>&nbsp;
                    <code style={{ background: '#fee2e2', padding: '2px 7px', borderRadius: '4px' }}>{baseline.target_token}</code>
                    &nbsp;@ idx {baseline.target_token_index}
                </p>
            </div>

            {/* ── Stats bar ── */}
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '18px' }}>
                {[
                    { label: 'Total Pairs',   value: stats.total, color: '#6b7280', bg: '#f9fafb' },
                    { label: '≥ Threshold',   value: stats.above, color: '#2563eb', bg: '#eff6ff' },
                    { label: '✓ Correct',     value: stats.correct,   color: '#15803d', bg: '#f0fdf4' },
                    { label: '✗ Incorrect',   value: stats.incorrect, color: '#b91c1c', bg: '#fef2f2' },
                    { label: '? Ambiguous',   value: stats.ambiguous, color: '#b45309', bg: '#fffbeb' },
                ].map(({ label, value, color, bg }) => (
                    <div key={label} style={{ background: bg, border: '1px solid #e5e7eb', borderRadius: '8px', padding: '8px 14px', textAlign: 'center', flex: '0 0 auto', minWidth: '88px' }}>
                        <div style={{ fontSize: '22px', fontWeight: 700, color, lineHeight: 1.2 }}>{value}</div>
                        <div style={{ fontSize: '10px', color: '#9ca3af', marginTop: '2px', fontWeight: 600, textTransform: 'uppercase' }}>{label}</div>
                    </div>
                ))}
                <button
                    onClick={exportJSON}
                    style={{ marginLeft: 'auto', padding: '0 20px', background: 'linear-gradient(135deg,#1d4ed8,#6366f1)', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 700, fontSize: '13px', cursor: 'pointer', minHeight: '60px', boxShadow: '0 2px 8px rgba(99,102,241,0.3)' }}
                >
                    ↓ Export JSON
                </button>
            </div>

            {/* ── Filters ── */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '14px 18px', marginBottom: '18px', display: 'flex', flexDirection: 'column', gap: '12px' }}>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', minWidth: '100px' }}>cos_sim ≥</span>
                    <input type="range" min={0} max={0.2} step={0.001} value={threshold}
                        onChange={e => setThreshold(parseFloat(e.target.value))}
                        style={{ width: '160px', accentColor: '#6366f1' }} />
                    <span style={{ fontWeight: 700, fontSize: '14px', color: '#4338ca', minWidth: '42px' }}>{threshold.toFixed(3)}</span>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>({stats.above} pairs · {groups.length} groups)</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', minWidth: '100px' }}>Test Corr</span>
                    <FilterBtn active={testSrcFilter === 'all'} onClick={() => setTestSrcFilter('all')}>All</FilterBtn>
                    {testCorrs.map(c => (
                        <button key={c.source_token_index} onClick={() => setTestSrcFilter(c.source_token_index)}
                            title={`saliency: ${c.saliency_score?.toFixed(4)}`}
                            style={{ padding: '3px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 600, fontFamily: 'monospace', cursor: 'pointer', transition: 'all 0.12s', background: testSrcFilter === c.source_token_index ? '#3b82f6' : '#fff', color: testSrcFilter === c.source_token_index ? '#fff' : '#1d4ed8', border: `1px solid ${testSrcFilter === c.source_token_index ? '#3b82f6' : '#bfdbfe'}` }}>
                            {c.source_token.trim() || '[SP]'} → {c.target_token.trim() || '[SP]'}
                        </button>
                    ))}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', minWidth: '100px' }}>Annotation</span>
                    {(['all', 'unannotated', 'correct', 'incorrect', 'ambiguous'] as const).map(f => (
                        <FilterBtn key={f} active={annFilter === f} onClick={() => setAnnFilter(f)}>
                            {f.charAt(0).toUpperCase() + f.slice(1)}
                        </FilterBtn>
                    ))}
                </div>
            </div>

            {/* ── Result count ── */}
            <div style={{ marginBottom: '14px', fontSize: '13px', color: '#64748b' }}>
                显示 <strong style={{ color: '#1e293b' }}>{groups.length}</strong> 个训练样本组 /&nbsp;
                <strong style={{ color: '#1e293b' }}>{displayPairs.length}</strong> 条 pair 记录
            </div>

            {/* ── Train Sample Groups ── */}
            {groups.length === 0 ? (
                <div className="no-data">无匹配记录，尝试调低 cos_sim 阈值或切换筛选条件。</div>
            ) : (
                <div>
                    {groups.map(({ id, pairs }) => (
                        <TrainSampleGroup
                            key={id}
                            trainIdx={id}
                            detail={details[String(id)]}
                            pairs={pairs}
                            annotations={annotations}
                            onToggle={toggleAnnotation}
                            onClear={clearAnnotation}
                        />
                    ))}
                </div>
            )}
        </section>
    );
}
