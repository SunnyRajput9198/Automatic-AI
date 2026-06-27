'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Brain, Zap, Users, ThumbsUp, ThumbsDown } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

export default function MemoryPage() {
    const router = useRouter();
    const [stats, setStats] = useState<any>(null);
    const [agentPerf, setAgentPerf] = useState<any>({});
    const [toolPerf, setToolPerf] = useState<any>({});
    const [feedback, setFeedback] = useState<any[]>([]);
    const [feedbackStats, setFeedbackStats] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        Promise.all([
            fetch(`${API_BASE_URL}/memory/stats`).then(r => r.json()),
            fetch(`${API_BASE_URL}/agents/performance`).then(r => r.json()),
            fetch(`${API_BASE_URL}/tools/performance`).then(r => r.json()),
            fetch(`${API_BASE_URL}/feedback/history`).then(r => r.json()),
            fetch(`${API_BASE_URL}/feedback/stats`).then(r => r.json()),
        ]).then(([s, ap, tp, fh, fs]) => {
            setStats(s);
            setAgentPerf(ap);
            setToolPerf(tp);
            setFeedback(fh);
            setFeedbackStats(fs);
            setLoading(false);
        }).catch(() => setLoading(false));
    }, []);

    const StatCard = ({ icon: Icon, label, value, color = '#6366f1' }: any) => (
        <div style={{
            background: '#16161a', border: '1px solid rgba(255,255,255,0.07)',
            borderRadius: 12, padding: '20px 24px',
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#5c5c6e', fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', marginBottom: 8 }}>
                <Icon size={12} color={color} />
                {label}
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, fontFamily: 'Sora, sans-serif', color: '#f0f0f2' }}>
                {value}
            </div>
        </div>
    );

    return (
        <>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d10; color: #f0f0f2; font-family: 'DM Sans', sans-serif; min-height: 100vh; }
        `}</style>

            <div style={{ minHeight: '100vh' }}>
                <header style={{
                    height: 52, display: 'flex', alignItems: 'center',
                    padding: '0 32px', borderBottom: '1px solid rgba(255,255,255,0.07)', gap: 16,
                }}>
                    <button onClick={() => router.push('/')} style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        background: 'none', border: 'none', color: '#9898a8',
                        cursor: 'pointer', fontSize: 13,
                    }}>
                        <ArrowLeft size={14} /> Back
                    </button>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#f0f0f2', fontFamily: 'Sora, sans-serif' }}>
                        Memory Dashboard
                    </span>
                </header>

                <div style={{ maxWidth: 900, margin: '0 auto', padding: '40px 24px' }}>
                    {loading && <div style={{ color: '#5c5c6e', fontSize: 13 }}>Loading...</div>}

                    {!loading && (
                        <>
                            {/* Stats Grid */}
                            {stats && (
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 32 }}>
                                    <StatCard icon={Brain} label="Feedback Entries" value={stats.feedback_entries} />
                                    <StatCard icon={Users} label="Agent Perf" value={stats.agent_performance_entries} color="#10b981" />
                                    <StatCard icon={Zap} label="Agent Prefs" value={stats.agent_preference_entries} color="#f59e0b" />
                                    <StatCard icon={Zap} label="Tool Success" value={stats.tool_success_entries} color="#6366f1" />
                                </div>
                            )}

                            {/* Feedback Stats */}
                            {feedbackStats && (
                                <div style={{ marginBottom: 32 }}>
                                    <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                        Feedback Overview
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                                        <div style={{ background: '#16161a', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: '16px 20px' }}>
                                            <div style={{ fontSize: 11, color: '#5c5c6e', marginBottom: 6 }}>Total Feedback</div>
                                            <div style={{ fontSize: 24, fontWeight: 700, color: '#f0f0f2' }}>{feedbackStats.total}</div>
                                        </div>
                                        <div style={{ background: '#16161a', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 12, padding: '16px 20px' }}>
                                            <div style={{ fontSize: 11, color: '#5c5c6e', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <ThumbsUp size={11} color="#10b981" /> Good
                                            </div>
                                            <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981' }}>{feedbackStats.good}</div>
                                        </div>
                                        <div style={{ background: '#16161a', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 12, padding: '16px 20px' }}>
                                            <div style={{ fontSize: 11, color: '#5c5c6e', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <ThumbsDown size={11} color="#ef4444" /> Bad
                                            </div>
                                            <div style={{ fontSize: 24, fontWeight: 700, color: '#ef4444' }}>{feedbackStats.bad}</div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Agent Performance */}
                            {Object.keys(agentPerf).length > 0 && (
                                <div style={{ marginBottom: 32 }}>
                                    <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                        Agent Performance
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column' as const, gap: 6 }}>
                                        {Object.entries(agentPerf).map(([name, stats]: any) => (
                                            <div key={name} style={{
                                                display: 'flex', alignItems: 'center', gap: 12,
                                                padding: '12px 16px', background: '#16161a',
                                                border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10,
                                            }}>
                                                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#6366f1', minWidth: 120 }}>{name}</span>
                                                <span style={{ fontSize: 12, color: '#9898a8' }}>calls: {stats.calls}</span>
                                                <span style={{ fontSize: 12, color: '#9898a8' }}>successes: {stats.successes}</span>
                                                <span style={{
                                                    fontSize: 12, marginLeft: 'auto',
                                                    color: stats.success_rate > 0.7 ? '#10b981' : '#ef4444',
                                                    fontFamily: 'JetBrains Mono, monospace',
                                                }}>
                                                    {(stats.success_rate * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Tool Performance */}
                            {Object.keys(toolPerf).length > 0 && (
                                <div style={{ marginBottom: 32 }}>
                                    <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                        Tool Performance
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column' as const, gap: 6 }}>
                                        {Object.entries(toolPerf).map(([name, data]: any) => (
                                            <div key={name} style={{
                                                display: 'flex', alignItems: 'center', gap: 12,
                                                padding: '12px 16px', background: '#16161a',
                                                border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10,
                                            }}>
                                                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#f59e0b', minWidth: 160 }}>{name}</span>
                                                <span style={{ fontSize: 12, color: '#9898a8', flex: 1 }}>
                                                    {JSON.stringify(data).slice(0, 80)}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Recent Feedback */}
                            {feedback.length > 0 && (
                                <div>
                                    <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                        Recent Feedback
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column' as const, gap: 6 }}>
                                        {feedback.slice(0, 10).map((f: any, i: number) => (
                                            <div key={i} style={{
                                                display: 'flex', alignItems: 'center', gap: 12,
                                                padding: '12px 16px', background: '#16161a',
                                                border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10,
                                            }}>
                                                <span style={{ fontSize: 16 }}>{f.feedback === 'good' ? '👍' : f.feedback === 'bad' ? '👎' : '😐'}</span>
                                                <span style={{ fontSize: 13, color: '#9898a8', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const }}>
                                                    {f.query}
                                                </span>
                                                <span style={{ fontSize: 11, color: '#5c5c6e', fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
                                                    {f.feedback}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </>
    );
}