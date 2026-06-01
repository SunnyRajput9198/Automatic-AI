'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, CheckCircle2, XCircle, Clock, Zap, DollarSign, BarChart2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

interface TaskData {
    task_id: string;
    duration_sec: number;
    success: boolean;
    total_llm_calls: number;
    total_retries: number;
    estimated_cost_usd: number;
    reasoning_calls: number;
    planning_calls: number;
    execution_calls: number;
    critic_calls: number;
    reflection_calls: number;
}

interface Analytics {
    total_tasks: number;
    successful: number;
    failed: number;
    success_rate: number;
    avg_duration: number;
    avg_llm_calls: number;
    avg_cost_usd: number;
    total_cost_usd: number;
    avg_retries: number;
    tasks: TaskData[];
}

export default function AnalyticsPage() {
    const router = useRouter();
    const [data, setData] = useState<Analytics | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_BASE_URL}/analytics`)
            .then(r => r.json())
            .then(d => { setData(d); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    const StatCard = ({ icon: Icon, label, value, color = '#6366f1' }: any) => (
        <div style={{
            background: '#16161a', border: '1px solid rgba(255,255,255,0.07)',
            borderRadius: 12, padding: '20px 24px', display: 'flex',
            flexDirection: 'column' as const, gap: 8,
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#5c5c6e', fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em' }}>
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
                        Analytics
                    </span>
                </header>

                <div style={{ maxWidth: 900, margin: '0 auto', padding: '48px 24px' }}>
                    {loading && <div style={{ color: '#5c5c6e', fontSize: 13 }}>Loading analytics...</div>}

                    {data && (
                        <>
                            {/* Stats Grid */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 32 }}>
                                <StatCard icon={BarChart2} label="Total Tasks" value={data.total_tasks} />
                                <StatCard icon={CheckCircle2} label="Success Rate" value={`${data.success_rate}%`} color="#10b981" />
                                <StatCard icon={Clock} label="Avg Duration" value={`${data.avg_duration}s`} color="#f59e0b" />
                                <StatCard icon={DollarSign} label="Total Cost" value={`$${data.total_cost_usd}`} color="#6366f1" />
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 40 }}>
                                <StatCard icon={Zap} label="Avg LLM Calls" value={data.avg_llm_calls} />
                                <StatCard icon={XCircle} label="Avg Retries" value={data.avg_retries} color="#ef4444" />
                                <StatCard icon={DollarSign} label="Avg Cost/Task" value={`$${data.avg_cost_usd}`} color="#10b981" />
                            </div>

                            {/* Success vs Failed bar */}
                            <div style={{ marginBottom: 40 }}>
                                <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                    Success vs Failed
                                </div>
                                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                                    <div style={{ flex: data.successful, background: '#10b981', height: 8, borderRadius: 4 }} />
                                    <div style={{ flex: data.failed, background: '#ef4444', height: 8, borderRadius: 4 }} />
                                </div>
                                <div style={{ display: 'flex', gap: 20, fontSize: 12, color: '#9898a8' }}>
                                    <span style={{ color: '#10b981' }}>✓ {data.successful} successful</span>
                                    <span style={{ color: '#ef4444' }}>✗ {data.failed} failed</span>
                                </div>
                            </div>

                            {/* Agent calls breakdown */}
                            <div style={{ marginBottom: 40 }}>
                                <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                    Agent Call Distribution (latest 20 tasks)
                                </div>
                                {['reasoning', 'planning', 'execution', 'critic', 'reflection'].map(agent => {
                                    const total = data.tasks.reduce((a, t) => a + (t as any)[`${agent}_calls`], 0);
                                    const max = data.tasks.length * 5;
                                    const colors: Record<string, string> = {
                                        reasoning: '#6366f1', planning: '#8b5cf6',
                                        execution: '#f59e0b', critic: '#ef4444', reflection: '#10b981'
                                    };
                                    return (
                                        <div key={agent} style={{ marginBottom: 10 }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#9898a8', marginBottom: 4 }}>
                                                <span style={{ textTransform: 'capitalize' as const }}>{agent}</span>
                                                <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{total} calls</span>
                                            </div>
                                            <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3 }}>
                                                <div style={{
                                                    height: '100%', borderRadius: 3,
                                                    background: colors[agent],
                                                    width: `${Math.min((total / max) * 100, 100)}%`,
                                                }} />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Recent tasks table */}
                            <div>
                                <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                    Recent Tasks
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column' as const, gap: 6 }}>
                                    {data.tasks.slice(0, 10).map((t, i) => (
                                        <div key={i} onClick={() => router.push(`/tasks/${t.task_id}`)} style={{
                                            display: 'flex', alignItems: 'center', gap: 12,
                                            padding: '12px 16px', background: '#16161a',
                                            border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10,
                                            cursor: 'pointer', fontSize: 12,
                                        }}>
                                            <span style={{ color: t.success ? '#10b981' : '#ef4444', flexShrink: 0 }}>
                                                {t.success ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                                            </span>
                                            <span style={{ fontFamily: 'JetBrains Mono, monospace', color: '#5c5c6e', fontSize: 10, flexShrink: 0 }}>
                                                {t.task_id.slice(0, 8)}
                                            </span>
                                            <span style={{ color: '#9898a8', flex: 1 }}>{t.duration_sec}s</span>
                                            <span style={{ color: '#9898a8' }}>{t.total_llm_calls} LLM calls</span>
                                            <span style={{ color: '#9898a8' }}>{t.total_retries} retries</span>
                                            <span style={{ color: '#6366f1', fontFamily: 'JetBrains Mono, monospace' }}>
                                                ${t.estimated_cost_usd.toFixed(6)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </>
    );
}