'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, CheckCircle2, XCircle, Server, Cpu } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

interface SystemStatus {
    version: string;
    env: string;
    features: {
        python_executor: boolean;
        shell_executor: boolean;
    };
    workspace_dir: string;
    costs_dir: string;
}

export default function SystemPage() {
    const router = useRouter();
    const [status, setStatus] = useState<SystemStatus | null>(null);
    const [health, setHealth] = useState<'ok' | 'error' | 'loading'>('loading');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        Promise.all([
            fetch(`${API_BASE_URL}/system/status`).then(r => r.json()),
            fetch(`${API_BASE_URL}/health`).then(r => r.ok ? 'ok' : 'error').catch(() => 'error'),
        ]).then(([sys, h]) => {
            setStatus(sys);
            setHealth(h as any);
            setLoading(false);
        }).catch(() => {
            setHealth('error');
            setLoading(false);
        });
    }, []);

    const Feature = ({ label, enabled }: { label: string; enabled: boolean }) => (
        <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            padding: '12px 16px', background: '#0d0d10',
            border: `1px solid ${enabled ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.07)'}`,
            borderRadius: 8,
        }}>
            {enabled
                ? <CheckCircle2 size={14} color="#10b981" />
                : <XCircle size={14} color="#5c5c6e" />}
            <span style={{ fontSize: 13, color: enabled ? '#f0f0f2' : '#5c5c6e' }}>{label}</span>
            <span style={{
                marginLeft: 'auto', fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                color: enabled ? '#10b981' : '#5c5c6e',
            }}>
                {enabled ? 'enabled' : 'disabled'}
            </span>
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
                        background: 'none', border: 'none', color: '#9898a8', cursor: 'pointer', fontSize: 13,
                    }}>
                        <ArrowLeft size={14} /> Back
                    </button>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#f0f0f2', fontFamily: 'Sora, sans-serif' }}>
                        System Status
                    </span>
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                            width: 7, height: 7, borderRadius: '50%',
                            background: health === 'ok' ? '#10b981' : health === 'loading' ? '#f59e0b' : '#ef4444',
                        }} />
                        <span style={{ fontSize: 11, color: '#5c5c6e' }}>
                            {health === 'ok' ? 'API online' : health === 'loading' ? 'Checking...' : 'API offline'}
                        </span>
                    </div>
                </header>

                <div style={{ maxWidth: 640, margin: '0 auto', padding: '40px 24px' }}>
                    {loading && <div style={{ color: '#5c5c6e', fontSize: 13 }}>Loading system info...</div>}

                    {status && (
                        <>
                            {/* Version + env */}
                            <div style={{
                                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 32,
                            }}>
                                {[
                                    { label: 'Version', value: status.version, icon: Server },
                                    { label: 'Environment', value: status.env, icon: Cpu },
                                ].map(({ label, value, icon: Icon }) => (
                                    <div key={label} style={{
                                        background: '#16161a', border: '1px solid rgba(255,255,255,0.07)',
                                        borderRadius: 12, padding: '20px 24px',
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#5c5c6e', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                                            <Icon size={11} color="#6366f1" /> {label}
                                        </div>
                                        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'Sora, sans-serif', color: '#f0f0f2' }}>
                                            {value}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Features */}
                            <div style={{ marginBottom: 32 }}>
                                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                    Features
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                    <Feature label="Python Executor" enabled={status.features.python_executor} />
                                    <Feature label="Shell Executor" enabled={status.features.shell_executor} />
                                </div>
                            </div>

                            {/* Paths */}
                            <div>
                                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#5c5c6e', marginBottom: 12 }}>
                                    Paths
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                    {[
                                        { label: 'Workspace', value: status.workspace_dir },
                                        { label: 'Costs', value: status.costs_dir },
                                    ].map(({ label, value }) => (
                                        <div key={label} style={{
                                            display: 'flex', alignItems: 'center', gap: 12,
                                            padding: '12px 16px', background: '#16161a',
                                            border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8,
                                        }}>
                                            <span style={{ fontSize: 12, color: '#5c5c6e', minWidth: 80 }}>{label}</span>
                                            <span style={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace', color: '#9898a8' }}>{value}</span>
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
