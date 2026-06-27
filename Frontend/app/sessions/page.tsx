'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, MessageSquare, CheckCircle2, XCircle, Loader2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

interface Session {
    session_id: string;
    task_count: number;
    latest_task: string;
    last_active: string;
    statuses: {
        completed: number;
        failed: number;
        running: number;
    };
}

export default function SessionsPage() {
    const router = useRouter();
    const [sessions, setSessions] = useState<Session[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedSession, setSelectedSession] = useState<string | null>(null);
    const [sessionTasks, setSessionTasks] = useState<any[]>([]);
    const [loadingTasks, setLoadingTasks] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE_URL}/sessions`)
            .then(r => r.json())
            .then(d => { setSessions(d.sessions || []); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    const loadSessionTasks = async (sessionId: string) => {
        if (selectedSession === sessionId) {
            setSelectedSession(null);
            setSessionTasks([]);
            return;
        }
        setSelectedSession(sessionId);
        setLoadingTasks(true);
        try {
            const res = await fetch(`${API_BASE_URL}/tasks?session_id=${encodeURIComponent(sessionId)}&limit=20`);
            const data = await res.json();
            setSessionTasks(data);
        } finally {
            setLoadingTasks(false);
        }
    };

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
                        Sessions
                    </span>
                    <span style={{ marginLeft: 'auto', fontSize: 11, color: '#5c5c6e', fontFamily: 'JetBrains Mono, monospace' }}>
                        {sessions.length} sessions
                    </span>
                </header>

                <div style={{ maxWidth: 800, margin: '0 auto', padding: '40px 24px' }}>
                    {loading && <div style={{ color: '#5c5c6e', fontSize: 13 }}>Loading sessions...</div>}

                    {!loading && sessions.length === 0 && (
                        <div style={{ color: '#5c5c6e', fontSize: 13 }}>
                            No sessions yet. Create a task with a session_id to group tasks into a session.
                        </div>
                    )}

                    {sessions.map(session => (
                        <div key={session.session_id} style={{ marginBottom: 12 }}>
                            {/* Session header row */}
                            <div
                                onClick={() => loadSessionTasks(session.session_id)}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: 12,
                                    padding: '14px 16px', background: '#16161a',
                                    border: `1px solid ${selectedSession === session.session_id ? 'rgba(99,102,241,0.4)' : 'rgba(255,255,255,0.07)'}`,
                                    borderRadius: selectedSession === session.session_id ? '10px 10px 0 0' : 10,
                                    cursor: 'pointer',
                                }}
                            >
                                <MessageSquare size={14} color="#6366f1" />
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#6366f1', marginBottom: 4 }}>
                                        {session.session_id}
                                    </div>
                                    <div style={{ fontSize: 13, color: '#9898a8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {session.latest_task}
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: 12, flexShrink: 0 }}>
                                    <span style={{ fontSize: 11, color: '#10b981' }}>
                                        ✓ {session.statuses.completed}
                                    </span>
                                    <span style={{ fontSize: 11, color: '#ef4444' }}>
                                        ✗ {session.statuses.failed}
                                    </span>
                                    {session.statuses.running > 0 && (
                                        <span style={{ fontSize: 11, color: '#6366f1' }}>
                                            ⟳ {session.statuses.running}
                                        </span>
                                    )}
                                </div>
                                <span style={{ fontSize: 11, color: '#5c5c6e', fontFamily: 'JetBrains Mono, monospace' }}>
                                    {session.task_count} tasks
                                </span>
                                <span style={{ fontSize: 11, color: '#5c5c6e' }}>
                                    {new Date(session.last_active).toLocaleDateString()}
                                </span>
                            </div>

                            {/* Expanded task list */}
                            {selectedSession === session.session_id && (
                                <div style={{
                                    border: '1px solid rgba(99,102,241,0.2)', borderTop: 'none',
                                    borderRadius: '0 0 10px 10px', overflow: 'hidden',
                                }}>
                                    {loadingTasks ? (
                                        <div style={{ padding: '16px 20px', color: '#5c5c6e', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}>
                                            <Loader2 size={13} style={{ animation: 'spin 0.7s linear infinite' }} /> Loading tasks...
                                        </div>
                                    ) : sessionTasks.length === 0 ? (
                                        <div style={{ padding: '16px 20px', color: '#5c5c6e', fontSize: 13 }}>No tasks found.</div>
                                    ) : (
                                        sessionTasks.map((task: any, i: number) => {
                                            const statusColor = task.status === 'COMPLETED' ? '#10b981' : task.status === 'FAILED' ? '#ef4444' : '#6366f1';
                                            return (
                                                <div
                                                    key={task.task_id}
                                                    onClick={() => router.push(`/tasks/${task.task_id}`)}
                                                    style={{
                                                        display: 'flex', alignItems: 'center', gap: 12,
                                                        padding: '12px 20px', cursor: 'pointer',
                                                        borderTop: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)',
                                                        background: '#0d0d10',
                                                    }}
                                                >
                                                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: statusColor, flexShrink: 0 }} />
                                                    <span style={{ fontSize: 13, color: '#9898a8', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                        {task.user_input}
                                                    </span>
                                                    <span style={{ fontSize: 11, color: '#5c5c6e', fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
                                                        {new Date(task.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                    </span>
                                                </div>
                                            );
                                        })
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </>
    );
}
