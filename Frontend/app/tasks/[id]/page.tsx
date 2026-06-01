'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
  ArrowLeft, CheckCircle2, XCircle, Loader2, Clock,
  ChevronDown, ChevronUp, Zap, GitBranch, Wrench, BarChart2
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

type StepStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'RETRYING' | 'SKIPPED';
type TaskStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';

interface Step {
  id: string;
  step_number: number;
  instruction: string;
  status: StepStatus;
  result: string | null;
  error: string | null;
  retry_count: number;
}

interface Task {
  task_id: string;
  user_input: string;
  status: TaskStatus;
  created_at: string;
  steps: Step[];
}

const STATUS_COLOR: Record<string, string> = {
  PENDING: '#9898a8',
  RUNNING: '#6366f1',
  COMPLETED: '#10b981',
  FAILED: '#ef4444',
  RETRYING: '#f59e0b',
  SKIPPED: '#9898a8',
};

const STATUS_ICON = {
  PENDING: <Clock size={14} />,
  RUNNING: <Loader2 size={14} style={{ animation: 'spin 0.7s linear infinite' }} />,
  COMPLETED: <CheckCircle2 size={14} />,
  FAILED: <XCircle size={14} />,
  RETRYING: <Loader2 size={14} style={{ animation: 'spin 0.7s linear infinite' }} />,
  SKIPPED: <Clock size={14} />,
};


function StepCard({ step }: { step: Step }) {
  const [expanded, setExpanded] = useState(step.status === 'COMPLETED' || step.status === 'FAILED');
  const color = STATUS_COLOR[step.status] || '#9898a8';
  return (
    <div style={{
      background: '#16161a',
      border: `1px solid rgba(255,255,255,0.07)`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 10,
      overflow: 'hidden',
    }}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '14px 16px', cursor: 'pointer',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <span style={{ color, flexShrink: 0 }}>{STATUS_ICON[step.status]}</span>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
          color: '#5c5c6e', flexShrink: 0
        }}>
          {String(step.step_number).padStart(2, '0')}
        </span>
        <span style={{ fontSize: 13.5, color: '#f0f0f2', flex: 1, lineHeight: 1.5 }}>
          {step.instruction}
        </span>
        {step.retry_count > 0 && (
          <span style={{
            fontSize: 11, padding: '2px 8px', borderRadius: 20,
            background: 'rgba(245,158,11,0.1)', color: '#f59e0b',
            border: '1px solid rgba(245,158,11,0.2)', flexShrink: 0
          }}>
            {step.retry_count} retr{step.retry_count === 1 ? 'y' : 'ies'}
          </span>
        )}
        <span style={{ color: '#5c5c6e', flexShrink: 0 }}>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </div>

      {expanded && (step.result || step.error) && (
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', padding: '14px 16px' }}>
          {step.result && (
            <pre style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
              color: '#9898a8', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              lineHeight: 1.7, margin: 0,
            }}>
              {step.result}
            </pre>
          )}
          {step.error && (
            <pre style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
              color: '#ef4444', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              lineHeight: 1.7, margin: 0,
            }}>
              {step.error}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
function extractFilesFromSteps(steps: Step[]): string[] {
  const files: string[] = [];
  for (const step of steps) {
    if (step.result) {
      const match = step.result.match(/wrote \d+ characters to (.+)/);
      if (match) files.push(match[1].trim());
    }
  }
  return files;
}

export default function TaskPage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.id as string;

  const [task, setTask] = useState<Task | null>(null);
  const [error, setError] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [wsEvents, setWsEvents] = useState<Array<{ phase: string, status: string, [key: string]: any }>>([]);
  const wsRef = React.useRef<WebSocket | null>(null);
  const [selectedFile, setSelectedFile] = useState<{ name: string, content: string } | null>(null);

  const openFile = async (filename: string) => {
    const res = await fetch(`${API_BASE_URL}/files/${filename}`);
    if (res.ok) {
      const content = await res.text();
      setSelectedFile({ name: filename, content });
    }
  };
  const fetchTask = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/tasks/${taskId}`);
      if (!res.ok) throw new Error('Task not found');
      const data = await res.json();
      setTask(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load task');
    }
  }, [taskId]);
  const cancelTask = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/tasks/${taskId}/cancel`, {
        method: 'POST',
      });
      if (res.ok) {
        fetchTask();
      }
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  };

  const taskFiles = task ? extractFilesFromSteps(task.steps) : [];

  useEffect(() => {
    fetchTask();

    // Connect WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/ws/${taskId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setWsEvents(prev => [...prev, data]);

      // Refresh task data when step completes or task completes
      if (data.phase === 'step' && data.status === 'completed') {
        fetchTask();
      }
      if (data.phase === 'completed' || data.phase === 'failed') {
        fetchTask();
      }
    };

    ws.onerror = () => console.log('WebSocket error');
    ws.onclose = () => console.log('WebSocket closed');

    // Keep polling as fallback
    const interval = setInterval(() => {
      if (task?.status === 'COMPLETED' || task?.status === 'FAILED') {
        clearInterval(interval);
        ws.close();
        return;
      }
      fetchTask();
    }, 5000);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [taskId, fetchTask]);

  // Elapsed timer
  useEffect(() => {
    if (task?.status === 'RUNNING' || task?.status === 'PENDING') {
      const timer = setInterval(() => setElapsed(e => e + 1), 1000);
      return () => clearInterval(timer);
    }
  }, [task?.status]);

  const isRunning = task?.status === 'PENDING' || task?.status === 'RUNNING';

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d10; color: #f0f0f2; font-family: 'DM Sans', sans-serif; min-height: 100vh; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
      `}</style>

      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

        {/* Topbar */}
        <header style={{
          height: 52, display: 'flex', alignItems: 'center',
          padding: '0 32px', borderBottom: '1px solid rgba(255,255,255,0.07)', gap: 16,
        }}>
          <button
            onClick={() => router.push('/')}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'none', border: 'none', color: '#9898a8',
              cursor: 'pointer', fontSize: 13, padding: '4px 8px',
              borderRadius: 6, transition: 'color 0.15s',
            }}
          >
            <ArrowLeft size={14} /> Back
          </button>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
            color: '#5c5c6e', marginLeft: 'auto'
          }}>
            {taskId}
          </div>
        </header>

        {/* Content */}
        <div style={{ flex: 1, maxWidth: 720, margin: '0 auto', width: '100%', padding: '48px 24px' }}>

          {error && (
            <div style={{
              padding: '12px 16px', borderRadius: 10,
              background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
              color: '#ef4444', fontSize: 13,
            }}>
              {error}
            </div>
          )}

          {task && (
            <>
              {/* Task header */}
              <div style={{ marginBottom: 32 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <span style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    fontSize: 12, fontWeight: 500, padding: '4px 10px',
                    borderRadius: 20, border: `1px solid ${STATUS_COLOR[task.status]}40`,
                    color: STATUS_COLOR[task.status],
                    background: `${STATUS_COLOR[task.status]}12`,
                  }}>
                    {STATUS_ICON[task.status]}
                    {task.status}
                  </span>
                  {isRunning && (
                    <span style={{ fontSize: 12, color: '#5c5c6e', fontFamily: 'JetBrains Mono, monospace' }}>
                      {elapsed}s elapsed
                    </span>
                  )}
                  {isRunning && (
                    <button
                      onClick={cancelTask}
                      style={{
                        marginLeft: 'auto',
                        display: 'flex', alignItems: 'center', gap: 6,
                        background: 'rgba(239,68,68,0.08)',
                        border: '1px solid rgba(239,68,68,0.2)',
                        color: '#ef4444', cursor: 'pointer',
                        fontSize: 12, fontWeight: 500,
                        padding: '5px 12px', borderRadius: 8,
                      }}
                    >
                      <XCircle size={13} />
                      Cancel
                    </button>
                  )}
                </div>
                <h1 style={{
                  fontFamily: 'Sora, sans-serif', fontSize: 22,
                  fontWeight: 700, lineHeight: 1.3, color: '#f0f0f2',
                }}>
                  {task.user_input}
                </h1>
              </div>

              {/* Progress bar */}
              {task.steps.length > 0 && (
                <div style={{ marginBottom: 24 }}>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    fontSize: 12, color: '#5c5c6e', marginBottom: 8,
                  }}>
                    <span>{task.steps.filter(s => s.status === 'COMPLETED').length} / {task.steps.length} steps completed</span>
                    {task.steps.some(s => s.retry_count > 0) && (
                      <span style={{ color: '#f59e0b' }}>
                        {task.steps.reduce((a, s) => a + s.retry_count, 0)} retries
                      </span>
                    )}
                  </div>
                  <div style={{ height: 3, background: 'rgba(255,255,255,0.07)', borderRadius: 2 }}>
                    <div style={{
                      height: '100%', borderRadius: 2,
                      background: task.status === 'FAILED' ? '#ef4444' : '#6366f1',
                      width: `${(task.steps.filter(s => s.status === 'COMPLETED').length / task.steps.length) * 100}%`,
                      transition: 'width 0.5s ease',
                    }} />
                  </div>
                </div>
              )}

              {/* Steps */}
              {task.steps.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {task.steps.map(step => (
                    <StepCard key={step.id} step={step} />
                  ))}
                </div>
              ) : (
                isRunning && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '20px 16px', borderRadius: 10,
                    background: '#16161a', border: '1px solid rgba(255,255,255,0.07)',
                  }}>
                    <Loader2 size={16} color="#6366f1" style={{ animation: 'spin 0.7s linear infinite' }} />
                    <span style={{ fontSize: 13, color: '#9898a8' }}>Agent is planning your task…</span>
                  </div>
                )
              )}
              {wsEvents.length > 0 && (
                <div style={{ marginTop: 24 }}>
                  <div style={{
                    fontSize: 11, fontWeight: 500, letterSpacing: '0.08em',
                    textTransform: 'uppercase', color: '#5c5c6e', marginBottom: 10
                  }}>
                    Live Events
                  </div>
                  <div style={{
                    background: '#0d0d10', border: '1px solid rgba(255,255,255,0.07)',
                    borderRadius: 10, padding: '12px 16px', fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 11, color: '#5c5c6e', maxHeight: 200, overflowY: 'auto'
                  }}>
                    {wsEvents.map((e, i) => (
                      <div key={i} style={{ marginBottom: 4 }}>
                        <span style={{ color: '#6366f1' }}>{e.phase}</span>
                        {' → '}
                        <span style={{ color: e.status === 'completed' ? '#10b981' : e.status === 'failed' ? '#ef4444' : '#f59e0b' }}>
                          {e.status}
                        </span>
                        {e.step_number && <span> (step {e.step_number})</span>}
                        {e.problem_type && <span style={{ color: '#9898a8' }}> [{e.problem_type}]</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
          {/* Files Section */}
          {taskFiles.length > 0 && (
            <div
              style={{
                marginTop: 32,
                padding: 20,
                borderRadius: 12,
                background: '#16161a',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <h2
                style={{
                  fontSize: 16,
                  fontWeight: 600,
                  marginBottom: 16,
                  color: '#f0f0f2',
                }}
              >
                Generated Files
              </h2>

              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                }}
              >
                {taskFiles.map((file, index) => (
                  <a
                    key={index}
                    onClick={(e) => { e.preventDefault(); openFile(file); }}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '12px 16px',
                      borderRadius: 10,
                      background: '#0d0d10',
                      border: '1px solid rgba(255,255,255,0.06)',
                      textDecoration: 'none',
                      color: '#f0f0f2',
                      transition: '0.2s',
                    }}
                  >
                    <span
                      style={{
                        fontSize: 13,
                        wordBreak: 'break-all',
                      }}
                    >
                      {file}
                    </span>

                    <span
                      style={{
                        fontSize: 12,
                        color: '#6366f1',
                      }}
                    >
                      Open
                    </span>
                  </a>
                ))}
              </div>
            </div>
          )}
          {!task && !error && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#9898a8' }}>
              <Loader2 size={16} style={{ animation: 'spin 0.7s linear infinite' }} />
              <span style={{ fontSize: 13 }}>Loading task…</span>
            </div>
          )}
          {selectedFile && (
            <div style={{
              position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              zIndex: 100, padding: 24,
            }} onClick={() => setSelectedFile(null)}>
              <div style={{
                background: '#16161a', border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 12, width: '100%', maxWidth: 720, maxHeight: '80vh',
                overflow: 'hidden', display: 'flex', flexDirection: 'column',
              }} onClick={e => e.stopPropagation()}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.07)',
                }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#f0f0f2' }}>
                    {selectedFile.name}
                  </span>
                  <button onClick={() => setSelectedFile(null)} style={{
                    background: 'none', border: 'none', color: '#9898a8',
                    cursor: 'pointer', fontSize: 18, lineHeight: 1,
                  }}>×</button>
                </div>
                <pre style={{
                  padding: 20, overflow: 'auto', fontFamily: 'JetBrains Mono, monospace',
                  fontSize: 12, color: '#9898a8', lineHeight: 1.7, margin: 0,
                  whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                }}>
                  {selectedFile.content}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}