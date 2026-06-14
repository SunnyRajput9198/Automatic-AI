'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { AlertCircle, Loader2, ArrowRight, Zap, GitBranch, Wrench, BarChart2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

const SUGGESTIONS = [
  'Research the latest papers on LLM reasoning and summarize key findings',
  'Analyze the top 10 competitors in the AI agent space',
  'Write and test a Python script to scrape HackerNews top stories',
  'Explain quantum computing with real-world analogies',
];

const HOW_IT_WORKS = [
  { icon: ArrowRight, label: 'Submit', desc: 'Describe what you want accomplished' },
  { icon: GitBranch, label: 'Plan', desc: 'Task is broken into executable steps' },
  { icon: Wrench, label: 'Execute', desc: 'Agent runs each step with available tools' },
  { icon: BarChart2, label: 'Review', desc: 'Results and step details on the task page' },
];

export default function Home() {
  const router = useRouter();
  const [taskInput, setTaskInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [taskId, setTaskId] = useState('');
  const [charCount, setCharCount] = useState(0);
  const [recentTasks, setRecentTasks] = useState<Array<{
    task_id: string;
    user_input: string;
    status: string;
    created_at: string;
  }>>([]);

  useEffect(() => {
    const fetchRecent = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/tasks?limit=5`);
        if (res.ok) {
          const data = await res.json();
          setRecentTasks(data);
        }
      } catch { }
    };
    fetchRecent();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTaskInput(e.target.value);
    setCharCount(e.target.value.length);
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    setError('');
    setTaskId('');

    if (!taskInput.trim()) {
      setError('Please enter a task prompt');
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: taskInput }),
      });

      if (!response.ok) throw new Error(`API error: ${response.statusText}`);

      const data = await response.json();

      if (data.task_id) {
        setTaskId(data.task_id);
        setTimeout(() => router.push(`/tasks/${data.task_id}`), 1000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create task. Check if the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit();
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
          --bg:        #0d0d10;
          --surface:   #16161a;
          --surface-2: #1e1e24;
          --surface-3: #26262e;
          --border:    rgba(255,255,255,0.07);
          --border-2:  rgba(255,255,255,0.13);
          --text:      #f0f0f2;
          --text-2:    #9898a8;
          --text-3:    #5c5c6e;
          --accent:    #6366f1;
          --accent-dim:rgba(99,102,241,0.12);
          --green:     #10b981;
          --red:       #ef4444;
          --r:         10px;
          --r-lg:      14px;
          --font-d:    'Sora', sans-serif;
          --font-b:    'DM Sans', sans-serif;
          --font-m:    'JetBrains Mono', monospace;
        }
        body { background: var(--bg); color: var(--text); font-family: var(--font-b); min-height: 100vh; }
        .page {
          min-height: 100vh;
          display: grid;
          grid-template-columns: 1fr minmax(0, 640px) 1fr;
          grid-template-rows: auto 1fr auto;
        }
        .topbar {
          grid-column: 1 / -1;
          height: 52px; display: flex; align-items: center;
          padding: 0 32px; border-bottom: 1px solid var(--border); gap: 10px;
        }
        .logo { display: flex; align-items: center; gap: 9px; font-family: var(--font-d); font-size: 14px; font-weight: 600; }
        .logo-icon {
          width: 28px; height: 28px; border-radius: 7px; background: var(--accent);
          display: flex; align-items: center; justify-content: center; font-size: 13px;
        }
        .topbar-badge {
          margin-left: auto; font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
          padding: 3px 9px; border-radius: 20px;
          border: 1px solid rgba(99,102,241,0.3); color: var(--accent); background: var(--accent-dim);
        }
        .center { grid-column: 2; padding: 64px 0 80px; display: flex; flex-direction: column; gap: 40px; }
        .header-eyebrow {
          font-size: 11px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase;
          color: var(--accent); margin-bottom: 14px; display: flex; align-items: center; gap: 6px;
        }
        .header-eyebrow::before { content: ''; display: inline-block; width: 16px; height: 1px; background: var(--accent); }
        .header-title {
          font-family: var(--font-d); font-size: clamp(28px, 4vw, 38px); font-weight: 700;
          line-height: 1.2; letter-spacing: -0.02em; color: var(--text); margin-bottom: 12px;
        }
        .header-sub { font-size: 14.5px; line-height: 1.7; color: var(--text-2); max-width: 520px; }
        .textarea-box {
          background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--r-lg); overflow: hidden;
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        .textarea-box:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-dim); }
        .task-textarea {
          width: 100%; background: none; border: none; outline: none;
          color: var(--text); font-family: var(--font-b); font-size: 14.5px;
          line-height: 1.7; resize: none; padding: 18px 20px 12px;
          caret-color: var(--accent); min-height: 130px;
        }
        .task-textarea::placeholder { color: var(--text-3); }
        .task-textarea:disabled { opacity: 0.5; }
        .textarea-footer {
          display: flex; align-items: center; justify-content: space-between;
          padding: 8px 16px 12px; border-top: 1px solid var(--border);
        }
        .char-count { font-size: 12px; color: var(--text-3); font-family: var(--font-m); }
        .hint { font-size: 12px; color: var(--text-3); }
        .hint kbd {
          background: var(--surface-3); border: 1px solid var(--border-2); border-radius: 4px;
          padding: 1px 5px; font-family: var(--font-m); font-size: 10px; color: var(--text-2);
        }
        .suggestions { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }
        .pill {
          padding: 6px 13px; border-radius: 20px; border: 1px solid var(--border-2);
          background: var(--surface); color: var(--text-2); font-size: 12px; font-family: var(--font-b);
          cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s; white-space: nowrap;
        }
        .pill:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
        .pill:disabled { opacity: 0.4; cursor: not-allowed; }
        .banner {
          display: flex; align-items: flex-start; gap: 10px; padding: 12px 16px;
          border-radius: var(--r); font-size: 13px; line-height: 1.5; margin-top: 12px;
        }
        .banner-error { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: var(--red); }
        .banner-success { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); color: var(--green); }
        .banner-icon { flex-shrink: 0; margin-top: 1px; }
        .task-id-code {
          display: inline-block; margin-top: 4px; font-family: var(--font-m); font-size: 12px;
          background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 4px; color: var(--text-2);
        }
        .submit-btn {
          width: 100%; margin-top: 14px;
          display: flex; align-items: center; justify-content: center; gap: 8px;
          background: var(--accent); color: #fff; border: none; border-radius: var(--r);
          padding: 13px 24px; font-family: var(--font-d); font-size: 14px; font-weight: 600;
          letter-spacing: 0.01em; cursor: pointer; transition: opacity 0.15s, transform 0.1s;
        }
        .submit-btn:hover:not(:disabled) { opacity: 0.87; }
        .submit-btn:active:not(:disabled) { transform: scale(0.985); }
        .submit-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .how-divider { border: none; border-top: 1px solid var(--border); margin-bottom: 24px; }
        .how-title {
          font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
          color: var(--text-3); margin-bottom: 16px;
        }
        .how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .how-card {
          background: var(--surface); border: 1px solid var(--border); border-radius: var(--r);
          padding: 16px 14px; display: flex; flex-direction: column; gap: 8px;
        }
        .how-card-icon {
          width: 30px; height: 30px; border-radius: 8px; background: var(--accent-dim);
          color: var(--accent); display: flex; align-items: center; justify-content: center;
        }
        .how-card-label { font-family: var(--font-d); font-size: 13px; font-weight: 600; color: var(--text); }
        .how-card-desc { font-size: 12px; line-height: 1.5; color: var(--text-3); }
        .history-title {
          font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
          color: var(--text-3); margin-bottom: 12px;
        }
        .history-item {
          display: flex; align-items: center; gap: 12px; padding: 12px 14px;
          background: var(--surface); border: 1px solid var(--border); border-radius: var(--r);
          cursor: pointer; transition: border-color 0.15s; margin-bottom: 6px;
        }
        .history-item:hover { border-color: var(--border-2); }
        .history-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
        .history-input {
          font-size: 13px; color: var(--text); flex: 1;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .history-time { font-size: 11px; color: var(--text-3); font-family: var(--font-m); flex-shrink: 0; }
        .footer {
          grid-column: 1 / -1; border-top: 1px solid var(--border); padding: 16px 32px;
          display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-3);
        }
        .footer-dot { width: 3px; height: 3px; border-radius: 50%; background: var(--text-3); }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 700px) {
          .page { grid-template-columns: 1fr; }
          .center { padding: 40px 20px 60px; }
          .topbar { padding: 0 20px; }
          .how-grid { grid-template-columns: 1fr 1fr; }
        }
      `}</style>
      <div className="page">
        <header className="topbar">
          <div className="logo">
            <div className="logo-icon">⚡</div>
            Agent
          </div>
          <button
            onClick={() => router.push('/analytics')}
            style={{
              marginLeft: 'auto', background: 'none', border: '1px solid rgba(255,255,255,0.1)',
              color: '#9898a8', cursor: 'pointer', fontSize: 12, padding: '4px 12px',
              borderRadius: 6,
            }}
          >
            Analytics
          </button>
          <button
            onClick={() => router.push('/files')}
            style={{
              background: 'none', border: '1px solid rgba(255,255,255,0.1)',
              color: '#9898a8', cursor: 'pointer', fontSize: 12,
              padding: '4px 12px', borderRadius: 6, marginLeft: 8,
            }}
          >
            Files
          </button>
          <span className="topbar-badge">Autonomous</span>
        </header>

        <div className="center">
          <div>
            <div className="header-eyebrow">Autonomous Agent System</div>
            <h1 className="header-title">What should the<br />agent accomplish?</h1>
            <p className="header-sub">
              Submit a task and the agent will plan, execute, and return results — using tools, search, and multi-step reasoning.
            </p>
          </div>

          <div>
            <div className="textarea-box">
              <textarea
                className="task-textarea"
                placeholder="Describe your task in detail. The more specific, the better the result…"
                value={taskInput}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
                rows={5}
              />
              <div className="textarea-footer">
                <span className="char-count">{charCount} chars</span>
                <span className="hint">
                  <kbd>⌘</kbd> + <kbd>↵</kbd> to submit
                </span>
              </div>
            </div>

            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="pill"
                  onClick={() => { setTaskInput(s); setCharCount(s.length); }}
                  disabled={isLoading}
                >
                  {s.length > 48 ? s.slice(0, 46) + '…' : s}
                </button>
              ))}
            </div>

            {error && (
              <div className="banner banner-error">
                <AlertCircle size={16} className="banner-icon" />
                <span>{error}</span>
              </div>
            )}

            {taskId && (
              <div className="banner banner-success">
                <Zap size={16} className="banner-icon" />
                <div>
                  Task created — redirecting…
                  <div><span className="task-id-code">{taskId}</span></div>
                </div>
              </div>
            )}

            <button className="submit-btn" onClick={handleSubmit} disabled={isLoading || !taskInput.trim()}>
              {isLoading ? (
                <>
                  <Loader2 size={16} style={{ animation: 'spin 0.7s linear infinite' }} />
                  Creating task…
                </>
              ) : (
                <>
                  Run Task
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>

          <div>
            <hr className="how-divider" />
            <div className="how-title">How it works</div>
            <div className="how-grid">
              {HOW_IT_WORKS.map(({ icon: Icon, label, desc }) => (
                <div key={label} className="how-card">
                  <div className="how-card-icon"><Icon size={15} /></div>
                  <div className="how-card-label">{label}</div>
                  <div className="how-card-desc">{desc}</div>
                </div>
              ))}
            </div>
          </div>

          {recentTasks.length > 0 && (
            <div>
              <hr className="how-divider" />
              <div className="history-title">Recent Tasks</div>
              {recentTasks.map((t) => {
                const statusColor = t.status === 'COMPLETED' ? '#10b981' : t.status === 'FAILED' ? '#ef4444' : '#6366f1';
                return (
                  <div
                    key={t.task_id}
                    className="history-item"
                    onClick={() => router.push(`/tasks/${t.task_id}`)}
                  >
                    <div className="history-dot" style={{ background: statusColor }} />
                    <span className="history-input">{t.user_input}</span>
                    <span className="history-time">
                      {new Date(t.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <footer className="footer">
          <span>Autonomous Agent</span>
          <div className="footer-dot" />
          <span>Powered by Claude + Groq</span>
          <div className="footer-dot" />
          <span style={{ marginLeft: 'auto' }}>v1.0</span>
        </footer>
      </div>
    </>
  );
}