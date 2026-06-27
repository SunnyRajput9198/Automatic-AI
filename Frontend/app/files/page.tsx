'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Search, Eye, Trash2 } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

export default function FilesPage() {
    const router = useRouter();
    const [files, setFiles] = useState<string[]>([]);
    const [search, setSearch] = useState('');
    const [selectedFile, setSelectedFile] = useState<{ name: string; content: string } | null>(null);
    const [loading, setLoading] = useState(true);
    const [deleting, setDeleting] = useState<string | null>(null);

    const loadFiles = () => {
        fetch(`${API_BASE_URL}/files`)
            .then(r => r.json())
            .then(d => { setFiles(d.files || []); setLoading(false); })
            .catch(() => setLoading(false));
    };

    useEffect(() => { loadFiles(); }, []);

    const openFile = async (filename: string) => {
        const res = await fetch(`${API_BASE_URL}/files/${filename}`);
        if (res.ok) {
            const content = await res.text();
            setSelectedFile({ name: filename, content });
        }
    };

    const deleteFile = async (filename: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm(`Delete ${filename}?`)) return;
        setDeleting(filename);
        try {
            const res = await fetch(`${API_BASE_URL}/files/${filename}`, { method: 'DELETE' });
            if (res.ok) {
                setFiles(prev => prev.filter(f => f !== filename));
                if (selectedFile?.name === filename) setSelectedFile(null);
            }
        } finally {
            setDeleting(null);
        }
    };

    const filtered = files.filter(f => f.toLowerCase().includes(search.toLowerCase()));

    const getFileIcon = (filename: string) => {
        if (filename.endsWith('.py')) return '🐍';
        if (filename.endsWith('.json')) return '📋';
        if (filename.endsWith('.md')) return '📝';
        if (filename.endsWith('.txt')) return '📄';
        return '📁';
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
                        background: 'none', border: 'none', color: '#9898a8',
                        cursor: 'pointer', fontSize: 13,
                    }}>
                        <ArrowLeft size={14} /> Back
                    </button>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#f0f0f2', fontFamily: 'Sora, sans-serif' }}>
                        Workspace Files
                    </span>
                    <span style={{ marginLeft: 'auto', fontSize: 11, color: '#5c5c6e', fontFamily: 'JetBrains Mono, monospace' }}>
                        {files.length} files
                    </span>
                </header>

                <div style={{ maxWidth: 800, margin: '0 auto', padding: '40px 24px' }}>
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        background: '#16161a', border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: 10, padding: '10px 16px', marginBottom: 24,
                    }}>
                        <Search size={14} color="#5c5c6e" />
                        <input
                            type="text"
                            placeholder="Search files..."
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                            style={{ background: 'none', border: 'none', outline: 'none', color: '#f0f0f2', fontSize: 14, flex: 1 }}
                        />
                    </div>

                    {loading ? (
                        <div style={{ color: '#5c5c6e', fontSize: 13 }}>Loading files...</div>
                    ) : filtered.length === 0 ? (
                        <div style={{ color: '#5c5c6e', fontSize: 13 }}>No files found.</div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {filtered.map((file, i) => (
                                <div key={i} style={{
                                    display: 'flex', alignItems: 'center', gap: 12,
                                    padding: '12px 16px', background: '#16161a',
                                    border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10,
                                    cursor: 'pointer',
                                }} onClick={() => openFile(file)}>
                                    <span style={{ fontSize: 16 }}>{getFileIcon(file)}</span>
                                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#f0f0f2', flex: 1 }}>
                                        {file}
                                    </span>
                                    <Eye size={14} color="#5c5c6e" />
                                    <button
                                        onClick={e => deleteFile(file, e)}
                                        disabled={deleting === file}
                                        style={{
                                            background: 'none', border: 'none', cursor: 'pointer',
                                            color: deleting === file ? '#5c5c6e' : '#ef4444',
                                            display: 'flex', alignItems: 'center', padding: '2px 4px',
                                        }}
                                        title="Delete file"
                                    >
                                        <Trash2 size={13} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {selectedFile && (
                    <div style={{
                        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100, padding: 24,
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
                                    background: 'none', border: 'none', color: '#9898a8', cursor: 'pointer', fontSize: 18,
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
        </>
    );
}

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

export default function FilesPage() {
    const router = useRouter();
    const [files, setFiles] = useState<string[]>([]);
    const [search, setSearch] = useState('');
    const [selectedFile, setSelectedFile] = useState<{ name: string; content: string } | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_BASE_URL}/files`)
            .then(r => r.json())
            .then(d => { setFiles(d.files || []); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    const openFile = async (filename: string) => {
        const res = await fetch(`${API_BASE_URL}/files/${filename}`);
        if (res.ok) {
            const content = await res.text();
            setSelectedFile({ name: filename, content });
        }
    };

    const filtered = files.filter(f => f.toLowerCase().includes(search.toLowerCase()));

    const getFileIcon = (filename: string) => {
        if (filename.endsWith('.py')) return '🐍';
        if (filename.endsWith('.json')) return '📋';
        if (filename.endsWith('.md')) return '📝';
        if (filename.endsWith('.txt')) return '📄';
        return '📁';
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
                        background: 'none', border: 'none', color: '#9898a8',
                        cursor: 'pointer', fontSize: 13,
                    }}>
                        <ArrowLeft size={14} /> Back
                    </button>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#f0f0f2', fontFamily: 'Sora, sans-serif' }}>
                        Workspace Files
                    </span>
                    <span style={{
                        marginLeft: 'auto', fontSize: 11, color: '#5c5c6e',
                        fontFamily: 'JetBrains Mono, monospace'
                    }}>
                        {files.length} files
                    </span>
                </header>

                <div style={{ maxWidth: 800, margin: '0 auto', padding: '40px 24px' }}>
                    {/* Search */}
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        background: '#16161a', border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: 10, padding: '10px 16px', marginBottom: 24,
                    }}>
                        <Search size={14} color="#5c5c6e" />
                        <input
                            type="text"
                            placeholder="Search files..."
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                            style={{
                                background: 'none', border: 'none', outline: 'none',
                                color: '#f0f0f2', fontSize: 14, flex: 1,
                            }}
                        />
                    </div>

                    {/* File list */}
                    {loading ? (
                        <div style={{ color: '#5c5c6e', fontSize: 13 }}>Loading files...</div>
                    ) : filtered.length === 0 ? (
                        <div style={{ color: '#5c5c6e', fontSize: 13 }}>No files found.</div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {filtered.map((file, i) => (
                                <div key={i} style={{
                                    display: 'flex', alignItems: 'center', gap: 12,
                                    padding: '12px 16px', background: '#16161a',
                                    border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10,
                                    cursor: 'pointer', transition: 'border-color 0.15s',
                                }}
                                    onClick={() => openFile(file)}
                                >
                                    <span style={{ fontSize: 16 }}>{getFileIcon(file)}</span>
                                    <span style={{
                                        fontFamily: 'JetBrains Mono, monospace',
                                        fontSize: 13, color: '#f0f0f2', flex: 1,
                                    }}>
                                        {file}
                                    </span>
                                    <Eye size={14} color="#5c5c6e" />
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* File preview modal */}
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
                                    cursor: 'pointer', fontSize: 18,
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
        </>
    );
}