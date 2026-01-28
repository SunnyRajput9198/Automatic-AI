'use client';

import React from "react"

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { AlertCircle, Loader2 } from 'lucide-react';

const API_BASE_URL = process.env.BACKEND_URL || 'http://localhost:8000/api/v1';
export default function Home() {
  const router = useRouter();
  const [taskInput, setTaskInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [taskId, setTaskId] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ task: taskInput }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.task_id) {
        setTaskId(data.task_id);
        // Optionally redirect to task page
        setTimeout(() => {
          router.push(`/tasks/${data.task_id}`);
        }, 1000);
      }
    } catch (err) {
      setError(
        err instanceof Error 
          ? err.message 
          : 'Failed to create task. Check if the backend is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background text-foreground flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        <div className="space-y-8">
          {/* Header */}
          <div className="space-y-2">
            <h1 className="text-4xl font-bold tracking-tight">
              Autonomous Agent System
            </h1>
            <p className="text-muted-foreground text-base leading-relaxed">
              Submit a task prompt to execute on the autonomous agent system. The agent will break down your task into steps, execute them using available tools, and return the results.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="task" className="text-sm font-medium">
                Task Prompt
              </label>
              <Textarea
                id="task"
                placeholder="Enter your task prompt here. Be specific about what you want the agent to accomplish..."
                value={taskInput}
                onChange={(e) => setTaskInput(e.target.value)}
                disabled={isLoading}
                className="min-h-32 resize-none"
              />
            </div>

            {/* Error Message */}
            {error && (
              <Card className="p-3 bg-destructive/10 border-destructive/20 flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <p className="text-sm text-destructive">{error}</p>
              </Card>
            )}

            {/* Task ID Display */}
            {taskId && (
              <Card className="p-3 bg-accent/10 border-accent/20 flex gap-3">
                <div className="flex-1">
                  <p className="text-sm font-medium text-accent-foreground">
                    Task created successfully
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Task ID: <code className="bg-muted px-2 py-1 rounded text-foreground">{taskId}</code>
                  </p>
                </div>
              </Card>
            )}

            {/* Submit Button */}
            <Button
              type="submit"
              disabled={isLoading}
              className="w-full"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating Task...
                </>
              ) : (
                'Run Task'
              )}
            </Button>
          </form>

          {/* Info Section */}
          <div className="border-t border-border pt-6">
            <h3 className="text-sm font-semibold mb-3">How it works</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex gap-2">
                <span className="text-accent">→</span>
                <span>Submit a task prompt describing what you want the agent to accomplish</span>
              </li>
              <li className="flex gap-2">
                <span className="text-accent">→</span>
                <span>The system breaks down your task into steps</span>
              </li>
              <li className="flex gap-2">
                <span className="text-accent">→</span>
                <span>Agents execute steps using available tools</span>
              </li>
              <li className="flex gap-2">
                <span className="text-accent">→</span>
                <span>View the results and step details on the task status page</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}
