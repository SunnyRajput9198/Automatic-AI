'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

import {
  AlertCircle,
  ArrowLeft,
  RefreshCw,
  Loader2,
  CheckCircle2,
  Clock,
  AlertTriangle,
} from 'lucide-react';

interface Step {
  id: string;
  step_number: number;
  instruction: string;
  status: string;
  result?: string;
  error?: string;
  retry_count: number;
}

interface TaskData {
  task_id: string;
  user_input: string;
  status: string;
  created_at: string;
  steps: Step[];
}
const API_BASE_URL = process.env.BACKEND_URL || 'http://localhost:8000/api/v1';
export default function TaskStatusPage() {
  const params = useParams();
  const taskId = params.taskId as string;

  const [taskData, setTaskData] = useState<TaskData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchTaskData = async (refresh = false) => {
    if (refresh) setIsRefreshing(true);

    try {
      const res = await fetch(
        `${API_BASE_URL}/tasks/${taskId}`
      );

      if (!res.ok) {
        throw new Error(`Failed to fetch task`);
      }

      const data = await res.json();
      setTaskData(data);
      setLastRefresh(new Date());
      setError('');
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to load task'
      );
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchTaskData();

    const interval = setInterval(() => {
      if (taskData?.status === 'RUNNING') {
        fetchTaskData(false);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId, taskData?.status]);

  const statusIcon = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle2 className="h-5 w-5 text-green-500" />;
      case 'RUNNING':
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
      case 'PENDING':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
    }
  };

  if (isLoading) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </main>
    );
  }

  return (
    <main className="min-h-screen p-4 bg-background">
      <div className="max-w-4xl mx-auto space-y-6">

        <Link href="/">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
        </Link>

        {error && (
          <Card className="p-4 border-destructive bg-destructive/10 flex gap-3">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <p className="text-sm text-destructive">{error}</p>
          </Card>
        )}

        {taskData && (
          <>
            {/* Task Header */}
            <Card className="p-6 space-y-4">
              <div className="flex justify-between items-start">
                <div>
                  <h1 className="text-2xl font-bold">
                    Task {taskData.task_id}
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    {taskData.user_input}
                  </p>
                </div>

                <Badge className="flex gap-2 px-3 py-1">
                  {statusIcon(taskData.status)}
                  {taskData.status}
                </Badge>
              </div>

              <div className="flex justify-between text-xs text-muted-foreground border-t pt-3">
                <span>
                  {lastRefresh &&
                    `Last updated: ${lastRefresh.toLocaleTimeString()}`}
                </span>

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => fetchTaskData(true)}
                  disabled={isRefreshing}
                >
                  {isRefreshing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </Card>

            {/* Steps */}
            <Card className="p-6 space-y-4">
              <h2 className="text-lg font-semibold">
                Steps ({taskData.steps.length})
              </h2>

              {taskData.steps.map(step => (
                <div
                  key={step.id}
                  className="border rounded-lg p-4 space-y-2"
                >
                  <div className="flex items-center gap-2">
                    {statusIcon(step.status)}
                    <span className="font-medium">
                      Step {step.step_number}
                    </span>
                  </div>

                  <p className="text-sm text-muted-foreground">
                    {step.instruction}
                  </p>

                  {step.result && (
                    <div className="bg-muted p-3 rounded text-xs font-mono whitespace-pre-wrap wrap-break-words">
                      {step.result}
                    </div>
                  )}

                  {step.error && (
                    <div className="text-sm text-destructive">
                      {step.error}
                    </div>
                  )}
                </div>
              ))}
            </Card>
          </>
        )}
      </div>
    </main>
  );
}
