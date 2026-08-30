"use client";

import { useEffect, useRef, useState } from "react";
import { Badge, Button, Card, Label, Select, Spinner, StatTile } from "@/components/ui";
import { ApiError, detectColumns, getMLRun, trainModel } from "@/lib/api";
import { coerceRowTypes, parseCsv } from "@/lib/csv";
import type { MLRun } from "@/lib/types";

type Task = "predictive" | "uplift";

// A training run outlives the component that started it (it's a real backend job),
// but the component's state doesn't — navigating to another page and back used to
// show a blank form with no sign a run was ever in progress. Persisting just the
// run id lets us reconnect to it and resume polling on remount.
const ACTIVE_RUN_KEY = "abtesting_active_ml_run";

function pollRun(id: string, onUpdate: (run: MLRun) => void, pollRef: React.MutableRefObject<ReturnType<typeof setInterval> | null>) {
  pollRef.current = setInterval(async () => {
    try {
      const updated = await getMLRun(id);
      onUpdate(updated);
      if (updated.status === "done" || updated.status === "failed") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch {
      if (pollRef.current) clearInterval(pollRef.current);
    }
  }, 2000);
}

export default function MLStudioPage() {
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [task, setTask] = useState<Task>("predictive");
  const [modelType, setModelType] = useState<"auto" | "classification" | "regression">("auto");
  const [run, setRun] = useState<MLRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const storedId = window.localStorage.getItem(ACTIVE_RUN_KEY);
    if (storedId) {
      getMLRun(storedId)
        .then((existing) => {
          if (cancelled) return;
          setRun(existing);
          if (existing.status === "pending" || existing.status === "running") {
            pollRun(storedId, setRun, pollRef);
          }
        })
        .catch(() => window.localStorage.removeItem(ACTIVE_RUN_KEY));
    }
    return () => {
      cancelled = true;
      // Intentionally reads the live ref, not a value captured when this effect ran:
      // pollRef is shared with handleSubmit's later poll too, and unmounting should
      // clear whichever poll is actually active, not just the one this effect started.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function handleFile(file: File) {
    const text = await file.text();
    const parsed = coerceRowTypes(parseCsv(text));
    setRows(parsed);
    const cols = parsed.length > 0 ? Object.keys(parsed[0]) : [];
    setColumns(cols);
    setRun(null);
    window.localStorage.removeItem(ACTIVE_RUN_KEY);

    try {
      const detection = await detectColumns(parsed);
      setGroupCol(detection.potential_group_cols?.[0] ?? "");
      setTargetCol(detection.potential_target_cols?.[0] ?? cols[cols.length - 1] ?? "");
    } catch {
      setGroupCol(cols[0] ?? "");
      setTargetCol(cols[cols.length - 1] ?? "");
    }
  }

  async function handleSubmit() {
    if (!rows || !targetCol) return;
    setError(null);
    setSubmitting(true);
    setRun(null);
    try {
      const created = await trainModel({
        rows,
        target_col: targetCol,
        group_col: groupCol || undefined,
        model_type: modelType,
        task,
      });
      setRun(created);
      window.localStorage.setItem(ACTIVE_RUN_KEY, created.id);
      pollRun(created.id, setRun, pollRef);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <h1 className="text-2xl font-semibold">ML Model Studio</h1>
      <p className="mt-1 text-sm text-muted">
        Train Gradient Boosting and Random Forest models, or run T-learner uplift modeling to see who benefits most.
      </p>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="space-y-4">
          <div>
            <Label>Dataset (CSV)</Label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              className="block w-full text-sm text-muted file:mr-4 file:rounded-lg file:border-0 file:bg-accent file:px-4 file:py-2 file:text-sm file:font-medium file:text-accent-foreground"
            />
          </div>

          {rows && (
            <>
              <p className="text-xs text-muted">{rows.length.toLocaleString()} rows loaded.</p>

              <div>
                <Label>Task</Label>
                <Select value={task} onChange={(e) => setTask(e.target.value as Task)}>
                  <option value="predictive">Predictive model (train on target column)</option>
                  <option value="uplift">Uplift model (who benefits from treatment?)</option>
                </Select>
              </div>

              <div>
                <Label>Target column</Label>
                <Select value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </Select>
              </div>

              <div>
                <Label>{task === "uplift" ? "Treatment column (required)" : "Group column (optional, excluded as a feature)"}</Label>
                <Select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                  <option value="">None</option>
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </Select>
              </div>

              {task === "predictive" && (
                <div>
                  <Label>Model type</Label>
                  <Select value={modelType} onChange={(e) => setModelType(e.target.value as typeof modelType)}>
                    <option value="auto">Auto-detect</option>
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </Select>
                </div>
              )}

              {error && <p className="text-sm text-danger">{error}</p>}
              <Button onClick={handleSubmit} disabled={submitting || (task === "uplift" && !groupCol)} className="w-full">
                {submitting ? "Starting..." : "Train model"}
              </Button>
            </>
          )}
        </Card>

        <div>{run && <RunStatus run={run} />}</div>
      </div>
    </>
  );
}

function RunStatus({ run }: { run: MLRun }) {
  if (run.status === "pending" || run.status === "running") {
    return (
      <Card className="flex items-center gap-3">
        <Spinner />
        <span className="text-sm text-muted">Training in the background ({run.status})...</span>
      </Card>
    );
  }

  if (run.status === "failed") {
    return (
      <Card>
        <Badge tone="danger">Training failed</Badge>
        <p className="mt-2 text-sm text-muted">{run.error_message}</p>
      </Card>
    );
  }

  const results = run.results ?? {};
  const isUplift = run.task_type === "uplift";

  return (
    <Card>
      <Badge tone="success">Training complete</Badge>

      {isUplift ? (
        <div className="mt-4 grid grid-cols-2 gap-3">
          <StatTile label="Avg. uplift" value={formatScore(results.avg_uplift)} />
          <StatTile label="% with positive uplift" value={formatPct(results.positive_uplift_pct)} />
          <StatTile label="Control score" value={formatScore(results.score_control)} />
          <StatTile label="Treatment score" value={formatScore(results.score_treatment)} />
        </div>
      ) : (
        <div className="mt-4 grid grid-cols-2 gap-3">
          <StatTile label="Best model" value={String(results.best_model ?? "-")} />
          <StatTile label="Best score" value={formatScore(results.best_score)} />
          <StatTile label="Task type" value={String(results.task_type ?? "-")} />
          <StatTile label="Features used" value={String(results.n_features ?? "-")} />
        </div>
      )}

      <FeatureImportance importance={results.feature_importance as Record<string, number> | undefined} />
    </Card>
  );
}

function FeatureImportance({ importance }: { importance?: Record<string, number> }) {
  if (!importance || Object.keys(importance).length === 0) return null;
  const entries = Object.entries(importance).slice(0, 8);
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);

  return (
    <div className="mt-6">
      <h4 className="text-sm font-medium">Feature importance</h4>
      <div className="mt-3 space-y-2">
        {entries.map(([name, value]) => (
          <div key={name}>
            <div className="flex justify-between text-xs text-muted">
              <span>{name}</span>
              <span>{value.toFixed(3)}</span>
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-surface-border">
              <div className="h-1.5 rounded-full bg-accent" style={{ width: `${(value / max) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatScore(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function formatPct(value: unknown): string {
  if (typeof value !== "number") return "-";
  return `${value.toFixed(2)}%`;
}
