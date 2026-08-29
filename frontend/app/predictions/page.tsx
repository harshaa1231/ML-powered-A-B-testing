"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Button, Card, EmptyState, Label, Select, Spinner } from "@/components/ui";
import { ApiError, listMLRuns, predict } from "@/lib/api";
import { coerceRowTypes, parseCsv } from "@/lib/csv";
import { downloadCsv } from "@/lib/downloadCsv";
import type { MLRun } from "@/lib/types";

export default function PredictionsPage() {
  const [runs, setRuns] = useState<MLRun[] | null>(null);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [predictions, setPredictions] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    listMLRuns().then((all) => {
      const done = all.filter((r) => r.status === "done");
      setRuns(done);
      if (done.length > 0) setSelectedRunId(done[0].id);
    });
  }, []);

  async function handleFile(file: File) {
    const text = await file.text();
    setRows(coerceRowTypes(parseCsv(text)));
    setPredictions(null);
  }

  async function handlePredict() {
    if (!rows || !selectedRunId) return;
    setError(null);
    setSubmitting(true);
    try {
      const result = await predict(selectedRunId, rows);
      setPredictions(result.predictions);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  function handleDownload() {
    if (!rows || !predictions) return;
    const withPredictions = rows.map((row, i) => ({ ...row, prediction: predictions[i] }));
    downloadCsv("predictions.csv", withPredictions);
  }

  return (
    <AppShell>
      <h1 className="text-2xl font-semibold">Predictions</h1>
      <p className="mt-1 text-sm text-muted">Score new data against a model you&apos;ve already trained.</p>

      {runs === null ? (
        <Spinner className="mt-6" />
      ) : runs.length === 0 ? (
        <EmptyState title="No trained models yet" description="Train a model in ML Model Studio first." />
      ) : (
        <Card className="mt-6 max-w-xl space-y-4">
          <div>
            <Label>Trained model</Label>
            <Select value={selectedRunId} onChange={(e) => setSelectedRunId(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.task_type} · {r.target_col} · {new Date(r.created_at).toLocaleString()}
                </option>
              ))}
            </Select>
          </div>

          <div>
            <Label>New data (CSV, same columns minus the target)</Label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              className="block w-full text-sm text-muted file:mr-4 file:rounded-lg file:border-0 file:bg-accent file:px-4 file:py-2 file:text-sm file:font-medium file:text-accent-foreground"
            />
          </div>

          {error && <p className="text-sm text-danger">{error}</p>}
          <Button onClick={handlePredict} disabled={!rows || submitting} className="w-full">
            {submitting ? "Scoring..." : "Run predictions"}
          </Button>

          {predictions && (
            <div>
              <p className="text-sm text-muted">Scored {predictions.length.toLocaleString()} rows.</p>
              <Button variant="secondary" className="mt-2 w-full" onClick={handleDownload}>
                Download results CSV
              </Button>
            </div>
          )}
        </Card>
      )}
    </AppShell>
  );
}
