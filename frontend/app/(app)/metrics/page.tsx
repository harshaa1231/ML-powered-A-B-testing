"use client";

import { useEffect, useState, type FormEvent } from "react";
import { Gauge, Plus, ShieldAlert, Trash2 } from "lucide-react";
import { ApiError, createMetric, deleteMetric, listMetrics } from "@/lib/api";
import { Badge, Button, Card, FadeIn, Input, Label, Skeleton } from "@/components/ui";
import type { Metric } from "@/lib/types";

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metric[] | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [columnName, setColumnName] = useState("");
  const [isGuardrail, setIsGuardrail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    listMetrics().then(setMetrics);
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const created = await createMetric({
        name,
        description: description || undefined,
        column_name: columnName,
        is_guardrail: isGuardrail,
      });
      setMetrics((prev) => [created, ...(prev ?? [])]);
      setName("");
      setDescription("");
      setColumnName("");
      setIsGuardrail(false);
      setShowForm(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: string) {
    await deleteMetric(id);
    setMetrics((prev) => (prev ?? []).filter((m) => m.id !== id));
  }

  return (
    <>
      <FadeIn className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Metrics</h1>
          <p className="mt-1 text-sm text-muted">
            Define what a metric means once — column name, description, whether it&apos;s a guardrail — and reuse it
            by name on every future experiment instead of re-picking raw columns each time.
          </p>
        </div>
        <Button icon={Plus} onClick={() => setShowForm((v) => !v)}>
          New metric
        </Button>
      </FadeIn>

      {showForm && (
        <FadeIn delay={0.03}>
          <Card className="mt-6 max-w-xl space-y-4">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Label>Name</Label>
                <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Checkout Conversion" required />
              </div>
              <div>
                <Label>Description (optional)</Label>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="% of visitors who complete checkout"
                />
              </div>
              <div>
                <Label>Column name</Label>
                <Input
                  value={columnName}
                  onChange={(e) => setColumnName(e.target.value)}
                  placeholder="converted"
                  required
                />
                <p className="mt-1 text-xs text-muted">
                  The column name this maps to in your usual dataset export — matched automatically when you upload
                  data for an Advanced analysis.
                </p>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={isGuardrail} onChange={(e) => setIsGuardrail(e.target.checked)} />
                This is a guardrail metric (something to watch, not optimize for)
              </label>
              {error && <p className="text-sm text-danger">{error}</p>}
              <div className="flex gap-2">
                <Button type="submit" loading={submitting}>
                  Save metric
                </Button>
                <Button type="button" variant="secondary" onClick={() => setShowForm(false)}>
                  Cancel
                </Button>
              </div>
            </form>
          </Card>
        </FadeIn>
      )}

      <div className="mt-6 space-y-3">
        {metrics === null ? (
          <>
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </>
        ) : metrics.length === 0 ? (
          <Card className="border-dashed text-center">
            <Gauge className="mx-auto h-6 w-6 text-muted" />
            <p className="mt-2 text-sm text-muted">
              No saved metrics yet. Define one to reuse it by name across every future experiment.
            </p>
          </Card>
        ) : (
          metrics.map((m, i) => (
            <FadeIn key={m.id} delay={i * 0.02}>
              <Card className="flex items-center justify-between gap-4">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent/10">
                    <Gauge className="h-4 w-4 text-accent" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium">{m.name}</h3>
                      {m.is_guardrail && (
                        <Badge tone="accent" icon={ShieldAlert}>
                          Guardrail
                        </Badge>
                      )}
                    </div>
                    {m.description && <p className="mt-0.5 text-sm text-muted">{m.description}</p>}
                    <p className="mt-0.5 text-xs text-muted">
                      column: <span className="font-mono">{m.column_name}</span>
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(m.id)}
                  className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-danger"
                  aria-label={`Delete ${m.name}`}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </Card>
            </FadeIn>
          ))
        )}
      </div>
    </>
  );
}
