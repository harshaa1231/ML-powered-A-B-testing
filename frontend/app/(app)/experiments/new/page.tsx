"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Gauge, Sparkles } from "lucide-react";
import { Button, Card, Input, Label, Select } from "@/components/ui";
import { ApiError, detectColumns, listMetrics, runAdvancedTest, runSimpleTest } from "@/lib/api";
import { coerceRowTypes, parseCsv } from "@/lib/csv";
import type { Metric } from "@/lib/types";

type Mode = "simple" | "advanced";

const HYPOTHESIS_PLACEHOLDER = "We believe that [change] for [segment] will [outcome] because [reason]...";

export default function NewExperimentPage() {
  const [mode, setMode] = useState<Mode>("simple");

  return (
    <>
      <h1 className="text-2xl font-semibold">New experiment</h1>
      <p className="mt-1 text-sm text-muted">Type in numbers for a quick answer, or upload a dataset for full analysis.</p>

      <div className="mt-6 inline-flex rounded-lg border border-surface-border p-1">
        {(["simple", "advanced"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-md px-4 py-1.5 text-sm font-medium capitalize transition ${
              mode === m ? "bg-accent text-accent-foreground" : "text-muted"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      <p className="mt-2 max-w-2xl text-xs text-muted">
        {mode === "simple"
          ? "Just the headline number — control vs. treatment counts, no dataset needed."
          : "Upload a CSV to add guardrail metrics, a full Scorecard, and auto-detected columns — we'll even suggest which ones look like guardrails."}
      </p>

      <div className="mt-6 max-w-2xl">{mode === "simple" ? <SimpleTestForm /> : <AdvancedTestForm />}</div>
    </>
  );
}

function HypothesisField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <Label>Hypothesis (optional)</Label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={HYPOTHESIS_PLACEHOLDER}
        rows={2}
        className="w-full rounded-xl border border-surface-border bg-surface-2/60 px-3.5 py-2.5 text-sm text-foreground outline-none transition-all placeholder:text-muted/70 focus:border-accent focus:ring-4 focus:ring-[var(--ring)]"
      />
    </div>
  );
}

function SimpleTestForm() {
  const router = useRouter();
  const [metricType, setMetricType] = useState<"conversion" | "continuous">("conversion");
  const [name, setName] = useState("Untitled experiment");
  const [hypothesis, setHypothesis] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const [controlConversions, setControlConversions] = useState("");
  const [controlTotal, setControlTotal] = useState("");
  const [treatmentConversions, setTreatmentConversions] = useState("");
  const [treatmentTotal, setTreatmentTotal] = useState("");

  const [controlValues, setControlValues] = useState("");
  const [treatmentValues, setTreatmentValues] = useState("");

  async function handleSubmit() {
    setError(null);
    setSubmitting(true);
    try {
      const experiment =
        metricType === "conversion"
          ? await runSimpleTest({
              name,
              hypothesis: hypothesis || undefined,
              metric_type: "conversion",
              control_conversions: Number(controlConversions),
              control_total: Number(controlTotal),
              treatment_conversions: Number(treatmentConversions),
              treatment_total: Number(treatmentTotal),
            })
          : await runSimpleTest({
              name,
              hypothesis: hypothesis || undefined,
              metric_type: "continuous",
              control_values: parseNumberList(controlValues),
              treatment_values: parseNumberList(treatmentValues),
            });
      router.push(`/experiments/${experiment.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card className="space-y-4">
      <div>
        <Label>Experiment name</Label>
        <Input value={name} onChange={(e) => setName(e.target.value)} />
      </div>

      <HypothesisField value={hypothesis} onChange={setHypothesis} />

      <div>
        <Label>Metric type</Label>
        <Select value={metricType} onChange={(e) => setMetricType(e.target.value as "conversion" | "continuous")}>
          <option value="conversion">Conversion rate (e.g. signups, clicks)</option>
          <option value="continuous">Continuous value (e.g. revenue, session length)</option>
        </Select>
      </div>

      {metricType === "conversion" ? (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>Control conversions</Label>
            <Input type="number" value={controlConversions} onChange={(e) => setControlConversions(e.target.value)} />
          </div>
          <div>
            <Label>Control total</Label>
            <Input type="number" value={controlTotal} onChange={(e) => setControlTotal(e.target.value)} />
          </div>
          <div>
            <Label>Treatment conversions</Label>
            <Input type="number" value={treatmentConversions} onChange={(e) => setTreatmentConversions(e.target.value)} />
          </div>
          <div>
            <Label>Treatment total</Label>
            <Input type="number" value={treatmentTotal} onChange={(e) => setTreatmentTotal(e.target.value)} />
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <Label>Control values (comma-separated)</Label>
            <Input value={controlValues} onChange={(e) => setControlValues(e.target.value)} placeholder="42, 51, 39, 60" />
          </div>
          <div>
            <Label>Treatment values (comma-separated)</Label>
            <Input value={treatmentValues} onChange={(e) => setTreatmentValues(e.target.value)} placeholder="55, 61, 58, 70" />
          </div>
        </div>
      )}

      {error && <p className="text-sm text-danger">{error}</p>}
      <Button onClick={handleSubmit} disabled={submitting} className="w-full">
        {submitting ? "Running..." : "Run test"}
      </Button>
    </Card>
  );
}

function AdvancedTestForm() {
  const router = useRouter();
  const [name, setName] = useState("Untitled experiment");
  const [hypothesis, setHypothesis] = useState("");
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [groupCol, setGroupCol] = useState("");
  const [metricCol, setMetricCol] = useState("");
  const [guardrailCols, setGuardrailCols] = useState<string[]>([]);
  const [suggestedGuardrails, setSuggestedGuardrails] = useState<string[]>([]);
  const [savedMetrics, setSavedMetrics] = useState<Metric[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    listMetrics().then(setSavedMetrics).catch(() => {});
  }, []);

  function metricFor(col: string): Metric | undefined {
    return savedMetrics.find((m) => m.column_name === col);
  }

  async function handleFile(file: File) {
    const text = await file.text();
    const parsed = coerceRowTypes(parseCsv(text));
    setRows(parsed);
    const cols = parsed.length > 0 ? Object.keys(parsed[0]) : [];
    setColumns(cols);
    setGroupCol(cols.find((c) => /group|variant|treatment/i.test(c)) ?? cols[0] ?? "");

    // A saved, non-guardrail metric that matches an uploaded column takes priority
    // over the naive "last column" default — this is the actual payoff of the
    // metrics catalog: recognize your own metric by name, not just pick a column.
    const recognizedMetric = cols.find((c) => savedMetrics.some((m) => m.column_name === c && !m.is_guardrail));
    setMetricCol(recognizedMetric ?? cols[cols.length - 1] ?? "");
    setGuardrailCols([]);
    setSuggestedGuardrails([]);

    const savedGuardrailCols = cols.filter((c) => savedMetrics.some((m) => m.column_name === c && m.is_guardrail));

    try {
      const detection = await detectColumns(parsed);
      const heuristic = detection.potential_guardrail_cols ?? [];
      const suggested = Array.from(new Set([...savedGuardrailCols, ...heuristic]));
      setSuggestedGuardrails(suggested);
      setGuardrailCols(suggested);
    } catch {
      // heuristic detection is a nice-to-have; saved guardrail metrics still apply without it
      setSuggestedGuardrails(savedGuardrailCols);
      setGuardrailCols(savedGuardrailCols);
    }
  }

  function toggleGuardrail(col: string) {
    setGuardrailCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  }

  async function handleSubmit() {
    if (!rows || !groupCol || !metricCol) return;
    setError(null);
    setSubmitting(true);
    try {
      const experiment = await runAdvancedTest({
        name,
        hypothesis: hypothesis || undefined,
        group_col: groupCol,
        metric_col: metricCol,
        test_type: "auto",
        guardrail_cols: guardrailCols,
        rows,
      });
      router.push(`/experiments/${experiment.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  const guardrailCandidates = columns.filter((c) => c !== groupCol && c !== metricCol);

  return (
    <Card className="space-y-4">
      <div>
        <Label>Experiment name</Label>
        <Input value={name} onChange={(e) => setName(e.target.value)} />
      </div>

      <HypothesisField value={hypothesis} onChange={setHypothesis} />

      <div>
        <Label>CSV file</Label>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          className="block w-full text-sm text-muted file:mr-4 file:rounded-lg file:border-0 file:bg-accent file:px-4 file:py-2 file:text-sm file:font-medium file:text-accent-foreground"
        />
        <p className="mt-1 text-xs text-muted">
          Need sample data first? Grab one from{" "}
          <a href="/datasets" className="text-accent hover:underline">
            Sample Datasets
          </a>
          . Columns matching a{" "}
          <a href="/metrics" className="text-accent hover:underline">
            saved metric
          </a>{" "}
          are recognized automatically.
        </p>
      </div>

      {rows && (
        <>
          <p className="text-xs text-muted">{rows.length.toLocaleString()} rows loaded.</p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Group column</Label>
              <Select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Select>
            </div>
            <div>
              <Label>Metric column</Label>
              <Select value={metricCol} onChange={(e) => setMetricCol(e.target.value)}>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {metricFor(c) ? `${c} (${metricFor(c)!.name})` : c}
                  </option>
                ))}
              </Select>
              {metricFor(metricCol) && (
                <p className="mt-1 flex items-center gap-1.5 text-xs text-accent">
                  <Gauge className="h-3 w-3" />
                  Using your saved metric &ldquo;{metricFor(metricCol)!.name}&rdquo;
                </p>
              )}
            </div>
          </div>

          {guardrailCandidates.length > 0 && (
            <div>
              <Label>Guardrail metrics (optional)</Label>
              {suggestedGuardrails.length > 0 && (
                <p className="mb-2 flex items-center gap-1.5 text-xs text-accent">
                  <Sparkles className="h-3 w-3" />
                  Pre-selected columns that look like guardrails — uncheck any that don&apos;t belong.
                </p>
              )}
              <div className="flex flex-wrap gap-2">
                {guardrailCandidates.map((c) => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => toggleGuardrail(c)}
                    className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                      guardrailCols.includes(c)
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-surface-border text-muted hover:bg-surface-2"
                    }`}
                  >
                    {metricFor(c) ? metricFor(c)!.name : c}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {error && <p className="text-sm text-danger">{error}</p>}
      <Button onClick={handleSubmit} disabled={!rows || submitting} className="w-full">
        {submitting ? "Analyzing..." : "Run analysis"}
      </Button>
    </Card>
  );
}

function parseNumberList(text: string): number[] {
  return text
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => !Number.isNaN(n));
}
