"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Button, Card, Input, Label, Select } from "@/components/ui";
import { ApiError, runAdvancedTest, runSimpleTest } from "@/lib/api";
import { coerceRowTypes, parseCsv } from "@/lib/csv";

type Mode = "simple" | "advanced";

export default function NewExperimentPage() {
  const [mode, setMode] = useState<Mode>("simple");

  return (
    <AppShell>
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

      <div className="mt-6 max-w-2xl">{mode === "simple" ? <SimpleTestForm /> : <AdvancedTestForm />}</div>
    </AppShell>
  );
}

function SimpleTestForm() {
  const router = useRouter();
  const [metricType, setMetricType] = useState<"conversion" | "continuous">("conversion");
  const [name, setName] = useState("Untitled experiment");
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
              metric_type: "conversion",
              control_conversions: Number(controlConversions),
              control_total: Number(controlTotal),
              treatment_conversions: Number(treatmentConversions),
              treatment_total: Number(treatmentTotal),
            })
          : await runSimpleTest({
              name,
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
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [groupCol, setGroupCol] = useState("");
  const [metricCol, setMetricCol] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleFile(file: File) {
    const text = await file.text();
    const parsed = coerceRowTypes(parseCsv(text));
    setRows(parsed);
    const cols = parsed.length > 0 ? Object.keys(parsed[0]) : [];
    setColumns(cols);
    setGroupCol(cols.find((c) => /group|variant|treatment/i.test(c)) ?? cols[0] ?? "");
    setMetricCol(cols[cols.length - 1] ?? "");
  }

  async function handleSubmit() {
    if (!rows || !groupCol || !metricCol) return;
    setError(null);
    setSubmitting(true);
    try {
      const experiment = await runAdvancedTest({ name, group_col: groupCol, metric_col: metricCol, test_type: "auto", rows });
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
          .
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
                    {c}
                  </option>
                ))}
              </Select>
            </div>
          </div>
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
