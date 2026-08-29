"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Button, Card, Input, Label, Select, Spinner } from "@/components/ui";
import { generateDataset, getSampleDataset, listGeneratorDomains, listSampleDatasets } from "@/lib/api";
import { downloadCsv } from "@/lib/downloadCsv";
import type { SampleDatasetSummary } from "@/lib/types";

export default function DatasetsPage() {
  const [samples, setSamples] = useState<SampleDatasetSummary[] | null>(null);
  const [domains, setDomains] = useState<string[]>([]);

  useEffect(() => {
    listSampleDatasets().then(setSamples);
    listGeneratorDomains().then(setDomains);
  }, []);

  return (
    <AppShell>
      <h1 className="text-2xl font-semibold">Sample datasets</h1>
      <p className="mt-1 text-sm text-muted">
        Download one of these, then upload it under New Experiment → Advanced to see auto-detection in action.
      </p>

      {samples === null ? (
        <Spinner className="mt-6" />
      ) : (
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {samples.map((s) => (
            <SampleCard key={s.key} sample={s} />
          ))}
        </div>
      )}

      <h2 className="mt-12 text-xl font-semibold">Synthetic data generator</h2>
      <p className="mt-1 text-sm text-muted">Generate a larger, more realistic dataset for a specific domain.</p>
      <div className="mt-4 max-w-md">
        <GeneratorForm domains={domains} />
      </div>
    </AppShell>
  );
}

function SampleCard({ sample }: { sample: SampleDatasetSummary }) {
  const [downloading, setDownloading] = useState(false);

  async function handleDownload() {
    setDownloading(true);
    try {
      const detail = await getSampleDataset(sample.key);
      downloadCsv(`${sample.key}.csv`, detail.rows);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <Card>
      <h3 className="font-medium">{sample.name}</h3>
      <p className="mt-1 text-sm text-muted">{sample.description}</p>
      <p className="mt-3 text-xs text-muted">
        {sample.row_count.toLocaleString()} rows · group: {sample.group_col} · metric: {sample.metric_col}
      </p>
      <Button variant="secondary" className="mt-4 w-full" onClick={handleDownload} disabled={downloading}>
        {downloading ? "Preparing..." : "Download CSV"}
      </Button>
    </Card>
  );
}

function GeneratorForm({ domains }: { domains: string[] }) {
  const [domain, setDomain] = useState("");
  const [nSamples, setNSamples] = useState(5000);
  const [generating, setGenerating] = useState(false);
  const [rowCount, setRowCount] = useState<number | null>(null);

  const selectedDomain = domain || domains[0] || "";

  async function handleGenerate() {
    if (!selectedDomain) return;
    setGenerating(true);
    setRowCount(null);
    try {
      const result = await generateDataset(selectedDomain, nSamples);
      downloadCsv(`${selectedDomain}_synthetic.csv`, result.rows);
      setRowCount(result.row_count);
    } finally {
      setGenerating(false);
    }
  }

  return (
    <Card className="space-y-4">
      <div>
        <Label>Domain</Label>
        <Select value={selectedDomain} onChange={(e) => setDomain(e.target.value)}>
          {domains.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </Select>
      </div>
      <div>
        <Label>Number of samples</Label>
        <Input type="number" min={100} max={200000} value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))} />
      </div>
      <Button onClick={handleGenerate} disabled={generating} className="w-full">
        {generating ? "Generating..." : "Generate & download"}
      </Button>
      {rowCount !== null && <p className="text-xs text-muted">Generated {rowCount.toLocaleString()} rows.</p>}
    </Card>
  );
}
