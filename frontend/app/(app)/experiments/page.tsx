"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Beaker, Plus } from "lucide-react";
import { Badge, Button, EmptyState, FadeIn, Select, Skeleton } from "@/components/ui";
import { listExperiments } from "@/lib/api";
import type { Experiment } from "@/lib/types";

type SortKey = "created_at" | "p_value" | "uplift_percentage" | "name";
type SignificanceFilter = "all" | "significant" | "not_significant";

export default function ExperimentsListPage() {
  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [filter, setFilter] = useState<SignificanceFilter>("all");

  useEffect(() => {
    listExperiments().then(setExperiments).catch(() => setExperiments([]));
  }, []);

  const rows = useMemo(() => {
    if (!experiments) return [];
    let filtered = experiments;
    if (filter === "significant") filtered = filtered.filter((e) => e.results.is_significant);
    if (filter === "not_significant") filtered = filtered.filter((e) => !e.results.is_significant);

    const sorted = [...filtered].sort((a, b) => {
      let diff = 0;
      if (sortKey === "created_at") diff = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      else if (sortKey === "p_value") diff = a.results.p_value - b.results.p_value;
      else if (sortKey === "uplift_percentage") diff = a.results.uplift_percentage - b.results.uplift_percentage;
      else diff = a.name.localeCompare(b.name);
      return sortDir === "asc" ? diff : -diff;
    });
    return sorted;
  }, [experiments, sortKey, sortDir, filter]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  return (
    <>
      <FadeIn className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Experiments</h1>
          <p className="mt-1 text-sm text-muted">Every test you&apos;ve run, sortable and filterable.</p>
        </div>
        <Link href="/experiments/new">
          <Button icon={Plus}>New experiment</Button>
        </Link>
      </FadeIn>

      {experiments === null ? (
        <Skeleton className="h-96" />
      ) : experiments.length === 0 ? (
        <EmptyState
          icon={Beaker}
          title="No experiments yet"
          description="Run your first A/B test to see it show up here."
          action={
            <Link href="/experiments/new">
              <Button icon={Plus}>Run your first test</Button>
            </Link>
          }
        />
      ) : (
        <FadeIn delay={0.05}>
          <div className="mb-3 flex items-center justify-between">
            <Select value={filter} onChange={(e) => setFilter(e.target.value as SignificanceFilter)} className="w-56">
              <option value="all">All results</option>
              <option value="significant">Significant only</option>
              <option value="not_significant">Not significant only</option>
            </Select>
            <p className="text-xs text-muted">{rows.length.toLocaleString()} experiments</p>
          </div>

          <div className="overflow-x-auto rounded-xl border border-surface-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border bg-surface-2/50 text-left text-xs uppercase tracking-wider text-muted">
                  <SortableHeader label="Name" sortKey="name" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <th className="px-4 py-3 font-medium">Test type</th>
                  <SortableHeader label="P-value" sortKey="p_value" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortableHeader
                    label="Uplift"
                    sortKey="uplift_percentage"
                    activeKey={sortKey}
                    dir={sortDir}
                    onClick={toggleSort}
                  />
                  <th className="px-4 py-3 font-medium">Significant</th>
                  <th className="px-4 py-3 font-medium">Decision</th>
                  <SortableHeader label="Created" sortKey="created_at" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                </tr>
              </thead>
              <tbody>
                {rows.map((exp) => (
                  <tr key={exp.id} className="border-b border-surface-border last:border-0 hover:bg-surface-2/40">
                    <td className="px-4 py-3">
                      <Link href={`/experiments/${exp.id}`} className="font-medium hover:text-accent">
                        {exp.name}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-muted">{exp.results.test_name}</td>
                    <td className="px-4 py-3 tabular-nums text-muted">{exp.results.p_value.toFixed(4)}</td>
                    <td
                      className={`px-4 py-3 tabular-nums font-medium ${
                        exp.results.uplift_percentage >= 0 ? "text-success" : "text-danger"
                      }`}
                    >
                      {exp.results.uplift_percentage >= 0 ? "+" : ""}
                      {exp.results.uplift_percentage.toFixed(2)}%
                    </td>
                    <td className="px-4 py-3">
                      <Badge tone={exp.results.is_significant ? "success" : "neutral"}>
                        {exp.results.is_significant ? "Yes" : "No"}
                      </Badge>
                    </td>
                    <td className="px-4 py-3">
                      {exp.decision ? (
                        <Badge tone={exp.decision === "shipped" ? "success" : "danger"}>
                          {exp.decision === "shipped" ? "Shipped" : "Rolled back"}
                        </Badge>
                      ) : (
                        <span className="text-xs text-muted">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs text-muted">{new Date(exp.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </FadeIn>
      )}
    </>
  );
}

function SortableHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onClick,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: "asc" | "desc";
  onClick: (key: SortKey) => void;
}) {
  const active = activeKey === sortKey;
  return (
    <th className="px-4 py-3 font-medium">
      <button onClick={() => onClick(sortKey)} className={`flex items-center gap-1 ${active ? "text-foreground" : ""}`}>
        {label}
        {active && (dir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />)}
      </button>
    </th>
  );
}
