"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  AlertTriangle,
  CheckCircle2,
  MessageCircle,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Users,
  XCircle,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Badge, Button, Card, FadeIn, Markdown, Spinner, StatTile } from "@/components/ui";
import { getExperiment } from "@/lib/api";
import type { Experiment, GuardrailResult } from "@/lib/types";

export default function ExperimentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    getExperiment(id)
      .then(setExperiment)
      .catch(() => setNotFound(true));
  }, [id]);

  return (
    <AppShell>
      {notFound ? (
        <p className="text-sm text-muted">Experiment not found.</p>
      ) : !experiment ? (
        <Spinner />
      ) : (
        <ExperimentDetail experiment={experiment} />
      )}
    </AppShell>
  );
}

function ExperimentDetail({ experiment }: { experiment: Experiment }) {
  const r = experiment.results;
  const controlValue = r.mean_control ?? (r.p_control !== undefined ? r.p_control * 100 : undefined);
  const treatmentValue = r.mean_treatment ?? (r.p_treatment !== undefined ? r.p_treatment * 100 : undefined);
  const isRate = r.p_control !== undefined;
  const srm = r.health_checks?.sample_ratio_mismatch;

  const chartData =
    controlValue !== undefined && treatmentValue !== undefined
      ? [
          { name: "Control", value: controlValue },
          { name: "Treatment", value: treatmentValue },
        ]
      : [];

  return (
    <div className="max-w-4xl">
      <FadeIn className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{experiment.name}</h1>
          <p className="mt-1 text-sm text-muted">
            {experiment.mode} · {r.test_name} · {new Date(experiment.created_at).toLocaleString()}
          </p>
        </div>
        <Badge tone={r.is_significant ? "success" : "neutral"} icon={r.is_significant ? CheckCircle2 : XCircle}>
          {r.is_significant ? "Statistically significant" : "Not significant"}
        </Badge>
      </FadeIn>

      {experiment.hypothesis && (
        <FadeIn delay={0.02}>
          <div className="mt-4 rounded-xl border border-surface-border bg-surface-2/40 px-4 py-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Hypothesis</p>
            <p className="mt-1 text-sm italic text-foreground/90">&ldquo;{experiment.hypothesis}&rdquo;</p>
          </div>
        </FadeIn>
      )}

      {srm && (
        <FadeIn delay={0.03}>
          <div
            className={`mt-4 flex items-center gap-2.5 rounded-xl border px-4 py-3 text-sm ${
              srm.passed ? "border-success/20 bg-success/5 text-success" : "border-danger/20 bg-danger/5 text-danger"
            }`}
          >
            {srm.passed ? <ShieldCheck className="h-4 w-4 shrink-0" /> : <AlertTriangle className="h-4 w-4 shrink-0" />}
            <span>
              {srm.passed
                ? "Health check passed — control/treatment split matches what was expected."
                : `Sample ratio mismatch detected — observed a ${(srm.observed_ratio! * 100).toFixed(0)}/${(
                    (1 - srm.observed_ratio!) *
                    100
                  ).toFixed(0)} split against an expected ${(srm.expected_ratio * 100).toFixed(0)}/${(
                    (1 - srm.expected_ratio) *
                    100
                  ).toFixed(0)}. This usually means broken randomization — treat this result with caution.`}
            </span>
          </div>
        </FadeIn>
      )}

      <FadeIn delay={0.05} className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="P-value" value={r.p_value.toFixed(4)} icon={Sparkles} />
        <StatTile
          label="Uplift"
          value={`${r.uplift_percentage >= 0 ? "+" : ""}${r.uplift_percentage.toFixed(2)}%`}
          tone={r.uplift_percentage >= 0 ? "success" : "danger"}
          icon={r.uplift_percentage >= 0 ? TrendingUp : TrendingDown}
        />
        <StatTile label="Control (n)" value={(r.n_control ?? 0).toLocaleString()} icon={Users} />
        <StatTile label="Treatment (n)" value={(r.n_treatment ?? 0).toLocaleString()} icon={Users} />
      </FadeIn>

      {r.ai_summary && (
        <FadeIn delay={0.08}>
          <Card className="mt-6" glow>
            <div className="flex items-center gap-2 text-sm font-medium">
              <Sparkles className="h-4 w-4 text-accent" />
              AI Summary
            </div>
            <Markdown className="mt-2 text-muted">{r.ai_summary}</Markdown>
          </Card>
        </FadeIn>
      )}

      {chartData.length > 0 && (
        <FadeIn delay={0.1}>
          <Card className="mt-6">
            <h3 className="mb-4 text-sm font-medium">Control vs. treatment {isRate ? "(rate %)" : "(mean)"}</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--muted)" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--muted)" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--surface-border)",
                    borderRadius: 12,
                    fontSize: 13,
                  }}
                />
                <Bar dataKey="value" fill="var(--accent)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>
      )}

      {r.guardrails && r.guardrails.length > 0 && (
        <FadeIn delay={0.12}>
          <Card className="mt-6">
            <h3 className="mb-4 text-sm font-medium">Scorecard</h3>
            <div className="space-y-3">
              <ScorecardRow label={r.metric ?? "Primary metric"} result={r} isPrimary />
              {r.guardrails.map((g) => (
                <ScorecardRow key={g.metric} label={g.metric} result={g} />
              ))}
            </div>
          </Card>
        </FadeIn>
      )}

      <FadeIn delay={0.15}>
        <Card className="mt-6">
          <h3 className="text-sm font-medium">What should I do next?</h3>
          <p className="mt-2 text-sm text-muted">
            Ask ABBot to explain this result in plain English, or dig into which segments benefit most with an uplift model.
          </p>
          <div className="mt-4 flex gap-3">
            <Link href={`/chat?experiment_id=${experiment.id}`}>
              <Button icon={MessageCircle}>Ask ABBot about this result</Button>
            </Link>
            <Link href="/ml-studio">
              <Button variant="secondary">Train a model</Button>
            </Link>
          </div>
        </Card>
      </FadeIn>
    </div>
  );
}

function ScorecardRow({
  label,
  result,
  isPrimary = false,
}: {
  label: string;
  result: GuardrailResult | { is_significant: boolean; uplift_percentage: number; p_value: number };
  isPrimary?: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg border px-3.5 py-2.5 ${
        isPrimary ? "border-accent/30 bg-accent/5" : "border-surface-border bg-surface-2/40"
      }`}
    >
      <div className="flex items-center gap-2">
        <span className={`text-sm font-medium ${isPrimary ? "text-accent" : ""}`}>{label}</span>
        {isPrimary && <span className="rounded-full bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">primary</span>}
      </div>
      <div className="flex items-center gap-4 text-xs">
        <span className="tabular-nums text-muted">p={result.p_value.toFixed(4)}</span>
        <span className={`tabular-nums font-medium ${result.uplift_percentage >= 0 ? "text-success" : "text-danger"}`}>
          {result.uplift_percentage >= 0 ? "+" : ""}
          {result.uplift_percentage.toFixed(2)}%
        </span>
        <Badge tone={result.is_significant ? "success" : "neutral"}>{result.is_significant ? "Significant" : "Not significant"}</Badge>
      </div>
    </div>
  );
}
