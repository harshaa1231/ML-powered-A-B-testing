"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Beaker, CheckCircle2, GraduationCap, MessageCircle, Plus, Sparkles, XCircle } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { getAnalyticsOverview, listExperiments } from "@/lib/api";
import type { AnalyticsOverview, Experiment } from "@/lib/types";
import { Badge, Card, EmptyState, FadeIn, GroundedIn, Markdown, Skeleton, StatTile } from "@/components/ui";

export default function DashboardPage() {
  const { user } = useAuth();
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [recent, setRecent] = useState<Experiment[] | null>(null);
  const isLearner = user?.persona === "learner";

  useEffect(() => {
    getAnalyticsOverview().then(setOverview).catch(() => setOverview(null));
    listExperiments()
      .then((all) => setRecent(all.slice(0, 5)))
      .catch(() => setRecent([]));
  }, []);

  const trendData = overview?.trend.map((t) => ({ week: t.week.replace(/^\d{4}-W/, "W"), count: t.count, significant: t.significant })) ?? [];

  return (
    <>
      <FadeIn className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">{isLearner ? "Your learning dashboard" : "Program Overview"}</h1>
        <p className="mt-1 text-sm text-muted">
          {isLearner
            ? "Practice runs and saved tests — but the real learning happens in Learn A/B Testing and ABBot."
            : "Trends across every experiment your program has run."}
        </p>
      </FadeIn>

      {isLearner && (
        <FadeIn delay={0.03} className="mb-6">
          <Card className="flex flex-col items-start gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/10">
                <GraduationCap className="h-4.5 w-4.5 text-accent" />
              </div>
              <div>
                <h3 className="font-medium">New to A/B testing?</h3>
                <p className="mt-1 text-sm text-muted">Start with the lessons, then ask ABBot anything that&apos;s unclear.</p>
              </div>
            </div>
            <div className="flex shrink-0 gap-2">
              <Link href="/learn">
                <button className="rounded-lg border border-surface-border px-3 py-1.5 text-sm font-medium hover:bg-surface-2">
                  Start learning
                </button>
              </Link>
              <Link href="/chat">
                <button className="flex items-center gap-1.5 rounded-lg gradient-accent px-3 py-1.5 text-sm font-medium text-accent-foreground">
                  <MessageCircle className="h-3.5 w-3.5" />
                  Ask ABBot
                </button>
              </Link>
            </div>
          </Card>
        </FadeIn>
      )}

      {overview === null ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : overview.total_experiments === 0 ? (
        <EmptyState
          icon={Beaker}
          title="No experiments yet"
          description="Run your first A/B test and program trends will show up here."
          action={
            <Link href="/experiments/new">
              <button className="flex items-center gap-1.5 rounded-lg gradient-accent px-4 py-2 text-sm font-medium text-accent-foreground">
                <Plus className="h-4 w-4" />
                Run your first test
              </button>
            </Link>
          }
        />
      ) : (
        <>
          <FadeIn delay={0.05} className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatTile label="Total experiments" value={overview.total_experiments.toLocaleString()} icon={Beaker} />
            <StatTile
              label="Significance rate"
              value={`${(overview.significance_rate * 100).toFixed(0)}%`}
              tone={overview.significance_rate >= 0.5 ? "success" : "neutral"}
            />
            <StatTile label="This week" value={overview.experiments_this_week.toLocaleString()} />
            <StatTile
              label="Guardrail flags"
              value={overview.guardrail_failure_rate !== null ? `${(overview.guardrail_failure_rate * 100).toFixed(0)}%` : "—"}
              tone={overview.guardrail_failure_rate && overview.guardrail_failure_rate > 0 ? "danger" : "success"}
            />
          </FadeIn>

          <FadeIn delay={0.1}>
            <Card className="mt-6">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Sparkles className="h-4 w-4 text-accent" />
                AI trends summary
              </div>
              <Markdown className="mt-2 text-muted">{overview.ai_summary}</Markdown>
              <GroundedIn sources={overview.sources} />
            </Card>
          </FadeIn>

          {trendData.length > 1 && (
            <FadeIn delay={0.15}>
              <Card className="mt-6">
                <h3 className="mb-4 text-sm font-medium">Experiments per week</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" vertical={false} />
                    <XAxis dataKey="week" stroke="var(--muted)" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="var(--muted)" fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{ background: "var(--surface)", border: "1px solid var(--surface-border)", borderRadius: 12, fontSize: 13 }}
                    />
                    <Bar dataKey="count" fill="var(--surface-border)" radius={[6, 6, 0, 0]} name="Total" />
                    <Bar dataKey="significant" fill="var(--accent)" radius={[6, 6, 0, 0]} name="Significant" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </FadeIn>
          )}

          <FadeIn delay={0.2} className="mt-6 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Recent activity</h2>
            <Link href="/experiments" className="text-sm text-accent hover:underline">
              View all
            </Link>
          </FadeIn>

          <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {(recent ?? []).map((exp, i) => (
              <FadeIn key={exp.id} delay={0.2 + i * 0.03}>
                <Link href={`/experiments/${exp.id}`}>
                  <Card className="h-full transition-colors hover:border-accent/50">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="font-medium">{exp.name}</h3>
                      <Badge tone={exp.results.is_significant ? "success" : "neutral"} icon={exp.results.is_significant ? CheckCircle2 : XCircle}>
                        {exp.results.is_significant ? "Significant" : "Not significant"}
                      </Badge>
                    </div>
                    <p className="mt-1 text-xs uppercase tracking-wide text-muted">{exp.mode} · {exp.results.test_name}</p>
                    <p className="mt-4 text-sm text-muted">
                      p-value {exp.results.p_value.toFixed(4)} · uplift {exp.results.uplift_percentage.toFixed(2)}%
                    </p>
                  </Card>
                </Link>
              </FadeIn>
            ))}
          </div>
        </>
      )}
    </>
  );
}
