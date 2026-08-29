"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, MessageCircle, Sparkles, TrendingDown, TrendingUp, Users, XCircle } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Badge, Button, Card, FadeIn, Spinner, StatTile } from "@/components/ui";
import { getExperiment } from "@/lib/api";
import type { Experiment } from "@/lib/types";

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
