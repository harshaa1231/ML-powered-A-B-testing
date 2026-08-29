"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Beaker, CheckCircle2, FlaskConical, GraduationCap, MessageCircle, Plus, XCircle } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useAuth } from "@/lib/auth";
import { listExperiments } from "@/lib/api";
import type { Experiment } from "@/lib/types";
import { Badge, Button, Card, EmptyState, FadeIn, Skeleton } from "@/components/ui";

export default function DashboardPage() {
  const { user } = useAuth();
  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const isLearner = user?.persona === "learner";

  useEffect(() => {
    listExperiments().then(setExperiments).catch(() => setExperiments([]));
  }, []);

  return (
    <AppShell>
      <FadeIn className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{isLearner ? "Your learning dashboard" : "Experiment history"}</h1>
          <p className="mt-1 text-sm text-muted">
            {isLearner
              ? "Practice runs and saved tests — but the real learning happens in Learn A/B Testing and ABBot."
              : "Every test you've run, saved to your account."}
          </p>
        </div>
        <Link href="/experiments/new">
          <Button icon={Plus}>New experiment</Button>
        </Link>
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
                <Button variant="secondary" size="sm">
                  Start learning
                </Button>
              </Link>
              <Link href="/chat">
                <Button size="sm" icon={MessageCircle}>
                  Ask ABBot
                </Button>
              </Link>
            </div>
          </Card>
        </FadeIn>
      )}

      {experiments === null ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-40" />
          ))}
        </div>
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
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {experiments.map((exp, i) => (
            <FadeIn key={exp.id} delay={i * 0.03}>
              <Link href={`/experiments/${exp.id}`}>
                <Card className="h-full transition-colors hover:border-accent/50">
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="font-medium">{exp.name}</h3>
                    <Badge tone={exp.results.is_significant ? "success" : "neutral"} icon={exp.results.is_significant ? CheckCircle2 : XCircle}>
                      {exp.results.is_significant ? "Significant" : "Not significant"}
                    </Badge>
                  </div>
                  <p className="mt-1 flex items-center gap-1.5 text-xs uppercase tracking-wide text-muted">
                    <FlaskConical className="h-3 w-3" />
                    {exp.mode} · {exp.results.test_name}
                  </p>
                  <p className="mt-4 text-sm text-muted">
                    p-value {exp.results.p_value.toFixed(4)} · uplift {exp.results.uplift_percentage.toFixed(2)}%
                  </p>
                  <p className="mt-2 text-xs text-muted">{new Date(exp.created_at).toLocaleString()}</p>
                </Card>
              </Link>
            </FadeIn>
          ))}
        </div>
      )}
    </AppShell>
  );
}
