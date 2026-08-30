"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Sparkles, XCircle } from "lucide-react";
import { StreakBadge } from "@/components/StreakBadge";
import { Badge, Button, Card, FadeIn, GroundedIn, Markdown, Select, Skeleton, StatTile } from "@/components/ui";
import { ApiError, getSampleDataset, listSampleDatasets, runAdvancedTest, submitPracticeFeedback } from "@/lib/api";
import { useProgress } from "@/lib/progress";
import type { Experiment, PracticeFeedbackResponse, SampleDatasetSummary } from "@/lib/types";

type Stage = "pick" | "conclude" | "result";

export default function PracticeLabPage() {
  const { completePracticeScenario } = useProgress();
  const [scenarios, setScenarios] = useState<SampleDatasetSummary[] | null>(null);
  const [scenarioKey, setScenarioKey] = useState("");
  const [stage, setStage] = useState<Stage>("pick");
  const [conclusion, setConclusion] = useState("");
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [feedback, setFeedback] = useState<PracticeFeedbackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    listSampleDatasets().then((all) => {
      setScenarios(all);
      if (all.length > 0) setScenarioKey(all[0].key);
    });
  }, []);

  const scenario = scenarios?.find((s) => s.key === scenarioKey);

  async function handleAnalyze() {
    if (!scenario) return;
    setError(null);
    setLoading(true);
    try {
      const detail = await getSampleDataset(scenario.key);
      const result = await runAdvancedTest({
        name: `Practice: ${scenario.name}`,
        group_col: detail.group_col,
        metric_col: detail.metric_col,
        test_type: "auto",
        rows: detail.rows,
      });
      setExperiment(result);
      setStage("conclude");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmitConclusion() {
    if (!experiment || !conclusion.trim()) return;
    setError(null);
    setLoading(true);
    try {
      const fb = await submitPracticeFeedback(scenario?.name ?? "this scenario", conclusion, experiment.results);
      setFeedback(fb);
      completePracticeScenario(scenarioKey);
      setStage("result");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setStage("pick");
    setConclusion("");
    setExperiment(null);
    setFeedback(null);
    setError(null);
  }

  return (
    <>
      <FadeIn className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Practice Lab</h1>
          <p className="mt-1 text-sm text-muted">
            Pick a scenario, form your own conclusion, then run the real analysis and see how you did.
          </p>
        </div>
        <StreakBadge />
      </FadeIn>

      {scenarios === null ? (
        <Skeleton className="mt-6 h-64" />
      ) : (
        <FadeIn delay={0.05} className="mt-6 max-w-2xl">
          <Card>
            {stage === "pick" && (
              <>
                <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-muted">Scenario</label>
                <Select value={scenarioKey} onChange={(e) => setScenarioKey(e.target.value)}>
                  {scenarios.map((s) => (
                    <option key={s.key} value={s.key}>
                      {s.name}
                    </option>
                  ))}
                </Select>
                {scenario && (
                  <p className="mt-2 text-sm text-muted">
                    {scenario.description}{" "}
                    {scenario.key === "cookie_cats" && (
                      <span className="font-medium text-accent">Based on a real, published experiment.</span>
                    )}
                  </p>
                )}
                {error && <p className="mt-3 text-sm text-danger">{error}</p>}
                <Button onClick={handleAnalyze} loading={loading} className="mt-4 w-full">
                  Analyze this scenario
                </Button>
              </>
            )}

            {stage === "conclude" && experiment && (
              <>
                <p className="text-sm font-medium">Before you see the numbers laid out — what do you conclude?</p>
                <p className="mt-1 text-xs text-muted">
                  We ran the real {experiment.results.test_name} on this data. Write your read on it before we show you ours.
                </p>
                <textarea
                  value={conclusion}
                  onChange={(e) => setConclusion(e.target.value)}
                  rows={4}
                  placeholder="e.g. I think this is significant and the effect is worth shipping because..."
                  className="mt-3 w-full rounded-xl border border-surface-border bg-surface-2/60 px-3.5 py-2.5 text-sm outline-none focus:border-accent focus:ring-4 focus:ring-[var(--ring)]"
                />
                {error && <p className="mt-3 text-sm text-danger">{error}</p>}
                <Button onClick={handleSubmitConclusion} loading={loading} disabled={!conclusion.trim()} className="mt-3 w-full">
                  Submit & compare
                </Button>
              </>
            )}

            {stage === "result" && experiment && feedback && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">The real result</h3>
                  <Badge tone={experiment.results.is_significant ? "success" : "neutral"} icon={experiment.results.is_significant ? CheckCircle2 : XCircle}>
                    {experiment.results.is_significant ? "Significant" : "Not significant"}
                  </Badge>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <StatTile label="P-value" value={experiment.results.p_value.toFixed(4)} />
                  <StatTile
                    label="Uplift"
                    value={`${experiment.results.uplift_percentage >= 0 ? "+" : ""}${experiment.results.uplift_percentage.toFixed(2)}%`}
                    tone={experiment.results.uplift_percentage >= 0 ? "success" : "danger"}
                  />
                  <StatTile label="Test" value={experiment.results.test_name} />
                </div>

                <div className="rounded-lg border border-surface-border bg-surface-2/40 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted">Your conclusion</p>
                  <p className="mt-1 text-sm text-muted">{conclusion}</p>
                </div>

                <div className="rounded-lg border border-accent/20 bg-accent/5 p-3">
                  <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent">
                    <Sparkles className="h-3.5 w-3.5" />
                    ABBot&apos;s feedback
                  </div>
                  <Markdown className="mt-2">{feedback.feedback}</Markdown>
                  <GroundedIn sources={feedback.sources} />
                </div>

                <Button onClick={reset} variant="secondary" className="w-full">
                  Try another scenario
                </Button>
              </div>
            )}
          </Card>
        </FadeIn>
      )}
    </>
  );
}
