"use client";

import { useEffect, useState, type FormEvent } from "react";
import { CheckCircle2, Sparkles, XCircle } from "lucide-react";
import { StreakBadge } from "@/components/StreakBadge";
import { Badge, Button, Card, FadeIn, GroundedIn, Input, Markdown, Select, Skeleton, Spinner, StatTile } from "@/components/ui";
import { ApiError, getSampleDataset, listSampleDatasets, runAdvancedTest, submitPracticeFeedback } from "@/lib/api";
import { useChatSession } from "@/lib/useChatSession";
import { useProgress } from "@/lib/progress";
import type { Experiment, PracticeFeedbackResponse, SampleDatasetSummary } from "@/lib/types";

type Stage = "pick" | "conclude" | "result";

// A guided multiple-choice pick, not an open "write your conclusion" prompt — someone
// starting from zero doesn't yet have the vocabulary to write a statistical conclusion
// unprompted, but can absolutely form and click an opinion when the options are laid
// out. This mirrors the same pattern the Learn section's case studies already use.
const CONCLUSION_OPTIONS = [
  "The treatment made things better — I'd ship this",
  "The treatment made things worse — I wouldn't ship this",
  "I can't really tell — this looks like it could just be random chance",
] as const;

export default function PracticeLabPage() {
  const { completePracticeScenario } = useProgress();
  const [scenarios, setScenarios] = useState<SampleDatasetSummary[] | null>(null);
  const [scenarioKey, setScenarioKey] = useState("");
  const [stage, setStage] = useState<Stage>("pick");
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [reasoning, setReasoning] = useState("");
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [feedback, setFeedback] = useState<PracticeFeedbackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [followUpInput, setFollowUpInput] = useState("");

  useEffect(() => {
    listSampleDatasets().then((all) => {
      setScenarios(all);
      if (all.length > 0) setScenarioKey(all[0].key);
    });
  }, []);

  const scenario = scenarios?.find((s) => s.key === scenarioKey);

  // The feedback card above already serves as the opening message for this
  // experiment's context, so this follow-up box starts empty — no auto-question,
  // and no pulling in an unrelated past conversation from elsewhere in the app.
  const {
    messages: followUpMessages,
    sending: followUpSending,
    error: followUpError,
    send: sendFollowUp,
    reset: resetFollowUp,
    scrollRef: followUpScrollRef,
  } = useChatSession(stage === "result" ? experiment?.id : undefined, {
    autoSend: false,
    restoreHistory: false,
  });

  function handleFollowUpSubmit(e: FormEvent) {
    e.preventDefault();
    const text = followUpInput;
    setFollowUpInput("");
    sendFollowUp(text);
  }

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

  function conclusionText(): string {
    if (selectedOption === null) return "";
    const choice = CONCLUSION_OPTIONS[selectedOption];
    return reasoning.trim() ? `${choice}. Reasoning: ${reasoning.trim()}` : choice;
  }

  async function handleSubmitConclusion() {
    if (!experiment || selectedOption === null) return;
    setError(null);
    setLoading(true);
    try {
      const fb = await submitPracticeFeedback(scenario?.name ?? "this scenario", conclusionText(), experiment.results);
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
    setSelectedOption(null);
    setReasoning("");
    setExperiment(null);
    setFeedback(null);
    setError(null);
    setFollowUpInput("");
    resetFollowUp();
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
                {scenario && (
                  <div className="mb-4 rounded-lg border border-surface-border bg-surface-2/40 p-3">
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">What you&apos;re looking at</p>
                    <p className="mt-1 text-sm text-muted">{scenario.description}</p>
                  </div>
                )}
                <p className="text-sm font-medium">Before you see the numbers laid out — what&apos;s your best guess?</p>
                <p className="mt-1 text-xs text-muted">
                  We ran a real {experiment.results.test_name} on this data. Pick the option closest to your read —
                  there&apos;s no wrong answer here, this is exactly what practice is for.
                </p>
                <div className="mt-3 space-y-2">
                  {CONCLUSION_OPTIONS.map((option, i) => (
                    <button
                      key={option}
                      type="button"
                      onClick={() => setSelectedOption(i)}
                      className={`flex w-full items-center rounded-lg border px-3.5 py-2.5 text-left text-sm transition-colors ${
                        selectedOption === i
                          ? "border-accent bg-accent/10 text-accent"
                          : "border-surface-border hover:bg-surface-2"
                      }`}
                    >
                      {option}
                    </button>
                  ))}
                </div>
                <label className="mt-4 block text-xs font-medium uppercase tracking-wider text-muted">
                  Why do you think that? (optional)
                </label>
                <textarea
                  value={reasoning}
                  onChange={(e) => setReasoning(e.target.value)}
                  rows={3}
                  placeholder="Only if you want to elaborate — totally fine to skip this."
                  className="mt-1.5 w-full rounded-xl border border-surface-border bg-surface-2/60 px-3.5 py-2.5 text-sm outline-none focus:border-accent focus:ring-4 focus:ring-[var(--ring)]"
                />
                {error && <p className="mt-3 text-sm text-danger">{error}</p>}
                <Button
                  onClick={handleSubmitConclusion}
                  loading={loading}
                  disabled={selectedOption === null}
                  className="mt-3 w-full"
                >
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
                  <p className="mt-1 text-sm text-muted">{selectedOption !== null ? CONCLUSION_OPTIONS[selectedOption] : ""}</p>
                  {reasoning.trim() && <p className="mt-1 text-sm italic text-muted">&ldquo;{reasoning.trim()}&rdquo;</p>}
                </div>

                <div className="rounded-lg border border-accent/20 bg-accent/5 p-3">
                  <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent">
                    <Sparkles className="h-3.5 w-3.5" />
                    ABBot&apos;s feedback
                  </div>
                  <Markdown className="mt-2">{feedback.feedback}</Markdown>
                  <GroundedIn sources={feedback.sources} />
                </div>

                <div className="rounded-lg border border-surface-border p-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted">
                    Still not sure, or want to push back? Ask ABBot
                  </p>
                  {followUpMessages.length > 0 && (
                    <div className="mt-2 max-h-64 space-y-3 overflow-y-auto">
                      {followUpMessages.map((m, i) => (
                        <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div
                            className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${
                              m.role === "user" ? "bg-accent text-accent-foreground" : "bg-surface-2"
                            }`}
                          >
                            {m.role === "assistant" ? (
                              <Markdown>{m.content}</Markdown>
                            ) : (
                              <p className="whitespace-pre-wrap">{m.content}</p>
                            )}
                            {m.role === "assistant" && <GroundedIn sources={m.sources ?? []} />}
                          </div>
                        </div>
                      ))}
                      {followUpSending && <Spinner className="h-3.5 w-3.5" />}
                      <div ref={followUpScrollRef} />
                    </div>
                  )}
                  {followUpError && <p className="mt-2 text-sm text-danger">{followUpError}</p>}
                  <form onSubmit={handleFollowUpSubmit} className="mt-2 flex gap-2">
                    <Input
                      value={followUpInput}
                      onChange={(e) => setFollowUpInput(e.target.value)}
                      placeholder="e.g. Why isn't a -0.6% drop good enough evidence?"
                      className="text-sm"
                    />
                    <Button type="submit" size="sm" disabled={followUpSending || !followUpInput.trim()}>
                      Send
                    </Button>
                  </form>
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
