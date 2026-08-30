"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Card, FadeIn } from "@/components/ui";
import { StreakBadge } from "@/components/StreakBadge";
import { SkillTree, type SkillNode } from "@/components/learn/SkillTree";
import { SampleSizeCalculator } from "@/components/learn/SampleSizeCalculator";
import { SignificanceSimulator } from "@/components/learn/SignificanceSimulator";
import { LessonQuiz } from "@/components/learn/LessonQuiz";
import { CaseStudy } from "@/components/learn/CaseStudy";
import { useProgress } from "@/lib/progress";

const DEFAULT_LESSON_ID = "what-is-ab-testing";

const TIERS = [
  { key: "foundational", label: "Foundational" },
  { key: "core", label: "Core" },
  { key: "advanced", label: "Advanced" },
  { key: "case-studies", label: "Case studies" },
];

const NODES: SkillNode[] = [
  { id: "what-is-ab-testing", title: "What is A/B testing?", tier: "foundational", prerequisites: [] },
  { id: "why-not-compare", title: "Why not just compare numbers?", tier: "foundational", prerequisites: ["what-is-ab-testing"] },
  { id: "significance", title: "Statistical significance", tier: "core", prerequisites: ["why-not-compare"] },
  { id: "choosing-test", title: "Choosing the right test", tier: "core", prerequisites: ["why-not-compare"] },
  { id: "sample-size", title: "Sample size & peeking", tier: "core", prerequisites: ["significance"] },
  { id: "simulator-practice", title: "Practice: significance simulator", tier: "advanced", prerequisites: ["sample-size"] },
  { id: "uplift-modeling", title: "Uplift modeling", tier: "advanced", prerequisites: ["choosing-test"] },
  { id: "case-paywall-gate", title: "The Paywall Gate", tier: "case-studies", prerequisites: ["significance"] },
  { id: "case-checkout-button", title: "The Checkout Button", tier: "case-studies", prerequisites: ["sample-size"] },
  { id: "case-onboarding-email", title: "The Onboarding Email", tier: "case-studies", prerequisites: ["choosing-test"] },
];

export default function LearnPage() {
  const { progress, completeLesson } = useProgress();
  const [selectedId, setSelectedId] = useState<string | null>(DEFAULT_LESSON_ID);

  // The default lesson is shown without the user clicking its node — mark it complete on
  // mount too, otherwise its dependents in the skill tree stay locked forever since
  // `select()` (below) is the only other place completion is recorded.
  useEffect(() => {
    completeLesson(DEFAULT_LESSON_ID);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function select(id: string) {
    setSelectedId(id);
    if (!id.startsWith("case-")) completeLesson(id);
  }

  return (
    <>
      <FadeIn className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Learn A/B Testing</h1>
          <p className="mt-1 text-sm text-muted">
            Work through the tree, or jump around — everything not greyed out is available. For anything deeper,{" "}
            <Link href="/chat" className="text-accent hover:underline">
              ask ABBot
            </Link>
            .
          </p>
        </div>
        <StreakBadge />
      </FadeIn>

      <FadeIn delay={0.05} className="mt-6">
        <SkillTree
          tiers={TIERS}
          nodes={NODES}
          completedIds={[...progress.completedLessons, ...progress.completedCaseStudies]}
          selectedId={selectedId}
          onSelect={select}
        />
      </FadeIn>

      <FadeIn delay={0.1} className="mt-6">
        {selectedId && <LessonContent id={selectedId} />}
      </FadeIn>
    </>
  );
}

function LessonContent({ id }: { id: string }) {
  switch (id) {
    case "what-is-ab-testing":
      return (
        <Card>
          <h3 className="font-medium">What is A/B testing?</h3>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Imagine you own a lemonade stand and want to know if a new sign brings in more customers than the old
            one. You try both, and compare results. That&apos;s A/B testing: split traffic into control (A) and
            treatment (B), and see which performs better on a metric you care about. Companies like Google, Amazon,
            and Netflix run thousands of these every year.
          </p>
        </Card>
      );

    case "why-not-compare":
      return (
        <div className="space-y-4">
          <Card>
            <h3 className="font-medium">Why not just compare the raw numbers?</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              If B gets 110 conversions and A gets 100, B looks better — but that gap could just be random noise. A/B
              testing uses statistics to answer: is this difference real, or could it be luck?
            </p>
          </Card>
          <LessonQuiz
            id="quiz-why-not-compare"
            question="Version B got 5% more conversions than Version A. What should you conclude?"
            options={[
              "B is definitely better — ship it immediately",
              "The difference might just be random chance — check statistical significance first",
              "The test is broken since they should be identical",
            ]}
            correctIndex={1}
            explanation="a raw difference alone can't tell you whether it's a real effect or noise — that's exactly what a statistical test is for."
          />
        </div>
      );

    case "significance":
      return (
        <div className="space-y-4">
          <Card>
            <h3 className="font-medium">Statistical significance</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              &quot;Statistically significant&quot; means you&apos;re at least 95% confident the difference is real,
              not chance (a p-value below 0.05). It&apos;s about confidence an effect exists, not how big or
              important it is — that&apos;s a separate question (practical significance). A tiny, meaningless effect
              can be &quot;significant&quot; with enough data, and a huge effect can miss significance with too
              little.
            </p>
          </Card>
          <LessonQuiz
            id="quiz-significance"
            question="A test has p = 0.03 but the uplift is only +0.1%, on a metric where 0.1% doesn't matter to the business. What's the right takeaway?"
            options={[
              "Ship it — it's statistically significant",
              "It's statistically significant but not practically significant — probably not worth shipping",
              "p = 0.03 means there's a 3% chance the effect is real",
            ]}
            correctIndex={1}
            explanation="statistical and practical significance are different questions — you need both a real effect (low p-value) and a big enough effect to matter."
          />
        </div>
      );

    case "choosing-test":
      return (
        <div className="space-y-4">
          <Card>
            <h3 className="font-medium">Choosing the right test</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Binary outcomes (converted or not) → chi-square or two-proportion z-test. Continuous, roughly normal
              metrics (revenue, session length) → Welch&apos;s t-test. Skewed continuous metrics with outliers →
              Mann-Whitney U. This app auto-recommends one based on your data&apos;s shape.
            </p>
          </Card>
          <LessonQuiz
            id="quiz-choosing-test"
            question="You're comparing revenue per user, and a few whale customers make the distribution heavily skewed. Which test fits best?"
            options={["Chi-square", "Welch's t-test", "Mann-Whitney U"]}
            correctIndex={2}
            explanation="Mann-Whitney U compares distributions by rank rather than mean, so it's robust to the outliers a skewed revenue distribution usually has."
          />
        </div>
      );

    case "sample-size":
      return (
        <div className="space-y-4">
          <Card>
            <h3 className="font-medium">Sample size and the peeking problem</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Testing with 20 people per group won&apos;t give reliable results — run a sample-size calculation
              before starting. And don&apos;t stop the test the moment it looks good: checking repeatedly and
              stopping early inflates your false-positive rate well past 5%. The most common real-world mistakes are
              testing too many things at once, stopping too early, and not correcting for multiple comparisons when
              checking many metrics or segments simultaneously.
            </p>
          </Card>
          <SampleSizeCalculator />
          <LessonQuiz
            id="quiz-sample-size"
            question="Your test hits p < 0.05 after just 6 hours. What's the safest move?"
            options={[
              "Stop now and ship — significance is significance",
              "Keep running until you hit the sample size you calculated beforehand",
              "Restart the test to double-check",
            ]}
            correctIndex={1}
            explanation="stopping the moment it first looks significant is the classic peeking problem — early significance is often a random blip, not a durable effect."
          />
        </div>
      );

    case "simulator-practice":
      return (
        <div className="space-y-4">
          <Card>
            <h3 className="font-medium">Practice: build intuition for significance</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Play with the simulator below. Try a small gap (10% vs. 11%) at a small sample size, then increase the
              sample size until it becomes significant — this is the exact tradeoff every real experiment navigates.
            </p>
          </Card>
          <SignificanceSimulator />
        </div>
      );

    case "uplift-modeling":
      return (
        <div className="space-y-4">
          <Card>
            <h3 className="font-medium">Uplift modeling: who actually benefits?</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              A standard test tells you the average effect. Uplift modeling (a T-learner trains separate models on
              control and treatment) estimates each user&apos;s personal treatment effect — useful when you want to
              target a rollout instead of shipping it to everyone.
            </p>
          </Card>
          <LessonQuiz
            id="quiz-uplift"
            question="An A/B test shows a null average effect, but you suspect it helps mobile users while hurting desktop users. What tool fits?"
            options={["Re-run the same t-test", "Uplift modeling, segmented by device", "A larger sample size"]}
            correctIndex={1}
            explanation="uplift modeling is designed exactly for this — surfacing heterogeneous effects that a single average result hides."
          />
        </div>
      );

    case "case-paywall-gate":
      return (
        <CaseStudy
          id="case-paywall-gate"
          title="The Paywall Gate"
          narrative="A mobile game moves its first paywall gate from level 30 to level 40, hoping players get more hooked before hitting it. After one day, retention looks nearly identical between the two versions."
          stat={{ label: "1-Day Retention", control: "44.8%", treatment: "44.2%" }}
          question="Based on 1-day retention alone, what would you conclude?"
          options={[
            "The change makes no difference — ship whichever is cheaper to maintain",
            "1-day retention isn't enough evidence either way — check a longer window before deciding",
            "Moving the gate later always helps, so ship it",
          ]}
          correctIndex={1}
          reveal="This is based on a real, published mobile game experiment (moving the gate from level 30 to 40). 1-day retention genuinely wasn't significantly different — but 7-day retention was: 19.0% at gate 30 vs. 18.2% at gate 40, a real and counterintuitive effect. Delaying the gate further actually hurt long-term retention. Try this exact dataset yourself in Practice Lab."
        />
      );

    case "case-checkout-button":
      return (
        <CaseStudy
          id="case-checkout-button"
          title="The Checkout Button"
          narrative="An e-commerce team tests a green checkout button against the current blue one. After 2 days, treatment is already showing a strong lift and the team wants to call it and ship immediately."
          stat={{ label: "Conversion (day 2)", control: "4.1%", treatment: "5.3%" }}
          question="The team wants to stop the test right now, at day 2. What's the risk?"
          options={[
            "None — a 2-day sample is always enough",
            "Stopping the moment it looks good is the peeking problem — early results are noisier and can reverse",
            "The colors themselves are the problem, not the timing",
          ]}
          correctIndex={1}
          reveal="The team ran it to the pre-planned sample size anyway. The lift held up — green really did convert better here — but that's not the point: they got lucky. Repeatedly checking and stopping at the first good-looking result inflates the false-positive rate well past 5%, even when, like this time, the effect turns out to be real."
        />
      );

    case "case-onboarding-email":
      return (
        <CaseStudy
          id="case-onboarding-email"
          title="The Onboarding Email"
          narrative="A marketing team redesigns their welcome email and sees a nice lift in open rate. They're ready to roll it out to the full list."
          stat={{ label: "Open rate", control: "22%", treatment: "27%" }}
          question="Open rate improved. What should the team check before declaring victory?"
          options={[
            "Nothing — open rate is the metric that matters",
            "Guardrail metrics like unsubscribe rate, to make sure the new email isn't also driving people away",
            "Whether the email client supports the new design",
          ]}
          correctIndex={1}
          reveal="When they checked, unsubscribe rate had also risen significantly in the treatment group — the punchier subject line was too aggressive for part of the list. Without a guardrail metric, they would have shipped a change that grew the list short-term while quietly damaging it long-term."
        />
      );

    default:
      return null;
  }
}
