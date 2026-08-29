import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui";

const LESSONS = [
  {
    title: "1. What is A/B testing?",
    body: "Imagine you own a lemonade stand and want to know if a new sign brings in more customers than the old one. You try both, and compare results. That's A/B testing: split traffic into control (A) and treatment (B), and see which performs better on a metric you care about. Companies like Google, Amazon, and Netflix run thousands of these every year.",
  },
  {
    title: "2. Why can't I just compare the raw numbers?",
    body: "If B gets 110 conversions and A gets 100, B looks better — but that gap could just be random noise. A/B testing uses statistics to answer: is this difference real, or could it be luck?",
  },
  {
    title: "3. Statistical significance",
    body: "\"Statistically significant\" means you're at least 95% confident the difference is real, not chance (a p-value below 0.05). It's about confidence an effect exists, not how big or important it is — that's a separate question (practical significance).",
  },
  {
    title: "4. Choosing the right test",
    body: "Binary outcomes (converted or not) → chi-square or two-proportion z-test. Continuous, roughly normal metrics (revenue, session length) → Welch's t-test. Skewed continuous metrics with outliers → Mann-Whitney U. This app auto-recommends one based on your data.",
  },
  {
    title: "5. Sample size and the peeking problem",
    body: "Testing with 20 people per group won't give reliable results — run a sample-size calculation before starting. And don't stop the test the moment it looks good: checking repeatedly and stopping early inflates your false-positive rate well past 5%.",
  },
  {
    title: "6. Common mistakes",
    body: "Testing too many things at once, stopping too early, ignoring \"no difference\" results (still valuable — it tells you to move on), and not correcting for multiple comparisons when checking many metrics or segments at once.",
  },
  {
    title: "7. Uplift modeling: who actually benefits?",
    body: "A standard test tells you the average effect. Uplift modeling (a T-learner trains separate models on control and treatment) estimates each user's personal treatment effect — useful when you want to target a rollout instead of shipping it to everyone.",
  },
];

export default function LearnPage() {
  return (
    <AppShell>
      <h1 className="text-2xl font-semibold">Learn A/B Testing</h1>
      <p className="mt-1 text-sm text-muted">
        A crash course — each lesson takes about a minute. For anything deeper,{" "}
        <Link href="/chat" className="text-accent hover:underline">
          ask ABBot
        </Link>
        , which is grounded in a fuller knowledge base than this page.
      </p>

      <div className="mt-6 space-y-4">
        {LESSONS.map((lesson) => (
          <Card key={lesson.title}>
            <h3 className="font-medium">{lesson.title}</h3>
            <p className="mt-2 text-sm text-muted">{lesson.body}</p>
          </Card>
        ))}
      </div>

      <Card className="mt-6 border-accent/40">
        <h3 className="font-medium">Ready to try it yourself?</h3>
        <p className="mt-2 text-sm text-muted">Start with a Simple Test — just type in your numbers, no file needed.</p>
        <Link href="/experiments/new" className="mt-4 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-foreground">
          Run your first test
        </Link>
      </Card>
    </AppShell>
  );
}
