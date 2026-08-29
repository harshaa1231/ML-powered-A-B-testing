"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Beaker,
  Briefcase,
  BrainCircuit,
  Database,
  GraduationCap,
  MessageCircle,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { Button, Card } from "@/components/ui";
import { ABTestDemo } from "@/components/ABTestDemo";
import { ThemeToggle } from "@/components/ThemeToggle";

const FEATURES = [
  {
    icon: BarChart3,
    title: "Statistical testing, done right",
    body: "Welch's t-test, Mann-Whitney U, chi-square, and two-proportion z-tests — the right test is auto-recommended from your data's shape.",
  },
  {
    icon: BrainCircuit,
    title: "ML Model Studio",
    body: "Train Gradient Boosting and Random Forest models, compare them automatically, and run T-learner uplift modeling to find who actually benefits from treatment.",
  },
  {
    icon: MessageCircle,
    title: "ABBot, a RAG-grounded assistant",
    body: "Answers are retrieved from a curated A/B testing knowledge base via pgvector similarity search, and grounded in your own live experiment results.",
  },
  {
    icon: Database,
    title: "Experiment history that persists",
    body: "Every test and trained model is saved to your account in Postgres — revisit past results instead of starting from zero each session.",
  },
];

const STEPS = [
  { icon: Beaker, title: "Run a test", body: "Type in numbers for a quick read, or upload a full dataset for auto-detected analysis." },
  { icon: TrendingUp, title: "Train a model", body: "Go beyond averages — see which segments respond, and predict outcomes on new data." },
  { icon: Sparkles, title: "Ask ABBot", body: "Get a plain-English explanation of what your results mean, grounded in real statistics." },
];

const PATHS = [
  {
    persona: "business" as const,
    icon: Briefcase,
    title: "I'm running experiments",
    subtitle: "Business teams, founders, PMs, analysts",
    body: "Run tests, train models, and get straight answers: is this significant, does it matter, and what should you do next. ABBot leads with the verdict, not the theory.",
    bullets: ["Result interpretation & ship/no-ship calls", "Business-impact framing, not just p-values", "Guardrail & risk flags on every result"],
  },
  {
    persona: "learner" as const,
    icon: GraduationCap,
    title: "I'm here to learn",
    subtitle: "Students, career switchers, the curious",
    body: "Build real statistical intuition. ABBot teaches from first principles, with patient explanations and analogies — a tutor, not a dashboard.",
    bullets: ["Concepts explained from the ground up", "Curated lessons + a knowledge-grounded tutor", "No pressure — learn at your own pace"],
  },
];

export default function LandingPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user) router.replace("/dashboard");
  }, [loading, user, router]);

  return (
    <div className="flex flex-1 flex-col overflow-x-hidden">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg gradient-accent">
            <Sparkles className="h-4 w-4 text-accent-foreground" />
          </div>
          <span className="text-sm font-semibold tracking-tight">AB Testing Pro</span>
        </div>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link href="/login">
            <Button variant="ghost" size="sm">
              Sign in
            </Button>
          </Link>
          <a href="#choose-path">
            <Button size="sm" icon={ArrowRight}>
              Get started
            </Button>
          </a>
        </div>
      </header>

      <section className="relative">
        <div
          className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[560px] opacity-60 grid-fade"
          aria-hidden
        />
        <div
          className="pointer-events-none absolute left-1/2 top-0 -z-10 h-[500px] w-[900px] -translate-x-1/2 rounded-full opacity-25 blur-[120px]"
          style={{ background: "radial-gradient(circle, var(--accent), transparent 70%)" }}
          aria-hidden
        />

        <div className="mx-auto max-w-4xl px-6 pb-16 pt-16 text-center md:pt-24">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-1.5 rounded-full border border-surface-border bg-surface px-3 py-1 text-xs font-medium text-muted"
          >
            <Sparkles className="h-3 w-3 text-accent" />
            Statistics · Machine learning · Retrieval-augmented AI
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.05 }}
            className="mt-6 text-4xl font-semibold leading-[1.1] tracking-tight md:text-6xl"
          >
            Run experiments.
            <br />
            <span className="gradient-text">Understand what actually works.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mx-auto mt-6 max-w-xl text-base text-muted md:text-lg"
          >
            A statistics and ML platform for A/B testing, with a retrieval-augmented AI assistant grounded in both
            experimentation best practices and your own live results — not generic chatbot advice.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15 }}
            className="mt-8 flex justify-center gap-3"
          >
            <a href="#choose-path">
              <Button size="lg" icon={ArrowRight}>
                Create a free account
              </Button>
            </a>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="mx-auto max-w-2xl px-6 pb-24"
        >
          <ABTestDemo />
          <p className="mt-3 text-center text-xs text-muted">
            A live simulation, computed with the same two-proportion z-test this product runs on real data.
          </p>
        </motion.div>
      </section>

      <section id="choose-path" className="border-t border-surface-border bg-surface/40 py-24">
        <div className="mx-auto max-w-4xl px-6">
          <div className="text-center">
            <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">Choose your path</h2>
            <p className="mt-3 text-muted">
              Same platform, different focus — ABBot&apos;s answers and your dashboard adapt to which one you pick.
            </p>
          </div>

          <div className="mt-12 grid grid-cols-1 gap-6 md:grid-cols-2">
            {PATHS.map((path, i) => (
              <motion.div
                key={path.persona}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
              >
                <Card className="flex h-full flex-col" glow>
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl gradient-accent">
                    <path.icon className="h-5 w-5 text-accent-foreground" />
                  </div>
                  <h3 className="mt-4 text-lg font-semibold">{path.title}</h3>
                  <p className="text-xs font-medium uppercase tracking-wider text-muted">{path.subtitle}</p>
                  <p className="mt-3 text-sm leading-relaxed text-muted">{path.body}</p>
                  <ul className="mt-4 flex-1 space-y-2">
                    {path.bullets.map((b) => (
                      <li key={b} className="flex items-start gap-2 text-sm">
                        <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent" />
                        {b}
                      </li>
                    ))}
                  </ul>
                  <Link href={`/signup?persona=${path.persona}`} className="mt-6">
                    <Button className="w-full" icon={ArrowRight}>
                      Continue as {path.persona === "business" ? "a business user" : "a learner"}
                    </Button>
                  </Link>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-5xl grid-cols-1 gap-5 px-6 py-24 md:grid-cols-2">
        {FEATURES.map((f, i) => (
          <motion.div
            key={f.title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.5, delay: i * 0.05 }}
            className="rounded-2xl border border-surface-border bg-surface p-6"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10">
              <f.icon className="h-4.5 w-4.5 text-accent" />
            </div>
            <h3 className="mt-4 font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </motion.div>
        ))}
      </section>

      <section className="border-t border-surface-border bg-surface/40 py-24">
        <div className="mx-auto max-w-5xl px-6">
          <h2 className="text-center text-2xl font-semibold tracking-tight md:text-3xl">From data to decision, in three steps</h2>
          <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="text-center"
              >
                <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl gradient-accent">
                  <s.icon className="h-5 w-5 text-accent-foreground" />
                </div>
                <h3 className="mt-4 font-semibold">{s.title}</h3>
                <p className="mt-2 text-sm text-muted">{s.body}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-surface-border py-8 text-center text-xs text-muted">
        &copy; 2026 A/B Testing Framework. All rights reserved. Built by Harsha Talapaka.
      </footer>
    </div>
  );
}
