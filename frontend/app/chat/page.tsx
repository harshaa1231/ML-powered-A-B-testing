"use client";

import { Suspense, useRef, useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { useAuth } from "@/lib/auth";
import { Badge, Button, Input, Spinner } from "@/components/ui";
import { ApiError, sendChatMessage } from "@/lib/api";
import type { ChatSource } from "@/lib/types";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

const BUSINESS_STARTERS = [
  "Is this result ready to ship?",
  "What's the business impact of this uplift?",
  "What guardrail metrics should I check before rolling this out?",
  "My test is inconclusive — what should I do next?",
  "How do I explain this result to my manager?",
];

const LEARNER_STARTERS = [
  "What is A/B testing and why does it matter?",
  "What does a p-value actually mean in plain English?",
  "What's the difference between statistical and practical significance?",
  "How do I know if my sample size is large enough?",
  "What is uplift modeling and when should I use it?",
];

function ChatPageInner() {
  const searchParams = useSearchParams();
  const experimentId = searchParams.get("experiment_id") ?? undefined;
  const { user } = useAuth();
  const starterQuestions = user?.persona === "learner" ? LEARNER_STARTERS : BUSINESS_STARTERS;

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  async function send(text: string) {
    if (!text.trim()) return;
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setSending(true);
    try {
      const response = await sendChatMessage(text, sessionId, experimentId);
      setSessionId(response.session_id);
      setMessages((prev) => [...prev, { role: "assistant", content: response.content, sources: response.sources }]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSending(false);
      setTimeout(() => scrollRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  return (
    <AppShell>
      <div className="mx-auto flex h-[calc(100vh-5rem)] max-w-3xl flex-col">
        <div className="mb-4">
          <h1 className="text-2xl font-semibold">Ask ABBot</h1>
          <p className="mt-1 text-sm text-muted">
            Grounded in a curated A/B testing knowledge base{experimentId ? " and your selected experiment's live results" : ""}.
          </p>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-surface-border bg-surface p-4">
          {messages.length === 0 && (
            <div className="flex flex-wrap gap-2">
              {starterQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="rounded-full border border-surface-border px-3 py-1.5 text-xs hover:border-accent"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm ${
                  m.role === "user" ? "bg-accent text-accent-foreground" : "bg-background/60"
                }`}
              >
                <p className="whitespace-pre-wrap">{m.content}</p>
                {m.sources && m.sources.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {m.sources.map((s) => (
                      <Badge key={s.slug}>{s.title}</Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {sending && (
            <div className="flex justify-start">
              <div className="rounded-xl bg-background/60 px-4 py-2.5">
                <Spinner />
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {error && <p className="mt-2 text-sm text-danger">{error}</p>}

        <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your results, p-values, sample size..."
          />
          <Button type="submit" disabled={sending || !input.trim()}>
            Send
          </Button>
        </form>
      </div>
    </AppShell>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  );
}
