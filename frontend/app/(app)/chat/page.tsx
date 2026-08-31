"use client";

import { Suspense, useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { RotateCcw, Sparkles } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useChatSession } from "@/lib/useChatSession";
import { Button, GroundedIn, Input, Markdown, Spinner } from "@/components/ui";
import { DocumentUpload } from "@/components/DocumentUpload";

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

const QUIZ_ME_PROMPT =
  "Quiz me: ask me one multiple-choice question about A/B testing (four options, labeled A-D). " +
  "Don't reveal the answer yet — wait for me to reply with my chosen letter, then tell me if I got it right and explain why.";

function ChatPageInner() {
  const searchParams = useSearchParams();
  const experimentId = searchParams.get("experiment_id") ?? undefined;
  const { user } = useAuth();
  const starterQuestions = user?.persona === "learner" ? LEARNER_STARTERS : BUSINESS_STARTERS;

  const { messages, sending, error, send, reset, scrollRef } = useChatSession(experimentId);
  const [input, setInput] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input;
    setInput("");
    send(text);
  }

  return (
    <>
      <div className="mx-auto flex h-[calc(100vh-5rem)] max-w-3xl flex-col">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">Ask ABBot</h1>
            <p className="mt-1 text-sm text-muted">
              Grounded in a curated A/B testing knowledge base{experimentId ? " and your selected experiment's live results" : ""}.
            </p>
          </div>
          {messages.length > 0 && (
            <button
              onClick={reset}
              className="flex shrink-0 items-center gap-1.5 rounded-lg border border-surface-border px-3 py-1.5 text-xs font-medium text-muted hover:bg-surface-2 hover:text-foreground"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              New conversation
            </button>
          )}
        </div>

        <DocumentUpload />

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
              {user?.persona === "learner" && (
                <button
                  onClick={() => send(QUIZ_ME_PROMPT)}
                  className="flex items-center gap-1 rounded-full border border-accent/30 bg-accent/5 px-3 py-1.5 text-xs font-medium text-accent hover:border-accent"
                >
                  <Sparkles className="h-3 w-3" />
                  Quiz me
                </button>
              )}
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm ${
                  m.role === "user" ? "bg-accent text-accent-foreground" : "bg-background/60"
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
    </>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  );
}
