"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useState, type FormEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { MessageCircle, Maximize2, RotateCcw, Sparkles, X } from "lucide-react";
import { useChatSession } from "@/lib/useChatSession";
import { Button, GroundedIn, Input, Markdown, Spinner } from "@/components/ui";

const EXPERIMENT_DETAIL_PATTERN = /^\/experiments\/(?!new$)([\w-]+)$/;

export function AbbotWidget() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");

  const experimentId = useMemo(() => {
    const match = pathname.match(EXPERIMENT_DETAIL_PATTERN);
    return match ? match[1] : undefined;
  }, [pathname]);

  const isStudyContext = pathname.startsWith("/learn") || pathname.startsWith("/practice");

  const { messages, sending, error, send, reset, scrollRef } = useChatSession(experimentId);

  // Full-page /chat already covers this experience for a longer conversation — don't double up the FAB there.
  if (pathname.startsWith("/chat")) return null;

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input;
    setInput("");
    send(text);
  }

  return (
    <>
      <motion.button
        whileTap={{ scale: 0.92 }}
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-6 right-6 z-40 flex h-14 w-14 items-center justify-center rounded-full gradient-accent shadow-xl shadow-accent/30"
        aria-label="Ask ABBot"
      >
        {open ? <X className="h-5 w-5 text-accent-foreground" /> : <MessageCircle className="h-5 w-5 text-accent-foreground" />}
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.97 }}
            transition={{ duration: 0.18 }}
            className="fixed bottom-24 right-6 z-40 flex h-[32rem] w-96 max-w-[calc(100vw-3rem)] flex-col overflow-hidden rounded-2xl border border-surface-border bg-surface shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-surface-border px-4 py-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-accent" />
                <span className="text-sm font-semibold">ABBot</span>
                {experimentId && <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] text-accent">this result</span>}
                {!experimentId && isStudyContext && (
                  <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] text-accent">study companion</span>
                )}
              </div>
              <div className="flex items-center gap-1">
                {messages.length > 0 && (
                  <button
                    onClick={reset}
                    className="rounded-md p-1 text-muted hover:bg-surface-2 hover:text-foreground"
                    aria-label="New conversation"
                    title="New conversation"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                )}
                <Link href="/chat" className="rounded-md p-1 text-muted hover:bg-surface-2 hover:text-foreground" aria-label="Open full chat">
                  <Maximize2 className="h-3.5 w-3.5" />
                </Link>
              </div>
            </div>

            <div className="flex-1 space-y-3 overflow-y-auto p-4">
              {messages.length === 0 && (
                <p className="text-xs text-muted">
                  {experimentId
                    ? "Ask about this experiment's result — I have its numbers."
                    : "Ask me anything about A/B testing, or your work here."}
                </p>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[85%] rounded-xl px-3 py-2 text-xs ${
                      m.role === "user" ? "bg-accent text-accent-foreground" : "bg-background/60"
                    }`}
                  >
                    {m.role === "assistant" ? (
                      <Markdown className="text-xs">{m.content}</Markdown>
                    ) : (
                      <p className="whitespace-pre-wrap">{m.content}</p>
                    )}
                    {m.role === "assistant" && <GroundedIn sources={m.sources ?? []} />}
                  </div>
                </div>
              ))}
              {sending && <Spinner className="h-3.5 w-3.5" />}
              <div ref={scrollRef} />
            </div>

            {error && <p className="px-4 text-xs text-danger">{error}</p>}

            <form onSubmit={handleSubmit} className="flex gap-2 border-t border-surface-border p-3">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask ABBot..."
                className="text-sm"
              />
              <Button type="submit" size="sm" disabled={sending || !input.trim()}>
                Send
              </Button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
