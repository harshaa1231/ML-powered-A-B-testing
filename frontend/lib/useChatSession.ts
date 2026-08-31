"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getLatestChatSession, sendChatMessage } from "@/lib/api";
import type { ChatSource } from "@/lib/types";

export interface ChatUiMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

const AUTO_QUESTION = "Can you walk me through this result and tell me what you'd do next?";

export interface UseChatSessionOptions {
  /** Restore the account's most recent conversation on mount. On by default for the
   * full /chat page and the floating widget; off for a scoped, single-purpose panel
   * (e.g. Practice Lab's follow-up box) where pulling in an unrelated past
   * conversation would be confusing rather than helpful. */
  restoreHistory?: boolean;
  /** Auto-ask the obvious first question when arriving with an experiment's context
   * and nothing to resume. Off where something else already serves as the opening
   * message (e.g. Practice Lab's own tailored feedback). */
  autoSend?: boolean;
}

/** Shared send-message logic for the full /chat page, the floating AbbotWidget, and
 * Practice Lab's follow-up box — one implementation, so these can't drift out of sync.
 *
 * Also restores the account's most recent conversation on mount instead of always
 * starting blank — chat history was persisted correctly all along, nothing ever
 * fetched it back, so every page visit (and every login) looked like total amnesia
 * even mid-conversation. */
export function useChatSession(experimentId?: string, options: UseChatSessionOptions = {}) {
  const { restoreHistory = true, autoSend = true } = options;
  const [messages, setMessages] = useState<ChatUiMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restored, setRestored] = useState(!restoreHistory);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoSentRef = useRef(false);

  const send = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      setError(null);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
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
    },
    [sessionId, experimentId]
  );

  // Restore the latest conversation once on mount.
  useEffect(() => {
    if (!restoreHistory) return;
    let cancelled = false;
    getLatestChatSession()
      .then((data) => {
        if (cancelled) return;
        if (data.session_id) {
          setSessionId(data.session_id);
          setMessages(
            data.messages.map((m) => ({
              role: m.role === "user" ? "user" : "assistant",
              content: m.content,
              sources: m.sources ?? undefined,
            }))
          );
        }
      })
      .catch(() => {
        // No reachable history — fine, just start fresh.
      })
      .finally(() => {
        if (!cancelled) setRestored(true);
      });
    return () => {
      cancelled = true;
    };
  }, [restoreHistory]);

  // Arriving with an experiment's context and truly nothing to resume — ask the
  // obvious first question automatically instead of showing an empty panel that
  // looks identical to opening chat with no context at all.
  useEffect(() => {
    if (autoSend && restored && experimentId && messages.length === 0 && !autoSentRef.current) {
      autoSentRef.current = true;
      send(AUTO_QUESTION);
    }
  }, [autoSend, restored, experimentId, messages.length, send]);

  // Persisted history is a real feature, not a trap: a user who wants to ask about
  // something unrelated needs an explicit way to start over, since simply refreshing
  // the page no longer clears anything (that's the point of restoring history).
  // Doesn't touch autoSentRef, so resetting won't re-trigger the auto-question above.
  const reset = useCallback(() => {
    setMessages([]);
    setSessionId(undefined);
    setError(null);
  }, []);

  return { messages, sending, error, send, reset, scrollRef };
}
