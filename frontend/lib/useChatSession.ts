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

/** Shared send-message logic for the full /chat page and the floating AbbotWidget —
 * one implementation, so the two surfaces can't drift out of sync.
 *
 * Also restores the account's most recent conversation on mount instead of always
 * starting blank — chat history was persisted correctly all along, nothing ever
 * fetched it back, so every page visit (and every login) looked like total amnesia
 * even mid-conversation. */
export function useChatSession(experimentId?: string) {
  const [messages, setMessages] = useState<ChatUiMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restored, setRestored] = useState(false);
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
  }, []);

  // Arriving with an experiment's context and truly nothing to resume — ask the
  // obvious first question automatically instead of showing an empty panel that
  // looks identical to opening chat with no context at all.
  useEffect(() => {
    if (restored && experimentId && messages.length === 0 && !autoSentRef.current) {
      autoSentRef.current = true;
      send(AUTO_QUESTION);
    }
  }, [restored, experimentId, messages.length, send]);

  return { messages, sending, error, send, scrollRef };
}
