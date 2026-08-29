"use client";

import { useCallback, useRef, useState } from "react";
import { ApiError, sendChatMessage } from "@/lib/api";
import type { ChatSource } from "@/lib/types";

export interface ChatUiMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

/** Shared send-message logic for the full /chat page and the floating AbbotWidget —
 * one implementation, so the two surfaces can't drift out of sync. */
export function useChatSession(experimentId?: string) {
  const [messages, setMessages] = useState<ChatUiMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

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

  return { messages, sending, error, send, scrollRef };
}
