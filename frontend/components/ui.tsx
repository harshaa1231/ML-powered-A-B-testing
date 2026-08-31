"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { BookOpen, Loader2, X } from "lucide-react";
import { useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiError, getKbDocument, getUserDocument } from "@/lib/api";
import type { ChatSource, KBDocument } from "@/lib/types";

export function Card({
  children,
  className = "",
  glow = false,
}: {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={`rounded-2xl border border-surface-border bg-surface p-6 shadow-[0_1px_0_0_rgba(255,255,255,0.02)_inset] ${
        glow ? "glow-surface" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

type NativeButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "onDrag" | "onDragStart" | "onDragEnd" | "onAnimationStart" | "onAnimationEnd" | "onAnimationIteration"
>;

export function Button({
  className = "",
  variant = "primary",
  size = "md",
  icon: Icon,
  loading = false,
  children,
  ...props
}: NativeButtonProps & {
  variant?: "primary" | "secondary" | "ghost" | "outline";
  size?: "sm" | "md" | "lg";
  icon?: LucideIcon;
  loading?: boolean;
}) {
  const variants = {
    primary: "gradient-accent text-accent-foreground shadow-lg shadow-accent/20 hover:shadow-accent/30",
    secondary: "bg-surface-2 text-foreground hover:bg-surface-border",
    ghost: "text-foreground hover:bg-surface-2",
    outline: "border border-surface-border bg-transparent hover:bg-surface-2",
  }[variant];

  const sizes = {
    sm: "px-3 py-1.5 text-xs gap-1.5",
    md: "px-4 py-2 text-sm gap-2",
    lg: "px-6 py-3 text-base gap-2.5",
  }[size];

  return (
    <motion.button
      whileTap={{ scale: 0.97 }}
      className={`inline-flex items-center justify-center rounded-xl font-medium transition-all disabled:cursor-not-allowed disabled:opacity-50 ${variants} ${sizes} ${className}`}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : Icon ? <Icon className="h-4 w-4" /> : null}
      {children}
    </motion.button>
  );
}

export function Input({ className = "", ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-xl border border-surface-border bg-surface-2/60 px-3.5 py-2.5 text-sm text-foreground outline-none transition-all placeholder:text-muted/70 focus:border-accent focus:ring-4 focus:ring-[var(--ring)] ${className}`}
      {...props}
    />
  );
}

export function Select({ className = "", ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`w-full rounded-xl border border-surface-border bg-surface-2/60 px-3.5 py-2.5 text-sm text-foreground outline-none transition-all focus:border-accent focus:ring-4 focus:ring-[var(--ring)] ${className}`}
      {...props}
    />
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-muted">{children}</label>;
}

export function Badge({
  children,
  tone = "neutral",
  icon: Icon,
}: {
  children: ReactNode;
  tone?: "neutral" | "success" | "danger" | "accent";
  icon?: LucideIcon;
}) {
  const styles = {
    neutral: "bg-surface-2 text-muted ring-1 ring-inset ring-surface-border",
    success: "bg-success/10 text-success ring-1 ring-inset ring-success/20",
    danger: "bg-danger/10 text-danger ring-1 ring-inset ring-danger/20",
    accent: "bg-accent/10 text-accent ring-1 ring-inset ring-accent/20",
  }[tone];
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium ${styles}`}>
      {Icon && <Icon className="h-3 w-3" />}
      {children}
    </span>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return <Loader2 className={`h-4 w-4 animate-spin text-accent ${className}`} />;
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded-lg ${className}`} />;
}

export function StatTile({
  label,
  value,
  tone = "neutral",
  icon: Icon,
}: {
  label: string;
  value: string;
  tone?: "neutral" | "success" | "danger";
  icon?: LucideIcon;
}) {
  const toneClass = { neutral: "text-foreground", success: "text-success", danger: "text-danger" }[tone];
  return (
    <div className="rounded-xl border border-surface-border bg-surface-2/50 p-4">
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {label}
      </div>
      <div className={`mt-1.5 text-2xl font-semibold tracking-tight ${toneClass}`}>{value}</div>
    </div>
  );
}

export function EmptyState({
  title,
  description,
  icon: Icon,
  action,
}: {
  title: string;
  description: string;
  icon?: LucideIcon;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-surface-border p-14 text-center">
      {Icon && (
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-surface-2">
          <Icon className="h-5 w-5 text-muted" />
        </div>
      )}
      <p className="font-medium">{title}</p>
      <p className="mt-1 max-w-sm text-sm text-muted">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

// Matches the backend's USER_DOC_SLUG_PREFIX (app/rag/vector_store.py) — a citation
// with this prefix came from something the user uploaded themselves, not the
// curated knowledge base, and needs a different endpoint to view its content.
const USER_DOC_SLUG_PREFIX = "user-doc:";

/** The "visual confidence indicator" pattern: distinguishes an answer grounded in
 * retrieved knowledge-base content from one the model produced from general
 * reasoning alone, instead of burying that distinction in a small badge row.
 * Each source pill is a real citation — clicking it opens the actual document,
 * whether that's a curated KB doc or something the user uploaded themselves. */
export function GroundedIn({ sources }: { sources: ChatSource[] }) {
  const [openSlug, setOpenSlug] = useState<string | null>(null);
  const [doc, setDoc] = useState<KBDocument | null>(null);
  const [docError, setDocError] = useState<string | null>(null);

  function openSource(slug: string) {
    setOpenSlug(slug);
    setDoc(null);
    setDocError(null);
    const isUserDoc = slug.startsWith(USER_DOC_SLUG_PREFIX);
    const fetchDoc = isUserDoc
      ? getUserDocument(slug.slice(USER_DOC_SLUG_PREFIX.length)).then((d) => ({
          slug,
          title: d.filename,
          content: d.content,
        }))
      : getKbDocument(slug);
    fetchDoc.then(setDoc).catch((err) => setDocError(err instanceof ApiError ? err.message : "Couldn't load this document."));
  }

  if (sources.length === 0) {
    return (
      <div className="mt-2 flex items-center gap-1.5 text-[11px] text-muted">
        <span className="h-1.5 w-1.5 rounded-full bg-surface-border" />
        General reasoning — not grounded in a specific document
      </div>
    );
  }

  const unique = Array.from(new Map(sources.map((s) => [s.slug, s])).values());

  return (
    <>
      <div className="mt-2 rounded-lg border border-accent/20 bg-accent/5 px-2.5 py-2">
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-accent">
          <BookOpen className="h-3 w-3" />
          {unique.every((s) => s.slug.startsWith(USER_DOC_SLUG_PREFIX))
            ? "Grounded in your uploaded documents"
            : unique.some((s) => s.slug.startsWith(USER_DOC_SLUG_PREFIX))
              ? "Grounded in the knowledge base and your documents"
              : "Grounded in the knowledge base"}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {unique.map((s) => (
            <button
              key={s.slug}
              type="button"
              onClick={() => openSource(s.slug)}
              className="rounded-full bg-surface px-2 py-0.5 text-[11px] text-muted underline decoration-dotted underline-offset-2 transition-colors hover:text-accent"
            >
              {s.slug.startsWith(USER_DOC_SLUG_PREFIX) && <span className="text-accent">Yours: </span>}
              {s.title}
            </button>
          ))}
        </div>
      </div>

      {openSlug && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={() => setOpenSlug(null)}
        >
          <div
            className="max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-surface-border bg-surface p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <h3 className="text-base font-semibold">{doc?.title ?? "Loading..."}</h3>
              <button
                onClick={() => setOpenSlug(null)}
                className="rounded-md p-1 text-muted hover:bg-surface-2 hover:text-foreground"
                aria-label="Close"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-4">
              {docError ? (
                <p className="text-sm text-danger">{docError}</p>
              ) : doc ? (
                <Markdown>{doc.content}</Markdown>
              ) : (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <Skeleton className="h-4 w-4/6" />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** Renders LLM/RAG output (chat replies, AI summaries, practice feedback, trends
 * narratives) as formatted Markdown instead of literal `###`/`**`/`|table|` text —
 * every one of those surfaces returns real Markdown from the model. */
export function Markdown({ children, className = "" }: { children: string; className?: string }) {
  return (
    <div
      className={`space-y-2.5 text-sm leading-relaxed text-foreground [&_a]:text-accent [&_a]:underline [&_code]:rounded [&_code]:bg-surface-2 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[13px] [&_h1]:mt-3 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:mt-3 [&_h2]:text-base [&_h2]:font-semibold [&_h3]:mt-3 [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:uppercase [&_h3]:tracking-wide [&_h3]:text-muted [&_hr]:border-surface-border [&_li]:ml-4 [&_ol]:list-decimal [&_ol]:space-y-1 [&_p]:my-0 [&_strong]:font-semibold [&_strong]:text-foreground [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-surface-border [&_td]:px-2.5 [&_td]:py-1.5 [&_th]:border [&_th]:border-surface-border [&_th]:bg-surface-2 [&_th]:px-2.5 [&_th]:py-1.5 [&_th]:text-left [&_ul]:list-disc [&_ul]:space-y-1 ${className}`}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

export function FadeIn({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
