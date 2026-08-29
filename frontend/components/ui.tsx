"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

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
