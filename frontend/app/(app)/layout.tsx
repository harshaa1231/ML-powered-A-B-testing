"use client";

import { AppShell } from "@/components/AppShell";

// A real Next.js layout, not a per-page wrapper: it mounts once and persists across
// navigation within this route group, instead of being torn down and rebuilt on
// every page change. That's the fix for a real bug — the floating ABBot widget used
// to lose its entire conversation (and the sidebar/theme-toggle their own state)
// every time you clicked to a different page, because each page previously wrapped
// itself in <AppShell> independently rather than sharing one persistent instance.
export default function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
