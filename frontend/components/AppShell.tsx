"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  BarChart3,
  Beaker,
  BookOpen,
  FlaskConical,
  Gauge,
  GraduationCap,
  LayoutDashboard,
  ListChecks,
  LogOut,
  Menu,
  MessageCircle,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { Spinner } from "./ui";
import { AbbotWidget } from "./AbbotWidget";
import { ThemeToggle } from "./ThemeToggle";
import type { Persona } from "@/lib/types";

const WORK_ITEMS = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/experiments", label: "Experiments", icon: ListChecks },
  { href: "/metrics", label: "Metrics", icon: Gauge },
  { href: "/ml-studio", label: "ML Model Studio", icon: FlaskConical },
  { href: "/predictions", label: "Predictions", icon: Target },
  { href: "/datasets", label: "Sample Datasets", icon: BarChart3 },
];

const ASK_ABBOT_ITEM = { href: "/chat", label: "Ask ABBot", icon: MessageCircle };
const GLOSSARY_ITEM = { href: "/glossary", label: "Glossary", icon: BookOpen };

const LEARN_ITEMS = [
  ASK_ABBOT_ITEM,
  { href: "/learn", label: "Learn A/B Testing", icon: GraduationCap },
  { href: "/practice", label: "Practice Lab", icon: Beaker },
  GLOSSARY_ITEM,
];

// Business gets quick-reference tools (ask a question, look up a term), not the full
// course — a practitioner who already knows the material shouldn't land in what reads
// as a "Learn" section with a skill tree and a practice quiz waiting for them.
const RESOURCE_ITEMS = [ASK_ABBOT_ITEM, GLOSSARY_ITEM];

const SECTIONS_FOR: Record<Persona, { label: string; items: typeof WORK_ITEMS }[]> = {
  business: [
    { label: "Workspace", items: WORK_ITEMS },
    { label: "Resources", items: RESOURCE_ITEMS },
  ],
  learner: [
    { label: "Learn", items: LEARN_ITEMS },
    { label: "Workspace", items: WORK_ITEMS },
  ],
};

const ALL_ITEMS = [...WORK_ITEMS, ...LEARN_ITEMS];

function pageTitleFor(pathname: string): string {
  if (pathname.startsWith("/experiments/new")) return "New Experiment";
  if (/^\/experiments\/[\w-]+$/.test(pathname)) return "Experiment Result";
  const match = ALL_ITEMS.find((item) => pathname.startsWith(item.href));
  return match?.label ?? "AB Testing Pro";
}

export function AppShell({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }

  if (!user) return null;

  const initial = (user.full_name || user.email)[0]?.toUpperCase();
  const sections = SECTIONS_FOR[user.persona];

  const navSections = (onNavigate?: () => void) => (
    <nav className="flex-1 space-y-4 px-3">
      {sections.map((section) => (
        <div key={section.label}>
          <p className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">{section.label}</p>
          <div className="space-y-0.5">
            {section.items.map((item) => {
              const active = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={onNavigate}
                  className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    active ? "bg-surface-2 text-foreground" : "text-muted hover:bg-surface-2/60 hover:text-foreground"
                  }`}
                >
                  <item.icon className={`h-4 w-4 ${active ? "text-accent" : ""}`} />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </nav>
  );

  const accountRow = (
    <div className="border-t border-surface-border p-4">
      <div className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-2 text-xs font-semibold text-muted">
          {initial}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium">{user.full_name || "Account"}</p>
          <p className="truncate text-xs text-muted">{user.email}</p>
        </div>
        <button onClick={logout} className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-foreground" aria-label="Sign out">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen bg-background">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-surface-border bg-surface md:flex">
        <Link href="/dashboard" className="flex items-center gap-2 px-6 py-6">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg gradient-accent">
            <Sparkles className="h-4 w-4 text-accent-foreground" />
          </div>
          <span className="text-sm font-semibold tracking-tight">AB Testing Pro</span>
        </Link>
        {navSections()}
        {accountRow}
      </aside>

      <AnimatePresence>
        {mobileNavOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileNavOpen(false)}
              className="fixed inset-0 z-40 bg-black/40 md:hidden"
            />
            <motion.aside
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "tween", duration: 0.2 }}
              className="fixed inset-y-0 left-0 z-50 flex w-72 flex-col bg-surface md:hidden"
            >
              <div className="flex items-center justify-between px-6 py-6">
                <Link href="/dashboard" className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg gradient-accent">
                    <Sparkles className="h-4 w-4 text-accent-foreground" />
                  </div>
                  <span className="text-sm font-semibold tracking-tight">AB Testing Pro</span>
                </Link>
                <button
                  onClick={() => setMobileNavOpen(false)}
                  className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-foreground"
                  aria-label="Close menu"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              {navSections(() => setMobileNavOpen(false))}
              {accountRow}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-surface-border px-4 md:px-10">
          <button
            onClick={() => setMobileNavOpen(true)}
            className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-foreground md:hidden"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
          <h1 className="flex-1 text-sm font-medium text-muted">{pageTitleFor(pathname)}</h1>
          <ThemeToggle />
        </header>
        <main className="flex-1 overflow-y-auto p-4 md:p-10">{children}</main>
      </div>

      <AbbotWidget />
    </div>
  );
}
