"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card, FadeIn, Input } from "@/components/ui";
import { GLOSSARY_TERMS } from "@/lib/glossaryData";

const CATEGORY_LABELS: Record<string, string> = {
  core: "Core concepts",
  stats: "Statistics",
  advanced: "Advanced",
};

export default function GlossaryPage() {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return GLOSSARY_TERMS;
    return GLOSSARY_TERMS.filter((t) => t.term.toLowerCase().includes(q) || t.definition.toLowerCase().includes(q));
  }, [query]);

  const grouped = useMemo(() => {
    const groups: Record<string, typeof GLOSSARY_TERMS> = {};
    for (const term of filtered) {
      (groups[term.category] ??= []).push(term);
    }
    return groups;
  }, [filtered]);

  return (
    <AppShell>
      <FadeIn>
        <h1 className="text-2xl font-semibold tracking-tight">Glossary</h1>
        <p className="mt-1 text-sm text-muted">
          Quick lookups, not a course — for the full walkthrough, head to{" "}
          <a href="/learn" className="text-accent hover:underline">
            Learn A/B Testing
          </a>
          .
        </p>
      </FadeIn>

      <FadeIn delay={0.05} className="relative mt-6 max-w-md">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search terms..."
          className="pl-9"
        />
      </FadeIn>

      <div className="mt-8 space-y-8">
        {Object.entries(CATEGORY_LABELS).map(([key, label]) => {
          const terms = grouped[key];
          if (!terms || terms.length === 0) return null;
          return (
            <div key={key}>
              <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">{label}</h2>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {terms.map((t, i) => (
                  <FadeIn key={t.term} delay={i * 0.02}>
                    <Card>
                      <h3 className="font-medium">{t.term}</h3>
                      <p className="mt-1.5 text-sm leading-relaxed text-muted">{t.definition}</p>
                    </Card>
                  </FadeIn>
                ))}
              </div>
            </div>
          );
        })}

        {filtered.length === 0 && <p className="text-sm text-muted">No terms match &quot;{query}&quot;.</p>}
      </div>
    </AppShell>
  );
}
