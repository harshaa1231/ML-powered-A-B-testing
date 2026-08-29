"use client";

import { CheckCircle2, Lock } from "lucide-react";

export interface SkillNode {
  id: string;
  title: string;
  tier: string;
  prerequisites: string[];
}

interface SkillTreeProps {
  tiers: { key: string; label: string }[];
  nodes: SkillNode[];
  completedIds: string[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function SkillTree({ tiers, nodes, completedIds, selectedId, onSelect }: SkillTreeProps) {
  function isLocked(node: SkillNode): boolean {
    return node.prerequisites.some((p) => !completedIds.includes(p));
  }

  return (
    <div className="rounded-2xl border border-surface-border bg-surface p-6">
      {tiers.map((tier, tierIndex) => {
        const tierNodes = nodes.filter((n) => n.tier === tier.key);
        if (tierNodes.length === 0) return null;

        return (
          <div key={tier.key}>
            <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-muted">{tier.label}</p>
            <div className="flex flex-wrap gap-3">
              {tierNodes.map((node) => {
                const locked = isLocked(node);
                const completed = completedIds.includes(node.id);
                const selected = selectedId === node.id;

                return (
                  <button
                    key={node.id}
                    onClick={() => !locked && onSelect(node.id)}
                    disabled={locked}
                    className={`flex items-center gap-2 rounded-xl border px-3.5 py-2.5 text-sm font-medium transition-colors ${
                      selected
                        ? "border-accent bg-accent/10 text-accent"
                        : completed
                          ? "border-success/30 bg-success/5 text-success hover:border-success/50"
                          : locked
                            ? "cursor-not-allowed border-surface-border text-muted/50"
                            : "border-surface-border text-foreground hover:border-accent/40 hover:bg-surface-2"
                    }`}
                  >
                    {locked ? (
                      <Lock className="h-3.5 w-3.5 shrink-0" />
                    ) : completed ? (
                      <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                    ) : null}
                    {node.title}
                  </button>
                );
              })}
            </div>

            {tierIndex < tiers.length - 1 && (
              <div className="my-4 ml-4 h-4 w-px bg-surface-border" aria-hidden />
            )}
          </div>
        );
      })}
    </div>
  );
}
