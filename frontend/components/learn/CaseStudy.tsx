"use client";

import { useState } from "react";
import { BookOpen, CheckCircle2, XCircle } from "lucide-react";
import { Badge, Card } from "@/components/ui";
import { useProgress } from "@/lib/progress";

interface CaseStudyProps {
  id: string;
  title: string;
  narrative: string;
  stat: { label: string; control: string; treatment: string };
  question: string;
  options: string[];
  correctIndex: number;
  reveal: string;
}

export function CaseStudy({ id, title, narrative, stat, question, options, correctIndex, reveal }: CaseStudyProps) {
  const { completeCaseStudy } = useProgress();
  const [selected, setSelected] = useState<number | null>(null);

  function choose(index: number) {
    if (selected !== null) return;
    setSelected(index);
    completeCaseStudy(id);
  }

  return (
    <Card>
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted">
        <BookOpen className="h-3.5 w-3.5" />
        Case study
      </div>
      <h4 className="mt-2 font-medium">{title}</h4>
      <p className="mt-2 text-sm leading-relaxed text-muted">{narrative}</p>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-surface-border bg-surface-2/40 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">Control — {stat.label}</p>
          <p className="mt-1 text-lg font-semibold tabular-nums">{stat.control}</p>
        </div>
        <div className="rounded-lg border border-surface-border bg-surface-2/40 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">Treatment — {stat.label}</p>
          <p className="mt-1 text-lg font-semibold tabular-nums">{stat.treatment}</p>
        </div>
      </div>

      <p className="mt-4 text-sm font-medium">{question}</p>
      <div className="mt-2 space-y-2">
        {options.map((option, i) => {
          const isSelected = selected === i;
          const isCorrect = i === correctIndex;
          const showState = selected !== null;
          return (
            <button
              key={i}
              onClick={() => choose(i)}
              disabled={selected !== null}
              className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition-colors ${
                showState && isCorrect
                  ? "border-success/40 bg-success/10 text-success"
                  : showState && isSelected
                    ? "border-danger/40 bg-danger/10 text-danger"
                    : "border-surface-border hover:bg-surface-2"
              }`}
            >
              {option}
              {showState && isCorrect && <CheckCircle2 className="h-4 w-4 shrink-0" />}
              {showState && isSelected && !isCorrect && <XCircle className="h-4 w-4 shrink-0" />}
            </button>
          );
        })}
      </div>

      {selected !== null && (
        <div className="mt-3 rounded-lg border border-accent/20 bg-accent/5 p-3">
          <Badge tone="accent">What actually happened</Badge>
          <p className="mt-2 text-sm text-muted">{reveal}</p>
        </div>
      )}
    </Card>
  );
}
