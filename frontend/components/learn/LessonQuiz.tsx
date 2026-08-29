"use client";

import { useState } from "react";
import { CheckCircle2, HelpCircle, XCircle } from "lucide-react";
import { Card } from "@/components/ui";
import { useProgress } from "@/lib/progress";

interface LessonQuizProps {
  id: string;
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export function LessonQuiz({ id, question, options, correctIndex, explanation }: LessonQuizProps) {
  const { recordQuizAnswer } = useProgress();
  const [selected, setSelected] = useState<number | null>(null);

  function choose(index: number) {
    if (selected !== null) return;
    setSelected(index);
    recordQuizAnswer(id, index === correctIndex);
  }

  return (
    <Card className="border-accent/20">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-accent">
        <HelpCircle className="h-3.5 w-3.5" />
        Quick check
      </div>
      <p className="mt-2 text-sm font-medium">{question}</p>

      <div className="mt-3 space-y-2">
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
        <p className="mt-3 text-sm text-muted">
          {selected === correctIndex ? "Correct — " : "Not quite — "}
          {explanation}
        </p>
      )}
    </Card>
  );
}
