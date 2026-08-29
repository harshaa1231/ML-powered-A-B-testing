"use client";

import { Briefcase, GraduationCap } from "lucide-react";
import { Label } from "@/components/ui";
import type { Persona } from "@/lib/types";

export function PersonaSelector({ value, onChange, label = "Account type" }: { value: Persona; onChange: (p: Persona) => void; label?: string }) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="grid grid-cols-2 gap-2">
        <PersonaOption icon={Briefcase} label="Business" active={value === "business"} onClick={() => onChange("business")} />
        <PersonaOption icon={GraduationCap} label="Learner" active={value === "learner"} onClick={() => onChange("learner")} />
      </div>
    </div>
  );
}

function PersonaOption({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: typeof Briefcase;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex flex-col items-center gap-1.5 rounded-xl border px-3 py-3 text-xs font-medium transition-colors ${
        active ? "border-accent bg-accent/10 text-accent" : "border-surface-border text-muted hover:bg-surface-2"
      }`}
    >
      <Icon className="h-4 w-4" />
      {label}
    </button>
  );
}
