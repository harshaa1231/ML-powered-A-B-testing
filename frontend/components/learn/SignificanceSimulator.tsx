"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";
import { Card, Badge } from "@/components/ui";
import { twoProportionZTest } from "@/lib/stats";

export function SignificanceSimulator() {
  const [baseline, setBaseline] = useState(10);
  const [treatment, setTreatment] = useState(12);
  const [sampleSize, setSampleSize] = useState(500);

  const { pValue } = useMemo(() => {
    const controlSuccess = Math.round((baseline / 100) * sampleSize);
    const treatmentSuccess = Math.round((treatment / 100) * sampleSize);
    return twoProportionZTest(controlSuccess, sampleSize, treatmentSuccess, sampleSize);
  }, [baseline, treatment, sampleSize]);

  const isSignificant = pValue < 0.05;

  return (
    <Card>
      <h4 className="font-medium">Significance simulator</h4>
      <p className="mt-1 text-sm text-muted">
        Move the sliders and watch the p-value change — the same two-proportion z-test this app runs on real data.
      </p>

      <div className="mt-5 space-y-5">
        <SliderField label="Control rate" value={baseline} onChange={setBaseline} min={1} max={50} suffix="%" />
        <SliderField label="Treatment rate" value={treatment} onChange={setTreatment} min={1} max={50} suffix="%" />
        <SliderField label="Sample size per group" value={sampleSize} onChange={setSampleSize} min={50} max={5000} step={50} />
      </div>

      <div className="mt-5 flex items-end gap-6 rounded-xl border border-surface-border bg-surface-2/40 p-4">
        <Bar label="Control" pct={baseline} accent={false} />
        <Bar label="Treatment" pct={treatment} accent />
        <div className="ml-auto text-right">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted">p-value</p>
          <p className="text-2xl font-semibold tabular-nums">{pValue.toFixed(4)}</p>
          <Badge tone={isSignificant ? "success" : "neutral"} icon={isSignificant ? CheckCircle2 : XCircle}>
            {isSignificant ? "Significant" : "Not significant"}
          </Badge>
        </div>
      </div>
    </Card>
  );
}

function Bar({ label, pct, accent }: { label: string; pct: number; accent: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex h-24 w-10 items-end rounded-md bg-surface-border/50">
        <motion.div
          animate={{ height: `${Math.min(100, pct * 2)}%` }}
          transition={{ duration: 0.2 }}
          className={`w-full rounded-md ${accent ? "gradient-accent" : "bg-surface-border"}`}
        />
      </div>
      <span className="text-xs font-medium text-muted">{label}</span>
      <span className="text-xs tabular-nums">{pct}%</span>
    </div>
  );
}

function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  suffix = "",
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <label className="font-medium uppercase tracking-wider text-muted">{label}</label>
        <span className="font-semibold tabular-nums text-foreground">
          {value}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
      />
    </div>
  );
}
