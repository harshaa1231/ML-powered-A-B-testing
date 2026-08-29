"use client";

import { useMemo, useState } from "react";
import { Card } from "@/components/ui";

// Standard two-proportion sample-size formula. z-scores for common significance/power
// choices, hardcoded rather than computing an inverse-normal CDF for two dropdowns.
const Z_ALPHA: Record<string, number> = { "0.05": 1.96, "0.01": 2.576 };
const Z_BETA: Record<string, number> = { "0.8": 0.84, "0.9": 1.282 };

function requiredSampleSize(p1: number, mde: number, zAlpha: number, zBeta: number): number {
  const p2 = Math.min(0.99, p1 + mde);
  const pooled = p1 * (1 - p1) + p2 * (1 - p2);
  const n = (Math.pow(zAlpha + zBeta, 2) * pooled) / Math.pow(p2 - p1, 2);
  return Math.ceil(n);
}

export function SampleSizeCalculator() {
  const [baseline, setBaseline] = useState(10);
  const [mde, setMde] = useState(2);
  const [power, setPower] = useState("0.8");
  const [alpha, setAlpha] = useState("0.05");

  const n = useMemo(
    () => requiredSampleSize(baseline / 100, mde / 100, Z_ALPHA[alpha], Z_BETA[power]),
    [baseline, mde, alpha, power]
  );

  return (
    <Card>
      <h4 className="font-medium">Sample size calculator</h4>
      <p className="mt-1 text-sm text-muted">Drag the sliders — the required sample size per group updates live.</p>

      <div className="mt-5 space-y-5">
        <SliderField label="Baseline conversion rate" value={baseline} onChange={setBaseline} min={1} max={50} suffix="%" />
        <SliderField label="Minimum detectable effect (absolute)" value={mde} onChange={setMde} min={0.5} max={20} step={0.5} suffix=" pts" />

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wider text-muted">Significance level</label>
            <select
              value={alpha}
              onChange={(e) => setAlpha(e.target.value)}
              className="w-full rounded-lg border border-surface-border bg-surface-2/60 px-3 py-1.5 text-sm outline-none focus:border-accent"
            >
              <option value="0.05">0.05 (95% confidence)</option>
              <option value="0.01">0.01 (99% confidence)</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium uppercase tracking-wider text-muted">Statistical power</label>
            <select
              value={power}
              onChange={(e) => setPower(e.target.value)}
              className="w-full rounded-lg border border-surface-border bg-surface-2/60 px-3 py-1.5 text-sm outline-none focus:border-accent"
            >
              <option value="0.8">80%</option>
              <option value="0.9">90%</option>
            </select>
          </div>
        </div>
      </div>

      <div className="mt-5 rounded-xl border border-accent/20 bg-accent/5 p-4 text-center">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted">Required sample size per group</p>
        <p className="mt-1 text-3xl font-semibold tabular-nums text-accent">{n.toLocaleString()}</p>
        <p className="mt-1 text-xs text-muted">
          To detect a {baseline}% → {(baseline + mde).toFixed(1)}% change with {Math.round(Number(power) * 100)}% power.
        </p>
      </div>
    </Card>
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
