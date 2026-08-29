"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, RotateCcw, Users } from "lucide-react";
import { twoProportionZTest } from "@/lib/stats";
import { Button } from "@/components/ui";

const CONTROL_RATE = 0.09;
const TREATMENT_RATE = 0.13;
const VISITORS_PER_TICK = [4, 12] as const;
const TICK_MS = 150;
const TARGET_VISITORS = 1800;
const SIGNIFICANCE_MIN_VISITORS = 150;

interface Variant {
  visitors: number;
  conversions: number;
}

function emptyVariant(): Variant {
  return { visitors: 0, conversions: 0 };
}

export function ABTestDemo() {
  const [control, setControl] = useState<Variant>(emptyVariant);
  const [treatment, setTreatment] = useState<Variant>(emptyVariant);
  const [running, setRunning] = useState(true);
  const [runId, setRunId] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!running) return;

    intervalRef.current = setInterval(() => {
      setControl((prevControl) => {
        const nextControl = tick(prevControl, CONTROL_RATE);
        setTreatment((prevTreatment) => {
          const nextTreatment = tick(prevTreatment, TREATMENT_RATE);
          if (nextControl.visitors >= TARGET_VISITORS && nextTreatment.visitors >= TARGET_VISITORS) {
            setRunning(false);
          }
          return nextTreatment;
        });
        return nextControl;
      });
    }, TICK_MS);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, runId]);

  function replay() {
    setControl(emptyVariant());
    setTreatment(emptyVariant());
    setRunning(true);
    setRunId((id) => id + 1);
  }

  const { pValue, uplift } = twoProportionZTest(control.conversions, control.visitors, treatment.conversions, treatment.visitors);
  const enoughData = control.visitors >= SIGNIFICANCE_MIN_VISITORS && treatment.visitors >= SIGNIFICANCE_MIN_VISITORS;
  const isSignificant = enoughData && pValue < 0.05;
  const progress = Math.min(100, (Math.min(control.visitors, treatment.visitors) / TARGET_VISITORS) * 100);

  return (
    <div className="glow-surface rounded-2xl border border-surface-border bg-surface p-2 shadow-2xl">
      <div className="flex items-center justify-between border-b border-surface-border px-4 py-2.5">
        <div className="flex items-center gap-1.5">
          <div className="h-2.5 w-2.5 rounded-full bg-danger/60" />
          <div className="h-2.5 w-2.5 rounded-full bg-amber-500/60" />
          <div className="h-2.5 w-2.5 rounded-full bg-success/60" />
        </div>
        <span className="text-[11px] font-medium text-muted">live A/B test simulation</span>
        <Button variant="ghost" size="sm" icon={RotateCcw} onClick={replay} className="!px-2 !py-1">
          Replay
        </Button>
      </div>

      <div className="p-5">
        <div className="grid grid-cols-2 gap-4">
          <VariantPanel label="Control (A)" variant={control} accent={false} />
          <VariantPanel label="Treatment (B)" variant={treatment} accent />
        </div>

        <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-surface-2">
          <motion.div className="h-full gradient-accent" animate={{ width: `${progress}%` }} transition={{ ease: "linear" }} />
        </div>

        <div className="mt-4 flex items-center justify-between rounded-xl border border-surface-border bg-surface-2/50 px-4 py-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">p-value (live)</p>
            <p className="mt-0.5 text-lg font-semibold tabular-nums">{enoughData ? pValue.toFixed(4) : "—"}</p>
          </div>
          <div className="text-right">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted">uplift</p>
            <p className="mt-0.5 text-lg font-semibold tabular-nums text-success">
              {enoughData ? `+${Math.max(uplift, 0).toFixed(1)}%` : "—"}
            </p>
          </div>
          <AnimatePresence>
            {isSignificant && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center gap-1.5 rounded-full bg-success/10 px-3 py-1.5 text-xs font-medium text-success ring-1 ring-inset ring-success/20"
              >
                <CheckCircle2 className="h-3.5 w-3.5" />
                Significant
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

function tick(prev: Variant, rate: number): Variant {
  if (prev.visitors >= TARGET_VISITORS) return prev;
  const [min, max] = VISITORS_PER_TICK;
  const newVisitors = min + Math.floor(Math.random() * (max - min + 1));
  let newConversions = 0;
  for (let i = 0; i < newVisitors; i++) {
    if (Math.random() < rate) newConversions++;
  }
  return { visitors: prev.visitors + newVisitors, conversions: prev.conversions + newConversions };
}

function VariantPanel({ label, variant, accent }: { label: string; variant: Variant; accent: boolean }) {
  const rate = variant.visitors > 0 ? (variant.conversions / variant.visitors) * 100 : 0;
  const barHeight = Math.min(100, rate * 6);

  return (
    <div className="rounded-xl border border-surface-border bg-surface-2/50 p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold">{label}</span>
        <span className="flex items-center gap-1 text-[11px] text-muted">
          <Users className="h-3 w-3" />
          {variant.visitors.toLocaleString()}
        </span>
      </div>

      <div className="mt-3 flex h-20 items-end">
        <motion.div
          className={`w-full rounded-t-md ${accent ? "gradient-accent" : "bg-surface-border"}`}
          animate={{ height: `${barHeight}%` }}
          transition={{ ease: "linear", duration: 0.15 }}
        />
      </div>

      <p className={`mt-3 text-xl font-semibold tabular-nums ${accent ? "text-accent" : ""}`}>{rate.toFixed(1)}%</p>
      <p className="text-[11px] text-muted">{variant.conversions.toLocaleString()} conversions</p>
    </div>
  );
}
