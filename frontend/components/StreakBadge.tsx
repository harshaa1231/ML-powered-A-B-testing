"use client";

import { Flame, Zap } from "lucide-react";
import { useProgress } from "@/lib/progress";

export function StreakBadge() {
  const { progress } = useProgress();

  if (progress.streak.current === 0 && progress.xp === 0) return null;

  return (
    <div className="flex items-center gap-3 text-xs text-muted">
      {progress.streak.current > 0 && (
        <span className="flex items-center gap-1">
          <Flame className="h-3.5 w-3.5 text-amber-500" />
          {progress.streak.current} day{progress.streak.current === 1 ? "" : "s"}
        </span>
      )}
      {progress.xp > 0 && (
        <span className="flex items-center gap-1">
          <Zap className="h-3.5 w-3.5 text-accent" />
          {progress.xp} XP
        </span>
      )}
    </div>
  );
}
