"use client";

import { useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";
import { applyTheme, getActiveTheme, subscribeToTheme, type Theme } from "@/lib/theme";

// `getServerSnapshot` returning null makes React use it for both the server render
// and the client's first (pre-hydration) render, so they match exactly; the real
// value from `getSnapshot` (which reads localStorage/matchMedia) only takes over
// immediately after hydration completes — the same guarantee an effect would give,
// without calling setState from inside one.
function getServerSnapshot(): Theme | null {
  return null;
}

/** Two icon-buttons (light / dark), not a single toggle switch — makes the current
 * choice legible at a glance instead of requiring the user to infer state from a
 * binary switch position. */
export function ThemeToggle({ className = "" }: { className?: string }) {
  const theme = useSyncExternalStore(subscribeToTheme, getActiveTheme, getServerSnapshot);

  return (
    <div className={`inline-flex items-center gap-0.5 rounded-lg border border-surface-border bg-surface-2/60 p-0.5 ${className}`}>
      <button
        type="button"
        onClick={() => applyTheme("light")}
        aria-label="Light background"
        aria-pressed={theme === "light"}
        className={`flex h-7 w-7 items-center justify-center rounded-md transition-colors ${
          theme === "light" ? "bg-surface text-foreground shadow-sm" : "text-muted hover:text-foreground"
        }`}
      >
        <Sun className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => applyTheme("dark")}
        aria-label="Dark background"
        aria-pressed={theme === "dark"}
        className={`flex h-7 w-7 items-center justify-center rounded-md transition-colors ${
          theme === "dark" ? "bg-surface text-foreground shadow-sm" : "text-muted hover:text-foreground"
        }`}
      >
        <Moon className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
