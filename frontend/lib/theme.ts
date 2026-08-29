export type Theme = "light" | "dark";

const THEME_KEY = "abtesting_theme";

// Lets every ThemeToggle instance on the page (header, mobile drawer, etc.) stay in
// sync via useSyncExternalStore, since setting the DOM attribute directly doesn't
// itself trigger a React re-render anywhere.
const listeners = new Set<() => void>();

export function subscribeToTheme(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getStoredTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function applyTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // localStorage unavailable (private mode, blocked site data) — theme still
    // applies for this page load via the attribute below, just won't persist.
  }
  document.documentElement.setAttribute("data-theme", theme);
  listeners.forEach((listener) => listener());
}

/** What's actually rendered right now: the explicit override if set, otherwise
 * the system preference. Used to initialize the toggle's displayed state. */
export function getActiveTheme(): Theme {
  const stored = getStoredTheme();
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
