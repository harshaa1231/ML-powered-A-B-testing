"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import * as api from "./api";
import { ApiError } from "./api";
import type { Persona, User } from "./types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string, persona: Persona) => Promise<void>;
  signup: (email: string, password: string, persona: Persona, fullName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  // Always true on both the server render and the client's first render — reading
  // localStorage here (even via a lazy initializer) would make the client's first paint
  // differ from the server's (which has no localStorage at all), causing a hydration
  // mismatch. The real check happens in the effect below, which only ever runs client-side.
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    async function restoreSession() {
      const token = api.getToken();
      if (!token) {
        setLoading(false);
        return;
      }
      // Render's free tier spins the backend down after ~15 minutes idle, and the next
      // request can take 30-50s to cold-start — that used to look identical to "your
      // token is invalid" and silently logged people out. Only a real 401 means that;
      // anything else (timeout, connection refused, 502 while cold-starting) gets
      // retried with backoff instead, and the token is never touched on those.
      const maxAttempts = 4;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          setUser(await api.getMe());
          break;
        } catch (err) {
          if (err instanceof ApiError && err.status === 401) {
            api.clearToken();
            break;
          }
          if (attempt === maxAttempts) break;
          await new Promise((resolve) => setTimeout(resolve, attempt * 4000));
        }
      }
      setLoading(false);
    }
    restoreSession();
  }, []);

  async function login(email: string, password: string, persona: Persona) {
    const { access_token } = await api.login(email, password, persona);
    api.setToken(access_token);
    setUser(await api.getMe());
    router.push("/dashboard");
  }

  async function signup(email: string, password: string, persona: Persona, fullName?: string) {
    const { access_token } = await api.signup(email, password, persona, fullName);
    api.setToken(access_token);
    setUser(await api.getMe());
    router.push("/dashboard");
  }

  function logout() {
    api.clearToken();
    setUser(null);
    router.push("/login");
  }

  return <AuthContext.Provider value={{ user, loading, login, signup, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
