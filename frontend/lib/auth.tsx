"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import * as api from "./api";
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
      try {
        setUser(await api.getMe());
      } catch {
        api.clearToken();
      } finally {
        setLoading(false);
      }
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
