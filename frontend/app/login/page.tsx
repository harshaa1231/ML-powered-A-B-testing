"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { ArrowRight, Sparkles } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { Button, Card, FadeIn, Input, Label } from "@/components/ui";

export default function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="relative flex flex-1 items-center justify-center p-6">
      <div
        className="pointer-events-none absolute left-1/2 top-0 -z-10 h-[400px] w-[700px] -translate-x-1/2 rounded-full opacity-20 blur-[120px]"
        style={{ background: "radial-gradient(circle, var(--accent), transparent 70%)" }}
        aria-hidden
      />
      <FadeIn className="w-full max-w-sm">
        <Link href="/" className="mb-6 flex items-center justify-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg gradient-accent">
            <Sparkles className="h-4 w-4 text-accent-foreground" />
          </div>
          <span className="text-sm font-semibold tracking-tight">AB Testing Pro</span>
        </Link>
        <Card>
          <h1 className="text-xl font-semibold tracking-tight">Sign in</h1>
          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <Label>Email</Label>
              <Input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <Label>Password</Label>
              <Input type="password" required value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>
            {error && <p className="text-sm text-danger">{error}</p>}
            <Button type="submit" className="w-full" loading={submitting} icon={submitting ? undefined : ArrowRight}>
              Sign in
            </Button>
          </form>
          <p className="mt-6 text-center text-sm text-muted">
            No account?{" "}
            <Link href="/signup" className="font-medium text-accent hover:underline">
              Create one
            </Link>
          </p>
        </Card>
      </FadeIn>
    </div>
  );
}
