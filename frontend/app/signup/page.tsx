"use client";

import Link from "next/link";
import { Suspense, useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { ArrowRight, Sparkles } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { Button, Card, FadeIn, Input, Label } from "@/components/ui";
import { PersonaSelector } from "@/components/PersonaSelector";
import type { Persona } from "@/lib/types";

function SignupForm() {
  const { signup } = useAuth();
  const searchParams = useSearchParams();
  const initialPersona: Persona = searchParams.get("persona") === "learner" ? "learner" : "business";

  const [persona, setPersona] = useState<Persona>(initialPersona);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await signup(email, password, persona, fullName || undefined);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <FadeIn className="w-full max-w-sm">
      <Link href="/" className="mb-6 flex items-center justify-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg gradient-accent">
          <Sparkles className="h-4 w-4 text-accent-foreground" />
        </div>
        <span className="text-sm font-semibold tracking-tight">AB Testing Pro</span>
      </Link>
      <Card>
        <h1 className="text-xl font-semibold tracking-tight">Create your account</h1>

        <div className="mt-5">
          <PersonaSelector value={persona} onChange={setPersona} label="I'm signing up as" />
          <p className="mt-2 text-xs text-muted">
            Business and learner are separate accounts — even with the same email, you can have both.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          <div>
            <Label>Full name (optional)</Label>
            <Input value={fullName} onChange={(e) => setFullName(e.target.value)} />
          </div>
          <div>
            <Label>Email</Label>
            <Input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <Label>Password</Label>
            <Input type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          {error && <p className="text-sm text-danger">{error}</p>}
          <Button type="submit" className="w-full" loading={submitting} icon={submitting ? undefined : ArrowRight}>
            Create account
          </Button>
        </form>
        <p className="mt-6 text-center text-sm text-muted">
          Already have an account?{" "}
          <Link href="/login" className="font-medium text-accent hover:underline">
            Sign in
          </Link>
        </p>
      </Card>
    </FadeIn>
  );
}

export default function SignupPage() {
  return (
    <div className="relative flex flex-1 items-center justify-center p-6">
      <div
        className="pointer-events-none absolute left-1/2 top-0 -z-10 h-[400px] w-[700px] -translate-x-1/2 rounded-full opacity-20 blur-[120px]"
        style={{ background: "radial-gradient(circle, var(--accent), transparent 70%)" }}
        aria-hidden
      />
      <Suspense fallback={null}>
        <SignupForm />
      </Suspense>
    </div>
  );
}
