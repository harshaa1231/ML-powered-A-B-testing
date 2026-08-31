# AB Testing Pro

A full-stack experimentation platform with two distinct products in one codebase: a **Statsig-style experimentation workspace** for practitioners, and a **gamified, hands-on A/B testing course** for people learning the subject — both grounded in the same retrieval-augmented AI assistant (ABBot) and the same real statistics engine.

**Live**: [ml-powered-a-b-testing.vercel.app](https://ml-powered-a-b-testing.vercel.app) (frontend) · [ml-powered-a-b-testing.onrender.com/docs](https://ml-powered-a-b-testing.onrender.com/docs) (API docs) — both on free tiers, so the first request after idle time can take 30-50s to cold-start.

This is a from-scratch rewrite of an earlier single-file Streamlit prototype (preserved for reference under [`legacy-streamlit/`](legacy-streamlit/)) into a real product: a FastAPI backend with persistent Postgres storage, a Next.js frontend, and a Groq-powered RAG assistant threaded through five separate surfaces — all running on free-tier infrastructure, hardened against that free tier's actual limits rather than just assumed to fit.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system design, request flows, and the reasoning behind each infra choice.

## Two products, two accounts

Signing up asks whether you're **Business** or **Learner** — this isn't a cosmetic toggle. Business and learner are separate accounts enforced at the database level (a composite `(email, persona)` unique constraint, not a mutable flag on one account), so the same email address can hold both, independently. This is deliberate: it's what makes different pricing per persona possible later without a data migration.

### Business — the experimentation workspace

- **Hypothesis-first experiment creation** — a templated "we believe that ___ will ___ because ___" field, shown prominently on every result.
- **Sample Ratio Mismatch (SRM) health check** — a real chi-square goodness-of-fit test that runs automatically on every experiment and flags broken randomization before you trust the result.
- **Guardrail metrics / Scorecard** — pick secondary metrics (latency, churn, cost) alongside your primary one; each gets its own test result in a scorecard, and the column picker proactively suggests likely guardrail columns from your data.
- **A metrics catalog** — define what "Checkout Conversion" means once (name, description, column, guardrail or not) and reuse it by name on every future experiment instead of re-picking raw columns each time; a Scorecard row shows your metric's real name, not a raw column header.
- **A decision workflow** — record what actually happened after a result came in (shipped / rolled back), so an experiment is a real outcome, not a number read once and forgotten.
- **A real Experiments list** — sortable, filterable, separate from a lightweight Program Overview (KPI tiles, weekly trend, most-used test types).
- **RAG-grounded AI Summary** on every result — not a bare model call. It retrieves the relevant knowledge-base doc for the specific test type and health-check outcome, and is fed the actual computed numbers (group rates/means, guardrail results) so it explains *this* experiment, not a generic one.
- **ABBot, ambient** — a floating assistant available on every page, not a separate destination. Opened from an experiment, it already has that experiment's numbers in context.

### Learner — an actually interactive course, not static text

- **Skill tree** — a dependency-gated map (Foundational → Core → Advanced → Case Studies) instead of a flat scrolling article, with locked/unlocked/completed states.
- **Interactive sample-size calculator** and **significance simulator** — drag sliders, watch the real math (the same two-proportion z-test the backend runs) update live.
- **Inline quizzes** with instant feedback, and **streaks + XP** tracked locally per device.
- **Real-world case studies**, including one built on a real, published mobile-game retention experiment (Cookie Cats) — form a conclusion before the reveal, same as the actual Practice Lab flow.
- **Practice Lab** — pick a real sample dataset, state your conclusion, run the actual statistical test, then get RAG-grounded feedback from ABBot that's fed your real result (not a canned answer, and not free to invent numbers it wasn't given).
- **Searchable glossary** and a **"Quiz me"** starter prompt that reuses the same chat pipeline, no bespoke quiz backend.

### Where RAG actually runs

One retrieval pipeline (`app/rag/retriever.py`), reused across five surfaces rather than sprinkled everywhere for appearance: **chat**, **the floating ABBot widget**, **experiment AI summaries**, **Practice Lab feedback**, and **Program Analytics trends**. Each call is grounded both in the curated knowledge base (pgvector similarity search) and in the real numbers for that specific situation — the context builder forwards every number the stats engine actually computed and explicitly instructs the model never to invent metrics beyond what it's given (a real failure mode caught during manual testing and pinned down as a regression test).

**Bring your own data** — upload a CSV, TXT, MD, or PDF directly from the chat page and ABBot answers from it, everywhere it answers, not just that one conversation. Uploaded content is chunked and embedded exactly like the curated knowledge base and merged into one re-ranked retrieval pool, scoped to your account. A citation from your own upload is labeled "Yours" and clicking it shows the real content, same as a curated source.

**A conversation that behaves like one** — chat history is actually persisted and restored (not just stored and forgotten), with an explicit "New conversation" action for when you want to start over rather than continue an old thread. **Tuned against Groq's real free-tier ceiling, not a guess** — every account shares one API key, so the response-length budget was set by measuring an actual full exchange, and a real retry (using Groq's own reported reset time) kicks in before ever showing a broken or cut-off answer.

## Also true of the whole app, not just one persona

- **Light and dark**, switchable in the header, persisted per device, applied before hydration so there's no flash of the wrong theme on load.
- **A real mobile nav** — the sidebar collapses into a hamburger-triggered drawer below the `md` breakpoint, rather than just disappearing with no way back to it.
- **Rate-limited auth** — login and signup are throttled per IP (`slowapi`, in-memory) to blunt brute-force and spam-signup attempts, tuned to not trip on realistic use (one person creating both a business and a learner account).
- **99 backend tests**, `ruff` clean, `tsc`/`eslint`/`next build` clean — covering the statistics engine, the ML engine, every RAG surface, persona-scoped auth, rate limiting, and the metrics/decision/document-upload features, not just the happy path.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2 |
| Database | Postgres + pgvector (tested against Supabase; also works on Render/Neon/Fly) |
| ML/Stats | scikit-learn, SciPy, pandas |
| GenAI | Groq (`openai/gpt-oss-120b`, free tier) + `fastembed` (local, free, ONNX-runtime embeddings) |
| Document parsing | `pypdf` (PDF text extraction) + pandas (CSV → prose summary) for user uploads |
| Frontend | Next.js 16 (App Router), TypeScript, Tailwind CSS, Framer Motion, Recharts, `react-markdown` |
| Auth | JWT (signup/login, persona-scoped), bcrypt password hashing, `slowapi` rate limiting |
| Infra | Docker Compose (local), Render + Vercel (cloud), all free-tier |

## Quick start (local, Docker)

```bash
cp .env.example .env   # fill in GROQ_API_KEY (free from console.groq.com) and a JWT_SECRET
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs

## Quick start (local, no Docker)

**Backend** — needs Postgres with the `vector` extension (Supabase's free tier works well):

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL / DATABASE_URL_SYNC from your Postgres provider
alembic upgrade head
uvicorn app.main:app --reload
```

**Frontend**:

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

## Running tests

```bash
cd backend
pytest                 # pure unit tests (stats/ML engines, RAG chunking, persona prompts,
                        # anti-hallucination context building) run with no infra;
                        # API integration tests auto-skip unless DATABASE_URL_SYNC is reachable
ruff check .
```

```bash
cd frontend
npx tsc --noEmit
npm run lint
npm run build
```

## Project structure

```
backend/    FastAPI app — see docs/ARCHITECTURE.md for the module breakdown
frontend/   Next.js app — business workspace + learner course, shared component library
docs/       Architecture and deployment notes
legacy-streamlit/   The original single-file Streamlit prototype this project replaced
```

## Deployment

See the **Deployment** section of [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the Supabase/Render/Vercel setup. In short: managed Postgres with `pgvector` (Supabase free tier), backend on Render's free web service tier, frontend on Vercel's free tier, generation via Groq's free tier — the whole stack runs at zero cost.
