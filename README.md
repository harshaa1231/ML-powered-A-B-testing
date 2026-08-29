# AB Testing Pro

A full-stack A/B testing and experimentation platform: statistical testing, ML-powered uplift modeling, and a retrieval-augmented AI assistant grounded in both a curated knowledge base and your own live experiment results.

This is a from-scratch rewrite of an earlier single-file Streamlit prototype (preserved for reference under [`legacy-streamlit/`](legacy-streamlit/)) into a real product: a FastAPI backend with persistent Postgres storage, a Next.js frontend, and a Groq-powered RAG chat assistant — all running on free-tier infrastructure.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system design, request flows, and the reasoning behind each infra choice.

## Features

- **Statistical testing** — Welch's t-test, Mann-Whitney U, chi-square, and two-proportion z-tests, with the right test auto-recommended from your data.
- **ML Model Studio** — trains Gradient Boosting and Random Forest models (classification or regression, auto-detected), compares them, and reports feature importance.
- **Uplift modeling** — a T-learner estimates each user's individual treatment effect, so you can see who actually benefits instead of just the average effect.
- **Persistent experiment history** — every test and trained model is saved per-user in Postgres (the original prototype reset on every session).
- **ABBot: a real RAG assistant** — retrieves from a 13-document A/B testing knowledge base (pgvector similarity search over locally-computed embeddings) and, when relevant, your own live experiment numbers, then answers via Groq's free-tier `openai/gpt-oss-120b`.
- **Background ML training** — training runs as an async job with DB-tracked status, not blocking the request.
- **Sample datasets + synthetic data generator** — six pre-built industry datasets, plus a generator for larger, more realistic synthetic data per domain.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2 |
| Database | Postgres + pgvector (tested against Supabase; also works on Render/Neon/Fly) |
| ML/Stats | scikit-learn, SciPy, pandas |
| GenAI | Groq (`openai/gpt-oss-120b`, free tier) + `sentence-transformers` (local, free embeddings) |
| Frontend | Next.js 16 (App Router), TypeScript, Tailwind CSS, Recharts |
| Auth | JWT (signup/login), bcrypt password hashing |
| Infra | Docker Compose (local), Render/Fly + Vercel (cloud) |

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
pytest                 # pure unit tests (stats/ML engines, RAG chunking) run with no infra;
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
frontend/   Next.js app
docs/       Architecture and deployment notes
legacy-streamlit/   The original single-file Streamlit prototype this project replaced
```

## Deployment

See the **Deployment** section of [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the Supabase/Render/Fly/Vercel setup.
