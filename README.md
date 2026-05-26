# Corpus Hybrid RAG

A backend RAG system I built to get hands-on with hybrid retrieval. The idea was to implement everything from scratch rather than using LangChain or LlamaIndex — wanted to understand what's actually happening inside these systems.

You upload PDFs, they get chunked and indexed into Qdrant, and then you can query against them. Answers come back with inline citations pointing to the source chunks.

---

## What it does

- Parses PDFs using `unstructured`, falls back to `pypdf` if that fails
- Chunks text semantically (splits at embedding similarity drops) with a recursive fallback
- Indexes both dense vectors (bge-m3) and sparse BM25 vectors into a single Qdrant collection
- Supports four retrieval strategies: dense, sparse, hybrid (RRF fusion), and hybrid with cross-encoder reranking
- Has a query intelligence layer — HyDE, query decomposition, conversation rewriting, and an intent classifier that picks the retrieval strategy automatically
- Generates grounded answers via gpt-4o-mini with source citations, supports streaming
- Evaluation harness using RAGAS to compare retrieval strategies against a golden dataset

Ingestion is async — you upload a PDF and poll for status while the worker processes it in the background.

---

## Stack

FastAPI, Qdrant, PostgreSQL, Redis + Arq, BAAI/bge-m3, LiteLLM, RAGAS

---

## Running locally

You'll need Docker, [uv](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
# start postgres, qdrant, redis
docker compose up -d

# install deps and run migrations
uv sync
uv run alembic upgrade head
```

Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY`.

Then start the API and worker in two terminals:

```bash
uv run uvicorn app.main:app --reload
uv run arq app.worker.WorkerSettings
```

API runs at `http://localhost:8000`. Docs at `/docs`.

All requests need `Authorization: Bearer dev-secret-key` (or whatever you set `API_KEY` to in `.env`).

---

## Basic flow

```
POST /documents/          → upload a PDF, get back a doc_id
GET  /documents/{id}/status → poll until status = "done"
POST /search/             → retrieve relevant chunks
POST /query/              → retrieve + generate an answer
POST /query/stream        → same but streams tokens as SSE
```

---
## Evaluation

Once you've ingested some documents, update `eval/golden.yaml` with real queries and run:

```bash
corpus eval compare --strategies dense,hybrid,hybrid_rerank --openai-key $OPENAI_API_KEY
```

Prints a table comparing faithfulness, answer relevancy, context precision, and recall across strategies.
