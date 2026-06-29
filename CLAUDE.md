# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the data ingestion server (port 8000)
uv run uvicorn app.import_process.api.file_import_server:app --host 0.0.0.0 --port 8000

# Run the query/retrieval server (port 8001)
uv run uvicorn app.query_process.api.query_server:app --host 0.0.0.0 --port 8001

# Download required ML models
uv run python app/tool/download_bgem3.py
uv run python app/tool/download_reranker.py
```

There are no tests in this repository.

## Architecture

This is a two-module enterprise RAG system built on **LangGraph** state machines with **FastAPI** serving.

### Two Core Pipelines

**Data Ingestion** (`app/import_process/`) — PDF/Markdown → Vector DB  
API on port 8000. LangGraph workflow: PDF→Markdown conversion (MinerU) → image extraction to MinIO → hierarchical text chunking → entity name recognition (LLM) → BGE-M3 dense+sparse embedding → Milvus storage.

**Intelligent Retrieval** (`app/query_process/`) — Query → Streamed Answer  
API on port 8001. LangGraph workflow with a 4-way parallel fan-out search: dense+sparse vector search, HyDE (Hypothetical Document Embeddings) search, Tavily web search, and Neo4j Knowledge Graph query. Results are merged via Reciprocal Rank Fusion (RRF), re-scored by BGE-Reranker, then fed to GPT-4o-mini for answer generation streamed back via SSE.

### Key Design Patterns

- **LangGraph state machines**: Each pipeline is a `StateGraph` with a `TypedDict` state (`ImportGraphState`, `QueryGraphState`). Nodes are independent functions that read/write to shared state. The query pipeline uses virtual "fan-out/join" nodes for parallel search branches.
- **Singleton service clients**: All external service connections (Milvus, MinIO, MongoDB, Neo4j) use the singleton pattern in `app/clients/`.
- **Environment-driven config**: All secrets, model paths, and endpoints are loaded from `.env` via modules in `app/conf/`. See README for required variables.
- **Prompt templates**: Stored as `.prompt` files in `prompts/`, loaded at runtime by `app/core/load_prompt.py`.
- **SSE streaming**: `app/utils/sse_utils.py` manages token queues; the query server streams LLM output token-by-token via Server-Sent Events.

### External Dependencies (must be running)

Milvus (vector DB), MinIO (object storage), MongoDB (session history), Neo4j (knowledge graph) — all run externally, typically via Docker.

### Module Map

| Path | Purpose |
|------|---------|
| `app/import_process/agent/main_graph.py` | Ingestion LangGraph definition |
| `app/query_process/agent/main_graph.py` | Retrieval LangGraph definition |
| `app/lm/` | BGE-M3 embedding, BGE-Reranker, LLM wrappers |
| `app/clients/` | Milvus, MinIO, MongoDB, Neo4j clients |
| `app/conf/` | Config modules (read from `.env`) |
| `app/utils/` | RRF scoring, SSE, task tracking, sparse vector utils |
| `prompts/` | Prompt templates for each LLM call |
