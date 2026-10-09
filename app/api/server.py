"""FastAPI server for Prophet RAG queries."""

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import load_config
from app.rag.llm import LLMRefused

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = "I'm not able to offer counsel on that question. Please try asking in a different way."


class RateLimiter:
    """Per-key sliding window over the last 60 seconds. A limit of 0 disables it."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        window = self._hits[key]
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Prophet RAG API starting up")
    yield
    logger.info("Prophet RAG API shutting down")


_server_cfg = load_config().get("server", {}) or {}
limiter = RateLimiter(int(_server_cfg.get("rate_limit_per_minute", 10)))

app = FastAPI(
    title="Prophet RAG",
    description="Answer queries using writings of LDS modern prophets with full citations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_server_cfg.get("cors_origins") or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

web_dir = Path(__file__).resolve().parents[2] / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    if not limiter.allow(client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")


@app.get("/")
async def serve_ui():
    return FileResponse(str(web_dir / "ui.html"))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class QuoteSource(BaseModel):
    quote: str
    speaker: str | None
    title: str
    source: str
    date: str | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuoteSource] = Field(..., description="Full quotes with citations used to generate the answer")


def _docs_to_sources(results) -> list[QuoteSource]:
    return [
        QuoteSource(
            quote=r.content,
            speaker=str(r.speaker) if r.speaker else None,
            title=str(r.source_title),
            source=r.source_type or "Unknown",
            date=str(r.date) if r.date else None,
        )
        for r in results
    ]


# Sync handler on purpose: FastAPI runs it in a worker thread so the LLM call doesn't block the event loop.
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(enforce_rate_limit)])
def query(request: QueryRequest):
    from app.rag.retriever import generate_answer, retrieve_contexts

    try:
        answer, results = generate_answer(request.query, k=request.k)
    except LLMRefused as e:
        logger.warning(f"Model refused query: {e}")
        answer, results = REFUSAL_ANSWER, retrieve_contexts(request.query, k=request.k)
    return QueryResponse(answer=answer, sources=_docs_to_sources(results))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/status")
def status():
    from app.ingestion.pipeline import get_vector_store

    store = get_vector_store()
    count = store._chroma_collection.count()
    return {"status": "ok", "documents_indexed": count}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)
