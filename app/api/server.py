"""FastAPI server for Prophet RAG queries."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Prophet RAG API starting up")
    yield
    logger.info("Prophet RAG API shutting down")


app = FastAPI(
    title="Prophet RAG",
    description="Answer queries using writings of LDS modern prophets with full citations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI
web_dir = Path(__file__).resolve().parents[2] / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

from fastapi.responses import FileResponse


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


def _docs_to_sources(docs) -> list[QuoteSource]:
    out = []
    for doc in docs:
        if hasattr(doc, 'content'):  # SearchResult
            content = doc.content
            title = doc.source_title
            speaker = str(doc.speaker) if doc.speaker else ""
            date_val = str(doc.date) if doc.date else ""
            source_name = doc.source_type or "Unknown"
        else:  # LangChain Document
            m = doc.metadata
            content = doc.page_content
            title = m.get("title", "Unknown")
            speaker = str(m.get("speaker") or m.get("sermon_number", ""))
            source_name = m.get("source_type", "Unknown")
            date_val = str(m.get("date")) if m.get("date") else ""
        out.append(QuoteSource(
            quote=content,
            speaker=speaker or None,
            title=str(title),
            source=source_name,
            date=date_val,
        ))
    return out


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    from app.rag.retriever import retrieve_contexts, generate_answer

    answer, retrieved_docs = generate_answer(request.query, k=request.k)
    sources = _docs_to_sources(retrieved_docs)
    
    return QueryResponse(answer=answer, sources=sources)


@app.get("/status")
async def status():
    from app.ingestion.pipeline import get_vector_store
    store = get_vector_store()
    count = store._chroma_collection.count()
    return {"status": "ok", "documents_indexed": count}


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)
