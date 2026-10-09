from fastapi.testclient import TestClient

from app.api import server
from app.api.server import RateLimiter
from app.rag.llm import LLMRefused
from app.rag.retriever import SearchResult


def test_rate_limiter_window():
    rl = RateLimiter(2)
    assert rl.allow("a") and rl.allow("a")
    assert not rl.allow("a")
    assert rl.allow("b")
    assert RateLimiter(0).allow("anyone")


def test_query_returns_answer_and_sources(monkeypatch):
    result = SearchResult(content="quote", source_title="T", speaker="S", date="April 2000", source_type="conference")
    monkeypatch.setattr("app.rag.retriever.generate_answer", lambda q, k=5: ("answer", [result]))
    monkeypatch.setattr(server, "limiter", RateLimiter(0))

    resp = TestClient(server.app).post("/query", json={"query": "hi", "k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer"
    assert body["sources"][0] == {"quote": "quote", "speaker": "S", "title": "T", "source": "conference", "date": "April 2000"}


def test_query_refusal_returns_gentle_answer(monkeypatch):
    def refuse(q, k=5):
        raise LLMRefused("no")

    monkeypatch.setattr("app.rag.retriever.generate_answer", refuse)
    monkeypatch.setattr("app.rag.retriever.retrieve_contexts", lambda q, k=5: [])
    monkeypatch.setattr(server, "limiter", RateLimiter(0))

    body = TestClient(server.app).post("/query", json={"query": "hi"}).json()
    assert body["answer"] == server.REFUSAL_ANSWER and body["sources"] == []


def test_query_rate_limited(monkeypatch):
    monkeypatch.setattr("app.rag.retriever.generate_answer", lambda q, k=5: ("a", []))
    monkeypatch.setattr(server, "limiter", RateLimiter(1))
    client = TestClient(server.app)
    assert client.post("/query", json={"query": "hi"}).status_code == 200
    assert client.post("/query", json={"query": "hi"}).status_code == 429


def test_healthz():
    assert TestClient(server.app).get("/healthz").json() == {"status": "ok"}
