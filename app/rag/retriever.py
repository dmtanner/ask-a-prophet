"""RAG retrieval and prompt templating for prophet persona answers."""

import logging
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


PROPHET_SYSTEM_TEMPLATE = """\
You are relaying teachings and counsel of the prophets. Respond as if speaking in a \
unified "prophet persona" — warm, inspiring, scripture-grounded counsel. Your answers should:

1. Blend the voice of many inspired speakers (modern prophets, apostles, and leaders) into one cohesive message

2. Always support your answer with DIRECT QUOTES taken verbatim from the context provided \
(include speaker name, talk title, and date for each quote). Never invent a quote or a citation; \
if the context does not cover the question, say so plainly.

3. Be encouraging and practical — focus on faith, family, personal worth, discipleship, and covenant blessings

4. When appropriate, reference both modern conference counsel and historical teachings from the \
Journal of Discourses when they appear in the context

5. Never speak with personal opinion as if it were doctrine — only relay or paraphrase what is written

Respond in about 3-5 paragraphs of counsel, with embedded quotes where relevant. \
Keep the tone warm, pastoral, and uplifting."""


ANSWER_USER_TEMPLATE = """\
Given the following context from talks and writings:

---

{context}

---

Provide counsel to this person: {query}"""


@dataclass
class SearchResult:
    content: str
    source_title: str
    speaker: str | None = None
    date: str | None = None
    source_type: str | None = None


def retrieve_contexts(query, k=5):
    """Query the vector store and return formatted results for RAG."""
    from app.ingestion.pipeline import get_vector_store

    store = get_vector_store()
    docs = store.similarity_search(query, k=k)

    results = []
    for doc in docs:
        m = doc.metadata
        source_title = m.get("title", m.get("source", "Unknown"))
        speaker = m.get("speaker") or m.get("sermon_number")
        date_val = m.get("date") or ""
        results.append(SearchResult(
            content=doc.page_content,
            source_title=source_title,
            speaker=speaker,
            date=date_val,
            source_type=m.get("source_type"),
        ))

    return results


def get_llm():
    from app.config import load_config
    config = load_config()
    ollama_cfg = config.get("ollama", {})
    return ChatOllama(
        model=ollama_cfg.get("llm_model", "qwen3.6:latest"),
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        temperature=0.2,
    )


def format_context(results: list[SearchResult]) -> str:
    parts = []
    for r in results:
        citation = f"[{r.speaker}, {r.date}] — {r.source_title}" if r.speaker else f"[Unknown] — {r.source_title}"
        parts.append(f"== {citation} ==\n{r.content.strip()}\n")
    return "\n\n".join(parts)


def generate_answer(query, k=5):
    """Retrieve context and generate a prophet-persona answer."""
    results = retrieve_contexts(query, k=k)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROPHET_SYSTEM_TEMPLATE),
        ("human", ANSWER_USER_TEMPLATE),
    ])
    chain = prompt | get_llm()
    response = chain.invoke({"context": format_context(results), "query": query})

    return response.content, results


if __name__ == "__main__":
    import typer

    logging.basicConfig(level=logging.INFO)

    @typer.command(name="ask")
    def main(query: str):
        answer, sources = generate_answer(query, k=5)

        print("\n" + "=" * 60)
        print(f"User question: {query}")
        print("=" * 60)

        print("\nAnswer:\n")
        print(answer)

        print("\n--- Sources ---\n")
        for i, s in enumerate(sources, 1):
            speaker = f"{s.speaker}, " if s.speaker else ""
            date_str = f" ({s.date})" if s.date else ""
            print(f"{i}. {speaker}{s.source_title}{date_str}")

    typer.Typer()(main)
