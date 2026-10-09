# Ask a Prophet

Local RAG app that answers questions using teachings from [General Conference](https://www.churchofjesuschrist.org/study/general-conference) talks and the [Journal of Discourses](https://archive.org/details/journalofdiscourses), with quotes and citations.

It retrieves relevant passages from a local ChromaDB index, then asks an LLM — a local Ollama model or Claude via API key — to answer in a unified “prophet persona” — warm, scripture-grounded counsel, without inventing quotes.

This is an unofficial personal project. It is not affiliated with, endorsed by, or a product of The Church of Jesus Christ of Latter-day Saints. Conference talks remain copyrighted by the Church; this repo does not redistribute that text. Ingest fetches publicly available pages onto your machine for local search only — do not republish the corpus.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- These models pulled (names must match `config.yaml`):

```bash
ollama pull qwen3.6:latest
ollama pull nomic-embed-text
```

Confirm Ollama is up at `http://localhost:11434`.

## Install

```bash
git clone https://github.com/dmtanner/ask-a-prophet.git
cd ask-a-prophet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Ingest the corpus

This fetches talks, chunks them, embeds with `nomic-embed-text`, and stores vectors in `./chroma_persistent_db`.

```bash
ingest
```

A full run can take a while (thousands of talks plus Journal of Discourses volumes). Progress is checkpointed in `.ingest_checkpoint.json`, so you can stop and re-run without starting over.

To try a smaller conference sample first, set `ingestion.sources.general_conference.max_ids` in `config.yaml` (for example `200`). `-1` means ingest everything.

You can also skip a source by setting `enabled: false` under `general_conference` or `journal_of_discourses`.

## Run the app

```bash
serve
```

Then open [http://localhost:8000](http://localhost:8000). Type a question and click **Ask**. Use **Show Sources** to see the retrieved quotes and citations.

The server host/port come from `config.yaml` (`0.0.0.0:8000` by default). Override with:

```bash
serve --host 127.0.0.1 --port 8000
```

### API

**Ask a question**

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How can I strengthen my family?", "k": 5}'
```

**Index status**

```bash
curl -s http://localhost:8000/status
```

## Configuration

Edit `config.yaml` for:

| Setting | Purpose |
|---|---|
| `llm.provider` | `ollama` (local) or `anthropic` (hosted, needs `ANTHROPIC_API_KEY`) |
| `llm.model` / `llm.effort` | Answer model and, for Anthropic, reasoning effort |
| `ollama.llm_model` / `embed_model` | Local answer model and the embedding model |
| `ollama.base_url` | Ollama endpoint (embeddings always run here) |
| `chroma.persist_directory` | Where the vector store lives |
| `ingestion.sources` | Which corpora to fetch |
| `splitting.chunk_size` / `chunk_overlap` | Text chunking |
| `server.host` / `port` | HTTP bind address |
| `server.cors_origins` | Allowed browser origins; set to your site in production |
| `server.rate_limit_per_minute` | Per-IP cap on `/query` (0 disables) |

Environment variables override the YAML, which is how a deployment differs from a laptop without editing files. Copy `.env.example` to `.env` (gitignored) and set what you need. `ANTHROPIC_API_KEY` is read directly by the SDK and never belongs in `config.yaml`.

### Using Claude instead of a local model

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
serve
```

Ollama is still required for query embeddings (`nomic-embed-text`), since the index was built with it. The large local answer model is not needed in this mode.

The vector store and ingest logs stay on disk and are gitignored. After cloning, you need to run `ingest` (or copy an existing `chroma_persistent_db`) before queries will return useful answers.

## Deploy with Docker

The compose file runs the app plus a small Ollama container that only serves embeddings.

```bash
cp .env.example .env            # set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY
docker compose up -d --build
docker compose exec ollama ollama pull nomic-embed-text
docker compose run --rm app ingest   # or copy an existing chroma_persistent_db into the chroma volume
```

The app listens on `$PORT` (default 8000) and exposes `/healthz` for platform health checks. The vector store lives in the `chroma` volume, so rebuilding the image does not lose it. Any host that runs a container with a persistent volume (Fly.io, Railway, Render, a VPS) works.

Before exposing it publicly, set `CORS_ORIGINS` to your domain, keep `RATE_LIMIT_PER_MINUTE` on, and put a spending limit on the API key in the Anthropic console — every query costs money.

## How it works

1. **Ingest** — loaders pull General Conference HTML and Journal of Discourses text, split into overlapping chunks, embed them, and upsert into ChromaDB.
2. **Retrieve** — a question is embedded and the top-k similar chunks are returned.
3. **Generate** — the LLM answers from those chunks only, citing speaker, title, and date.

With the default config all inference stays on your machine via Ollama. Source text comes from publicly available conference talks and historical writings.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
