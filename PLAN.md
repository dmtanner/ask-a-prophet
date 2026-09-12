# Finish "Ask a Prophet" RAG — Implementation Plan

## Current State

### What works
- BYU talks endpoint parses valid HTML (title, speaker, date, body via class-based selectors)
- Short talks (President's remarks without `gcera`/`gcbody`) are already skipped (<300 bytes filter)
- Politeness delay every 25 requests is in place
- Journal of Discourses volume URLs and sermon regex patterns look structurally correct
- Pipeline chunking (RecursiveCharacterTextSplitter, 800/100), embedding (Ollama nomic-embed-text), and ChromaDB batch indexing all functional
- FastAPI server (`/query`, `/status`, UI) is wired up

### What doesn't work
1. **Journal of Discourses** — `NameError: name 'requests' is not defined` (missing import on line 33)
2. **Loading strategy** — pipeline calls `loader.load()` which returns all ~8,000+ talks in one shot → kills the process before any chunking/embedding happens → total loss of all fetched data
3. **No checkpoint/resume** — a crashed ingest loses everything and must start from ID 1
4. **No `app/commands.py`** — `pyproject.toml` declares `ingest` and `serve` console scripts pointing to it, but the file doesn't exist
5. **Entire vector store is wiped on every ingest** (`delete_collection()`) — not a crash bug but wastes time during repeated runs


---

## Phase 1 — Fix blocking bugs (makes progress again)

### 1.1 Journal of Discourses: add missing `requests` import

- File: `app/ingestion/journal_of_discourses.py`, line 3
- Add `import requests` after the existing imports
- This is a one-line fix

### 1.2 Conference loader: stream results instead of collecting all in memory

- File: `app/ingestion/conference.py`
- Change `load()` from returning `list[dict]` to being a **generator** (`yield` per talk)
- Or add a new method `load_iterator()` that yields individual talks while `load()` calls it internally for backward compat
- This prevents loading all ~8,000 raw HTML parse results into memory simultaneously

### 1.3 Pipeline: incremental checkpointing with JSON state file

- File: `app/ingestion/pipeline.py` (new checkpoint logic)
- Add a `.ingest_checkpoint.json` file in the project root that tracks:
  - `general_conference.last_processed_id` — lowest unprocessed talk ID
  - `journal_of_discourses.volumes_processed` — list of volume numbers already done
- Load this state at start; skip already-processed items
- Write/update incrementally (after every 50 talks or per volume) so a crash only loses one batch

### 1.4 Pipeline: don't wipe existing collection on re-ingest

- File: `app/ingestion/pipeline.py`, around line 86
- Instead of always calling `tmp_store.delete_collection()`, use **idempotent upsert logic**:
  - Track chunk-level IDs using talk ID + chunk index in metadata
  - Delete only existing chunks with the same IDs before re-adding them
  - Or: only wipe if the checkpoint shows zero progress (first run)

### 1.5 Create `app/commands.py`

- File: `app/commands.py` (new file)
- Define two typer CLI commands:
  - `ingest()` — runs `ingest_all()`, logs completion count
  - `serve()` — runs uvicorn with the FastAPI app, host/port from config
- Mirror whatever signatures typos were intended in `pyproject.toml`

---

## Phase 2 — Make ingestion robust

### 2.1 Handle BYU talk content edge cases during chunking

- Some talks have very short bodies (<300 bytes after title extraction) that weren't caught by the HTTP-level filter
- Add a content length check after BeautifulSoup parsing: skip content with fewer than ~80 characters of body text, log a warning with the talk ID so they can be reviewed
- Filter out duplicate/metadata-only paragraphs (e.g., "About", speaker repetition, navigation chrome) more aggressively

### 2.2 Journal of Discourses: per-volume checkpointing

- Extend checkpoint state to track which volumes finished successfully
- If volume 2 fails mid-fetch, resuming skips volume 1 and starts volume 2 again (not from scratch)

### 2.3 Add retry logic for flaky HTTP requests

- Conference loader: retry failed BYU requests with exponential backoff (2–3 retries)
- Journal loader: same pattern — archive.org can timeout on large djvu.txt files

### 2.4 Progress logging during ingestion

- Per-talk or per-batch progress updates to stdout / log
- Estimated time remaining for conference talks (based on current rate × remaining count)
- Summary at the end: successes, skips (with reasons), failures

---

## Phase 3 — Testing and verification

### 3.1 Test with a small subset

- Set `max_ids: 200` in config.yaml, run full pipeline, verify:
  - All 200 talks are fetched or properly skipped (not all failed silently)
  - Chunks are indexed correctly in ChromaDB
  - Query endpoint returns answers with source citations from the ingested data

### 3.2 Test checkpoint/resume

- Set `max_ids: 500`, interrupt the process partway through (kill it manually)
- Verify `.ingest_checkpoint.json` exists and has partial state
- Re-run: confirm it resumes from~the last checkpoint ID, not from 1

### 3.3 Test Journal of Discourses in isolation

- Disable `general_conference` in config.yaml, enable only `journal_of_discourses`
- Run full pipeline, verify all 4 volumes ingest without error
- Query for a term that should match JoD content (e.g., "Second Coming", "Melchizedek")

### 3.4 Full production run

- Enable both sources, set `max_ids: -1` (auto-discover)
- Let it complete end-to-end — this may take 20–40 minutes with delays
- Verify final ChromaDB document count is reasonable (~7,500+ talks + ~700 JoD sermons → chunked to ~30,000–50,000 vectors)
- Test queries that should hit both sources

### 3.5 Query quality regression test

- Ask a known-answer question (e.g., "What did President Nelson say about the Atonement?")
- Verify returned answer includes:
  - Direct quote from a talk
  - Correct source metadata (speaker, date, title)
  - Relevance (not random garbage)

---

## Files to modify / create

| File | Action | Description |
|------|--------|-------------|
| `app/ingestion/journal_of_discourses.py` | **Edit** | Add `import requests` (1 line) |
| `app/ingestion/conference.py` | **Edit** | Convert loader to yield/chunked, add retry logic |
| `app/ingestion/pipeline.py` | **Edit** | Add checkpoint JSON, incremental ingest, stop deletion |
| `app/commands.py` | **Create** | CLI entry points for `ingest` and `serve` |
| `config.yaml` | **Temporary edit** | Temporarily set `max_ids: 200` for Phase 3 testing |
| `.ingest_checkpoint.json` | **Created by pipeline** | Auto-generated checkpoint state file |

---

## Success Criteria

- [ ] `pip install -e .` runs without error; `ingest` and `serve` CLI commands exist

Phase 1 complete: all blocking bugs fixed.
- [ ] Journal of Discourses fetches all 4 volumes successfully
- [ ] Conference loader streams talks, processes with <2GB peak RSS memory
- [ ] Checkpoint file is updated continuously — crash recovery works
- [ ] Post-ingest query at localhost:8000 returns relevant, sourced answers
- [ ] Full re-ingest (both sources) completes without OOM or HTTP failures
