"""Pipeline: fetch documents, chunk them, embed, and index into ChromaDB."""

import json
import logging
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import load_config
from app.ingestion.conference import GeneralConferenceLoader
from app.ingestion.journal_of_discourses import JournalOfDiscoursesLoader

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = ".ingest_checkpoint.json"


def get_embedding_model():
    config = load_config()
    ollama_cfg = config.get("ollama", {})
    return OllamaEmbeddings(
        model=ollama_cfg.get("embed_model", "nomic-embed-text"),
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
    )


def get_vector_store():
    chroma_cfg = load_config().get("chroma", {})
    persist_dir = Path(chroma_cfg.get("persist_directory", "./chroma_persistent_db"))
    return Chroma(
        collection_name=chroma_cfg.get("collection_name", "prophet_quotes"),
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_dir),
    )


def _load_checkpoint() -> dict:
    project_root = Path(__file__).resolve().parents[2] / CHECKPOINT_FILE
    if project_root.exists():
        with open(project_root) as f:
            return json.load(f)
    return {}


def _save_checkpoint(state: dict):
    """Merge `state` into the checkpoint file so one source's progress never clobbers another's."""
    project_root = Path(__file__).resolve().parents[2] / CHECKPOINT_FILE
    merged = _load_checkpoint()
    for source, values in state.items():
        merged[source] = {**merged.get(source, {}), **values}
    with open(project_root, "w") as f:
        json.dump(merged, f, indent=2)


def _resume_plan(checkpoint: dict, gc_enabled: bool, jod_enabled: bool) -> tuple[dict, dict]:
    """Decide skip/resume for each source.

    If every *enabled* source already finished, start a fresh pass (chunk IDs upsert).
    Otherwise skip sources marked complete and resume the rest.
    """
    gc = checkpoint.get("general_conference", {})
    jod = checkpoint.get("journal_of_discourses", {})
    enabled_complete = []
    if gc_enabled:
        enabled_complete.append(bool(gc.get("complete")))
    if jod_enabled:
        enabled_complete.append(bool(jod.get("complete")))
    fresh = bool(enabled_complete) and all(enabled_complete)

    if fresh:
        logger.info("Previous ingest finished; starting a fresh pass")
        return (
            {"skip": False, "resume_from_id": 0},
            {"skip": False, "resume_from_id": 0, "completed_volumes": set()},
        )
    return (
        {
            "skip": bool(gc.get("complete")),
            "resume_from_id": gc.get("resume_from_id", 0),
        },
        {
            "skip": bool(jod.get("complete")),
            "resume_from_id": jod.get("resume_from_id", 0),
            "completed_volumes": set(jod.get("completed_volumes", [])),
        },
    )


def _save_source_progress(source_type: str, last_item_id: int, done_volumes: set, complete: bool):
    if source_type == "conference":
        _save_checkpoint({
            "general_conference": {
                "resume_from_id": last_item_id,
                "complete": complete,
            }
        })
    else:
        _save_checkpoint({
            "journal_of_discourses": {
                "resume_from_id": last_item_id,
                "completed_volumes": sorted(done_volumes),
                "complete": complete,
            }
        })


def ingest_all():
    """Full ingestion pipeline with checkpoint/resume support.

    Returns count of document chunks indexed.
    """
    config = load_config()
    persist_dir = Path(config.get("chroma", {}).get("persist_directory", "./chroma_persistent_db"))
    persist_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint()
    logger.info(f"Checkpoint state: {checkpoint}")

    sources_cfg = config.get("ingestion", {}).get("sources", {})
    gc_enabled = sources_cfg.get("general_conference", {}).get("enabled", False)
    jod_enabled = sources_cfg.get("journal_of_discourses", {}).get("enabled", False)

    gc_state, jod_state = _resume_plan(checkpoint, gc_enabled, jod_enabled)

    split_cfg = config.get("splitting", {})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_cfg.get("chunk_size", 800),
        chunk_overlap=split_cfg.get("chunk_overlap", 100),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    embeddings = get_embedding_model()
    chroma_cfg = config.get("chroma", {})
    collection_name = chroma_cfg.get("collection_name", "prophet_quotes")

    loaders_to_run = []

    if gc_enabled and not gc_state["skip"]:
        loaders_to_run.append({
            "name": "General Conference",
            "loader": GeneralConferenceLoader(),
            "resume_id": gc_state["resume_from_id"],
            "source_type": "conference",
        })
    elif gc_enabled:
        logger.info("Skipping General Conference (already complete in this pass)")

    if jod_enabled and not jod_state["skip"]:
        loaders_to_run.append({
            "name": "Journal of Discourses",
            "loader": JournalOfDiscoursesLoader(),
            "resume_id": jod_state["resume_from_id"],
            "completed_volumes": jod_state["completed_volumes"],
            "source_type": "journal",
        })
    elif jod_enabled:
        logger.info("Skipping Journal of Discourses (already complete in this pass)")

    if not loaders_to_run:
        logger.warning("No sources enabled in config")
        return 0

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    total_chunks = 0

    for loader_info in loaders_to_run:
        name = loader_info["name"]
        loader = loader_info["loader"]
        source_type = loader_info["source_type"]
        resume_from = loader_info["resume_id"]
        completed_volumes = loader_info.get("completed_volumes", set())

        logger.info(f"Fetching {name}...")

        if source_type == "conference":
            items_generator = loader.load(resume_from=resume_from)
        else:
            items_generator = loader.load(
                resume_from=resume_from,
                skip_completed=completed_volumes,
            )

        yield_count = 0
        last_item_id = resume_from if isinstance(resume_from, int) else 0
        current_vol = None
        done_volumes = set(completed_volumes)

        BATCH_SIZE = 50
        batch_raw = []

        for item in items_generator:
            batch_raw.append(item)
            yield_count += 1

            m = item.get("metadata", {})
            item_id = m.get("id")
            if item_id is not None:
                last_item_id = item_id
            vol_num = m.get("volume") if source_type == "journal" else None
            if vol_num is not None:
                if current_vol is not None and vol_num != current_vol:
                    done_volumes.add(current_vol)
                current_vol = vol_num

            if len(batch_raw) >= BATCH_SIZE:
                chunked = _chunk_and_store(
                    batch_raw, source_type, splitter, vector_store,
                    collection_name, name, yield_count
                )
                total_chunks += chunked
                del batch_raw, chunked
                batch_raw = []
                _save_source_progress(
                    source_type, last_item_id, done_volumes, complete=False
                )

        if batch_raw:
            chunked = _chunk_and_store(
                batch_raw, source_type, splitter, vector_store,
                collection_name, name, yield_count
            )
            total_chunks += chunked
            del batch_raw, chunked

        if current_vol is not None:
            done_volumes.add(current_vol)
        _save_source_progress(source_type, last_item_id, done_volumes, complete=True)

        logger.info(f"Completed {name}: {yield_count} items yielded")

    logger.info(f"Ingestion complete — {total_chunks} total chunks chunked into ChromaDB")
    return total_chunks


def _chunk_and_store(batch, source_type, splitter, vector_store, collection_name, label, yield_count):
    """Chunk a batch of raw items and upsert into the vector store with per-talk chunk IDs."""
    from langchain_core.documents import Document as LCDoc

    prefix = "gc" if source_type == "conference" else "jord"
    doc_list, id_list = [], []
    for item in batch:
        meta = {k: v for k, v in item["metadata"].items() if v is not None}
        meta["source_type"] = source_type
        item_id = meta["id"]
        for n, chunk in enumerate(splitter.split_text(item["content"])):
            chunk_id = f"{prefix}_{item_id}_{n}"
            doc_list.append(LCDoc(page_content=chunk, metadata={**meta, "chunk_index": chunk_id}))
            id_list.append(chunk_id)

    total_add = 0
    BATCH = 100
    for i in range(0, len(doc_list), BATCH):
        try:
            vector_store.add_documents(doc_list[i:i + BATCH], ids=id_list[i:i + BATCH])
            total_add += len(doc_list[i:i + BATCH])
        except Exception as e:
            logger.error(f"Error adding batch from {label}: {e}")

    logger.info(f"  {label}/{yield_count}: {len(batch)} items -> {len(doc_list)} chunks, {total_add} added")
    return total_add


if __name__ == "__main__":
    import typer

    logging.basicConfig(level=logging.INFO)

    @typer.command()
    def main():
        count = ingest_all()
        print(f"Indexed {count} document chunks into ChromaDB")

    typer.Typer()(main)
