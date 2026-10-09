"""CLI entry points for Prophet RAG."""

import logging
from typing import Annotated

import typer

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per embed call otherwise
logger = logging.getLogger(__name__)

typer_app = typer.Typer(help="Prophet RAG — ingest and query LDS conference talks")


def _ingest() -> None:
    """Run the full ingestion pipeline."""
    from app.ingestion.pipeline import ingest_all

    count = ingest_all()
    print(f"Indexed {count} document chunks into ChromaDB")


typer_app.command(name="ingest")(_ingest)


def ingest() -> None:
    """Pip console script entry for `ingest`. Parses flags via Typer so `--help` doesn't start a run."""
    typer.run(_ingest)


def _serve(
    host: Annotated[str, typer.Option("--host", "-H")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    reload: Annotated[bool, typer.Option("--reload/--no-reload")] = True,
) -> None:
    """Start the FastAPI server."""
    from app.config import load_config

    cfg = load_config().get("server", {})
    bind_host = host if host != "0.0.0.0" else cfg.get("host", "0.0.0.0")
    bind_port = port if port != 8000 else cfg.get("port", 8000)

    import uvicorn

    uvicorn.run(
        "app.api.server:app",
        host=bind_host,
        port=int(bind_port),
        reload=reload,
    )


typer_app.command(name="serve")(_serve)


def serve() -> None:
    """Pip console script entry for `serve`. Parses --host/--port via Typer."""
    typer.run(_serve)


if __name__ == "__main__":
    typer_app()
