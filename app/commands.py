"""CLI entry points for Prophet RAG."""

import logging
import sys

import typer

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per embed call otherwise
logger = logging.getLogger(__name__)

typer_app = typer.Typer(help="Prophet RAG — ingest and query LDS conference talks")


@typer_app.command()
def ingest():
    """Run the full ingestion pipeline."""
    from app.ingestion.pipeline import ingest_all

    count = ingest_all()
    print(f"Indexed {count} document chunks into ChromaDB")


@typer_app.command(name="serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = True,
):
    """Start the FastAPI server."""
    from app.config import load_config
    
    cfg = load_config().get("server", {})
    final_host = host if host != "0.0.0.0" else cfg.get("host", "0.0.0.0")
    final_port = port if port != 8000 else cfg.get("port", 8000)
    final_reload = reload
    
    import uvicorn
    uvicorn.run(
        "app.api.server:app",
        host=final_host,
        port=final_port,
        reload=final_reload,
    )


if __name__ == "__main__":
    typer_app()
