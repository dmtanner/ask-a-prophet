FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml config.yaml ./
COPY app ./app
COPY web ./web
RUN pip install --no-cache-dir .

ENV CHROMA_PERSIST_DIR=/data/chroma \
    OLLAMA_BASE_URL=http://ollama:11434 \
    PORT=8000
VOLUME ["/data/chroma"]
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.api.server:app --host 0.0.0.0 --port ${PORT}"]
