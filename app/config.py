"""Load config.yaml, then overlay environment variables (see .env.example)."""

import os
from pathlib import Path
from typing import Callable

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# Environment variable -> (dotted config path, caster). ANTHROPIC_API_KEY is read by the SDK itself.
ENV_OVERRIDES: dict[str, tuple[str, Callable[[str], object]]] = {
    "LLM_PROVIDER": ("llm.provider", str),
    "LLM_MODEL": ("llm.model", str),
    "LLM_EFFORT": ("llm.effort", str),
    "OLLAMA_BASE_URL": ("ollama.base_url", str),
    "CHROMA_PERSIST_DIR": ("chroma.persist_directory", str),
    "CORS_ORIGINS": ("server.cors_origins", _csv),
    "RATE_LIMIT_PER_MINUTE": ("server.rate_limit_per_minute", int),
    "PORT": ("server.port", int),
}


def _set_path(config: dict, dotted: str, value: object) -> None:
    node = config
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def apply_env_overrides(config: dict, env: dict[str, str] | None = None) -> dict:
    env = os.environ if env is None else env
    for var, (dotted, cast) in ENV_OVERRIDES.items():
        raw = env.get(var)
        if raw:
            _set_path(config, dotted, cast(raw))
    return config


def load_config(path: str | None = None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return apply_env_overrides(config)
