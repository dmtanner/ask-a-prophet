"""Load YAML config from config.yaml."""

import yaml
from pathlib import Path


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
