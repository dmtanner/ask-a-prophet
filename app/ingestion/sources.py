"""Source loaders for document ingestion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    enabled: bool = True
    source_type: str = ""
    metadata: dict = field(default_factory=dict)


@abstractmethod
class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[dict]:
        """Return list of dicts with keys: content (str), source (str), metadata (dict)."""
        ...
