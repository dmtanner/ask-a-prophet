"""Answer-model providers. Pick one with llm.provider in config.yaml or LLM_PROVIDER."""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "ollama": "qwen3.6:latest",
    "anthropic": "claude-opus-5",
}


class LLMRefused(Exception):
    """The model declined to answer (Anthropic safety classifier)."""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str):
        from langchain_ollama import ChatOllama

        self.model = model
        self._chat = ChatOllama(model=model, base_url=base_url, temperature=0.2)

    def complete(self, system: str, user: str) -> str:
        response = self._chat.invoke([("system", system), ("human", user)])
        return response.content


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 16000, effort: str = "medium", client=None):
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        # fallbacks="default" reroutes a safety refusal to another model inside the same call.
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMRefused(getattr(response.stop_details, "explanation", None) or "refused")
        return "".join(block.text for block in response.content if block.type == "text")


def get_llm(config: dict | None = None) -> LLM:
    if config is None:
        from app.config import load_config

        config = load_config()

    llm_cfg = config.get("llm", {}) or {}
    ollama_cfg = config.get("ollama", {}) or {}
    provider = (llm_cfg.get("provider") or "ollama").lower()

    if provider == "ollama":
        model = llm_cfg.get("model") or ollama_cfg.get("llm_model") or DEFAULT_MODELS["ollama"]
        return OllamaLLM(model, ollama_cfg.get("base_url", "http://localhost:11434"))
    if provider == "anthropic":
        model = llm_cfg.get("model") or DEFAULT_MODELS["anthropic"]
        return AnthropicLLM(
            model,
            max_tokens=int(llm_cfg.get("max_tokens", 16000)),
            effort=llm_cfg.get("effort", "medium"),
        )
    raise ValueError(f"Unknown llm.provider {provider!r}; expected 'ollama' or 'anthropic'")
