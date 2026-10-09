from types import SimpleNamespace

import pytest

from app.rag.llm import AnthropicLLM, LLMRefused, OllamaLLM, get_llm


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_anthropic_complete_sends_system_and_user_and_joins_text():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", text=None), SimpleNamespace(type="text", text="Hello "), SimpleNamespace(type="text", text="friend")],
    )
    client, messages = fake_client(response)
    llm = AnthropicLLM("claude-opus-5", max_tokens=1234, effort="low", client=client)

    assert llm.complete("SYS", "USER") == "Hello friend"
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "USER"}]
    assert call["max_tokens"] == 1234
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"


def test_anthropic_refusal_raises():
    response = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(explanation="nope"), content=[])
    client, _ = fake_client(response)
    with pytest.raises(LLMRefused):
        AnthropicLLM("claude-opus-5", client=client).complete("SYS", "USER")


def test_get_llm_picks_provider_and_defaults():
    anth = get_llm({"llm": {"provider": "anthropic"}})
    assert isinstance(anth, AnthropicLLM) and anth.model == "claude-opus-5"

    oll = get_llm({"llm": {"provider": "ollama"}, "ollama": {"llm_model": "qwen3.6:latest"}})
    assert isinstance(oll, OllamaLLM) and oll.model == "qwen3.6:latest"

    legacy = get_llm({"ollama": {"llm_model": "gemma4:latest"}})
    assert isinstance(legacy, OllamaLLM) and legacy.model == "gemma4:latest"

    with pytest.raises(ValueError):
        get_llm({"llm": {"provider": "openai"}})
