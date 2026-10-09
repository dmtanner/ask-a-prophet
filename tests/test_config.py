from app.config import apply_env_overrides


def test_env_overrides_set_nested_values():
    config = {"llm": {"provider": "ollama"}, "server": {"cors_origins": ["*"]}}
    env = {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-opus-5",
        "CORS_ORIGINS": "https://a.example, https://b.example",
        "RATE_LIMIT_PER_MINUTE": "3",
        "CHROMA_PERSIST_DIR": "/data/chroma",
    }
    apply_env_overrides(config, env)
    assert config["llm"] == {"provider": "anthropic", "model": "claude-opus-5"}
    assert config["server"]["cors_origins"] == ["https://a.example", "https://b.example"]
    assert config["server"]["rate_limit_per_minute"] == 3
    assert config["chroma"]["persist_directory"] == "/data/chroma"


def test_empty_env_values_are_ignored():
    config = {"llm": {"provider": "ollama"}}
    apply_env_overrides(config, {"LLM_PROVIDER": ""})
    assert config["llm"]["provider"] == "ollama"
