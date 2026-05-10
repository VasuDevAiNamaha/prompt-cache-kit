from prompt_cache_kit import (
    CachePoint,
    PromptLayout,
    analyze_messages,
    extract_langchain_usage,
    normalize_usage,
)


def test_normalize_openai_openrouter_usage():
    stats = normalize_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 10},
            }
        },
        provider="openrouter",
    )

    assert stats.input_tokens == 100
    assert stats.output_tokens == 20
    assert stats.cache_read_input_tokens == 80
    assert stats.cache_write_input_tokens == 10
    assert stats.cache_hit is True
    assert stats.to_openai_usage()["prompt_tokens_details"]["cached_tokens"] == 80
    assert stats.to_otel_attributes()["gen_ai.usage.cache_read.input_tokens"] == 80


def test_normalize_anthropic_usage():
    stats = normalize_usage(
        {
            "input_tokens": 50,
            "output_tokens": 7,
            "cache_creation_input_tokens": 40,
            "cache_read_input_tokens": 20,
        },
        provider="anthropic",
    )

    assert stats.input_tokens == 50
    assert stats.output_tokens == 7
    assert stats.cache_write_input_tokens == 40
    assert stats.cache_read_input_tokens == 20
    assert stats.total_tokens == 57


def test_analyze_messages_with_cache_points():
    stats = analyze_messages(
        [
            {
                "role": "system",
                "content": "stable policy text",
                "stable": True,
                "cache_point": {"id": "policy-v1", "strategy": "prefix"},
            },
            {"role": "user", "content": "question"},
        ]
    )

    assert stats.input_tokens == 4
    assert stats.message_usages[0].cache_point_id == "policy-v1"
    assert stats.cache_points[0].id == "policy-v1"


def test_prompt_layout_usage_includes_cache_point():
    layout = PromptLayout().stable_system("rules").cache_point("rules-v1", provider_hint="anthropic").dynamic_user("ask")

    stats = layout.usage()

    assert stats.message_usages[0].cache_point_id == "rules-v1"
    assert stats.cache_points[0] == CachePoint(id="rules-v1", provider_hint="anthropic")


def test_extract_langchain_usage_metadata():
    class Message:
        usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
        }

    stats = extract_langchain_usage(Message())

    assert stats.input_tokens == 10
    assert stats.output_tokens == 3
    assert stats.cache_read_input_tokens == 4
