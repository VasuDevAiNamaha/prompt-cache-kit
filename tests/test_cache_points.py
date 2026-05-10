from prompt_cache_kit import (
    CacheDirective,
    PromptCachingStrategy,
    apply_cache_points,
    cache_at,
    cache_until,
    create_bedrock_cache_point,
    plan_cache_points,
    rolling_cache,
    suggest_cache_points,
)


def count_words(text):
    return len(str(text).split())


def test_suggest_cache_points_stops_before_dynamic_user_message():
    messages = [
        {"role": "system", "content": "one two three", "stable": True},
        {"role": "user", "content": "four five", "stable": False},
        {"role": "user", "content": "six seven", "stable": True},
    ]

    suggestions = suggest_cache_points(messages, min_tokens=3, token_counter=count_words)

    assert len(suggestions) == 1
    assert suggestions[0].message_index == 0
    assert suggestions[0].cumulative_tokens == 3


def test_apply_anthropic_cache_points_marks_last_text_block():
    messages = [{"role": "system", "content": "one two three", "stable": True}]
    suggestions = suggest_cache_points(messages, provider="anthropic", min_tokens=3, token_counter=count_words)

    marked = apply_cache_points(messages, suggestions, provider="anthropic")

    assert marked[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_apply_bedrock_cache_points_appends_cache_point_block():
    messages = [{"role": "system", "content": "one two three", "stable": True}]
    suggestions = suggest_cache_points(messages, provider="bedrock", min_tokens=3, token_counter=count_words)

    marked = apply_cache_points(messages, suggestions, provider="bedrock")

    assert marked[0]["content"][-1] == create_bedrock_cache_point()


def test_apply_generic_cache_points_adds_metadata():
    messages = [{"role": "system", "content": "one two three", "stable": True}]

    marked = apply_cache_points(messages, provider="openai", min_tokens=3, token_counter=count_words)

    assert marked[0]["cache_point"]["id"] == "openai-prefix-0"


def test_manual_cache_until_overrides_dynamic_default():
    messages = [
        {"role": "system", "content": "one two"},
        {"role": "user", "content": "three four"},
        {"role": "user", "content": "dynamic question"},
    ]

    suggestions = plan_cache_points(messages, cache_until(1, id="intro-turn"), provider="openai", token_counter=count_words)

    assert len(suggestions) == 1
    assert suggestions[0].message_index == 1
    assert suggestions[0].cache_point.id == "intro-turn"


def test_manual_cache_at_multiple_points_with_exclusion():
    messages = [
        {"role": "system", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "three"},
    ]
    plan = cache_at(0, 1, 2, ids={0: "system", 2: "assistant"}).excluding(1)

    suggestions = plan_cache_points(messages, plan, provider="openai", token_counter=count_words)

    assert [item.message_index for item in suggestions] == [0, 2]
    assert [item.cache_point.id for item in suggestions] == ["system", "assistant"]


def test_rolling_cache_keeps_old_stable_turns():
    messages = [
        {"role": "system", "content": "one two", "stable": True},
        {"role": "user", "content": "three four", "stable": True},
        {"role": "assistant", "content": "five six", "stable": True},
        {"role": "user", "content": "seven eight", "stable": True},
        {"role": "user", "content": "current question", "stable": False},
    ]

    suggestions = plan_cache_points(
        messages,
        rolling_cache(every_messages=2, min_tokens=2, max_points=4),
        provider="openai",
        token_counter=count_words,
    )

    assert [item.message_index for item in suggestions] == [0, 2, 3]


def test_apply_cache_points_accepts_plan_directly():
    messages = [
        {"role": "system", "content": "one two"},
        {"role": "user", "content": "three four"},
    ]

    marked = apply_cache_points(messages, plan=cache_until(0, id="system-v1"), provider="openai", token_counter=count_words)

    assert marked[0]["cache_point"]["id"] == "system-v1"
    assert "cache_point" not in marked[1]


def test_custom_prompt_caching_strategy_can_be_extended():
    class LastAssistantStrategy(PromptCachingStrategy):
        def select(self, context):
            for index in range(len(context.messages) - 1, -1, -1):
                if context.messages[index]["role"] == "assistant":
                    return [CacheDirective(message_index=index, id="last-assistant", reason="custom")]
            return []

    messages = [
        {"role": "system", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    suggestions = plan_cache_points(messages, LastAssistantStrategy(), provider="openai", token_counter=count_words)

    assert suggestions[0].message_index == 1
    assert suggestions[0].cache_point.id == "last-assistant"
