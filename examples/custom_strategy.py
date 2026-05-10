"""Create a custom PromptCachingStrategy."""

from prompt_cache_kit import CacheDirective, PromptCachingStrategy, apply_cache_points, plan_cache_points


class FirstLargeUserMessageStrategy(PromptCachingStrategy):
    def __init__(self, min_tokens=1000):
        self.min_tokens = min_tokens

    def select(self, context):
        for index, message in enumerate(context.messages):
            if message.get("role") in {"user", "human"} and context.message_tokens(index) >= self.min_tokens:
                return [
                    CacheDirective(
                        message_index=index,
                        id=f"large-user-message-{index}",
                        reason="custom-first-large-user-message",
                    )
                ]
        return []


messages = [
    {"role": "system", "content": "Stable system prompt."},
    {"role": "user", "content": "Long uploaded document text " * 100},
    {"role": "user", "content": "Question"},
]

strategy = FirstLargeUserMessageStrategy(min_tokens=50)
print(plan_cache_points(messages, strategy, provider="anthropic"))
print(apply_cache_points(messages, plan=strategy, provider="anthropic"))
