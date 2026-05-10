"""Use Prompt Cache Kit with a CrewAI LLM object.

Prerequisites:
    pip install crewai
"""

from prompt_cache_kit import CachePolicy, MemoryCacheBackend, cache_until, apply_cache_points, wrap_crewai_llm


def build_cached_crewai_llm(llm):
    cached_llm = wrap_crewai_llm(
        llm,
        backend=MemoryCacheBackend(),
        policy=CachePolicy(namespace="crewai-demo", ttl_seconds=300),
    )

    messages = [
        {"role": "system", "content": "Long stable crew policy and role instructions.", "stable": True},
        {"role": "user", "content": "Do the current task.", "stable": False},
    ]
    messages = apply_cache_points(messages, plan=cache_until(0, id="crew-policy-v1"), provider="anthropic")

    # Depending on CrewAI version/provider, LLMs expose call(), invoke(), or __call__().
    return cached_llm.call(messages)


# Example:
# from crewai import LLM
# llm = LLM(model="anthropic/claude-sonnet-4-5")
# result = build_cached_crewai_llm(llm)
