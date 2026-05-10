"""Cache a custom LLM inference function.

Run:
    python examples/custom_inference.py
"""

from prompt_cache_kit import CachePolicy, MemoryCacheBackend, cache_until, apply_cache_points, cached


backend = MemoryCacheBackend()
policy = CachePolicy(namespace="custom-demo", ttl_seconds=300)


@cached(backend=backend, policy=policy)
def local_model_generate(messages, *, model="local-dev-model", temperature=0):
    print("Calling real model...")
    return {
        "model": model,
        "content": "Pretend this came from your in-house inference service.",
        "temperature": temperature,
    }


raw_messages = [
    {"role": "system", "content": "Long stable policy text.", "stable": True},
    {"role": "user", "content": "Summarize the policy.", "stable": False},
]

messages = apply_cache_points(raw_messages, plan=cache_until(0, id="policy-v1"), provider="generic")

print(local_model_generate(messages))
print(local_model_generate(messages))
print(backend.stats())
