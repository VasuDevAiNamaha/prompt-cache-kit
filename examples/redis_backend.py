"""Use Redis as a response-cache backend.

Prerequisites:
    pip install "prompt-cache-kit[redis]"
    docker run --rm -p 6379:6379 redis:7

Run:
    python examples/redis_backend.py
"""

from prompt_cache_kit import CachePolicy, RedisCacheBackend, cached


backend = RedisCacheBackend(url="redis://localhost:6379/0", namespace="prompt-cache-kit-example")
policy = CachePolicy(namespace="redis-demo", ttl_seconds=600)


@cached(backend=backend, policy=policy)
def call_llm(prompt):
    print("Cache miss: calling LLM")
    return {"answer": f"response for: {prompt}"}


print(call_llm("What is prefix caching?"))
print(call_llm("What is prefix caching?"))
print(backend.stats())
