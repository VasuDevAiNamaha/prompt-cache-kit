# Prompt Cache Kit

Prompt Cache Kit is a framework-neutral Python package for LLM cache orchestration.
It helps applications decide what should be cacheable, mark provider-specific
prompt cache boundaries, cache final responses safely, and normalize cache/token
usage telemetry.

It does **not** claim to implement universal KV-cache reuse around arbitrary model
objects. True KV caching belongs inside inference engines such as vLLM, SGLang,
TensorRT-LLM, or LMCache. This package gives your application a clean control
plane around those systems.

## What It Provides

- Response caching for Python callables and model-like objects.
- Prompt caching strategies using an extendable Strategy pattern.
- Provider-specific cache-point compilation for Anthropic and Bedrock/LangChain.
- Redis and in-memory response-cache backends.
- LangChain, LangGraph, and CrewAI-friendly wrapping helpers.
- Usage normalization for OpenAI/OpenRouter, Anthropic, and LangChain metadata.
- OpenTelemetry GenAI-style usage attribute export.
- vLLM prefix-caching and LMCache connector config helpers.

## Install

For local development:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "prompt-cache-kit[redis]"
pip install "prompt-cache-kit[langchain]"
```

## Mental Model

Prompt Cache Kit separates four layers:

1. **Response cache**: stores the final result of deterministic or acceptable-to-replay LLM calls.
2. **Prompt cache strategy**: decides which message boundaries should become cache points.
3. **Provider compiler**: renders cache points as Anthropic `cache_control`, Bedrock `cachePoint`, or generic metadata.
4. **Engine helper**: generates vLLM/LMCache configuration and checks LMCache HTTP health/status endpoints.

This split matters because response caching, provider prompt caching, and engine
KV caching have different guarantees.

## Quick Start

```python
from prompt_cache_kit import CachePolicy, MemoryCacheBackend, cached

backend = MemoryCacheBackend()
policy = CachePolicy(namespace="research-agent", ttl_seconds=3600)

@cached(backend=backend, policy=policy)
def call_model(messages, model="gpt-4.1-mini", temperature=0):
    return {"content": "real model response here"}

call_model([{"role": "user", "content": "Explain KV caching"}])
call_model([{"role": "user", "content": "Explain KV caching"}])

print(backend.stats())
```

## Response Cache Backends

### In Memory

```python
from prompt_cache_kit import MemoryCacheBackend

backend = MemoryCacheBackend(max_size=1000)
```

### Redis

```python
from prompt_cache_kit import CachePolicy, RedisCacheBackend, cached

backend = RedisCacheBackend(
    url="redis://localhost:6379/0",
    namespace="my-agent-cache",
)
policy = CachePolicy(namespace="invoice-agent", ttl_seconds=300)

@cached(backend=backend, policy=policy)
def call_llm(messages, model="gpt-4.1-mini"):
    return client.chat.completions.create(model=model, messages=messages)
```

`RedisCacheBackend` stores arbitrary Python responses with pickle. Use a dedicated
Redis namespace/database and avoid sharing it across trust boundaries.

## Wrapping Model Objects

Prompt Cache Kit uses duck typing. It can wrap common model methods without
importing the framework.

```python
from prompt_cache_kit import CachedModel, MemoryCacheBackend

cached_model = CachedModel(
    model=some_model,
    backend=MemoryCacheBackend(),
    methods=("invoke", "ainvoke", "__call__"),
)

result = cached_model.invoke("hello")
```

Convenience wrappers:

```python
from prompt_cache_kit import wrap_crewai_llm, wrap_langchain_model

cached_langchain_model = wrap_langchain_model(chat_model)
cached_crewai_llm = wrap_crewai_llm(llm)
```

Streaming/generator methods pass through by default. Correct streamed-response
caching needs a recorder/replay layer and is intentionally not hidden here.

## Prompt Caching Strategy Pattern

The main LLD extension point is `PromptCachingStrategy`.

```python
from prompt_cache_kit import CacheDirective, PromptCachingStrategy

class LastAssistantStrategy(PromptCachingStrategy):
    def select(self, context):
        for index in range(len(context.messages) - 1, -1, -1):
            if context.messages[index]["role"] == "assistant":
                return [CacheDirective(message_index=index, id="last-assistant-prefix")]
        return []
```

Built-in strategies:

- `CacheUntilPromptCachingStrategy`
- `ManualPromptCachingStrategy`
- `RollingPromptCachingStrategy`
- `StablePrefixPromptCachingStrategy`

Convenience factories:

```python
from prompt_cache_kit import cache_at, cache_until, rolling_cache

manual = cache_at(0, 2, 4, ids={0: "system-v3", 4: "old-thread-v1"}).excluding(2)
boundary = cache_until(1, id="system-and-document-v1")
rolling = rolling_cache(every_messages=2, min_tokens=1024, max_points=4)
```

Compile a strategy into provider-specific messages:

```python
from prompt_cache_kit import apply_cache_points

messages = [
    {"role": "system", "content": "Long stable system prompt...", "stable": True},
    {"role": "user", "content": "Long stable uploaded document...", "stable": True},
    {"role": "user", "content": "What changed in section 4?", "stable": False},
]

anthropic_messages = apply_cache_points(messages, plan=boundary, provider="anthropic")
bedrock_messages = apply_cache_points(messages, plan=boundary, provider="bedrock")
```

Anthropic output marks the text block with:

```python
{"cache_control": {"type": "ephemeral"}}
```

Bedrock/LangChain output appends:

```python
{"cachePoint": {"type": "default"}}
```

## Prompt Layout

`PromptLayout` helps keep stable content before dynamic content.

```python
from prompt_cache_kit import PromptLayout

layout = (
    PromptLayout()
    .stable_system("You are a precise research assistant.")
    .cache_point("system-v1", provider_hint="anthropic")
    .stable_context("Long product manual or policy document...")
    .cache_point("manual-v1")
    .dynamic_user("What changed in section 4?")
)

messages = layout.to_anthropic_messages()
instrumented = layout.to_instrumented_messages()
issues = layout.lint()
usage_before_call = layout.usage()
```

Provider payloads are clean. Internal metadata such as `stable` and `cache_point`
only appears in `to_instrumented_messages()`.

## Usage And Telemetry

Normalize provider usage objects:

```python
from prompt_cache_kit import normalize_usage

stats = normalize_usage({
    "usage": {
        "prompt_tokens": 10339,
        "completion_tokens": 60,
        "total_tokens": 10399,
        "prompt_tokens_details": {
            "cached_tokens": 10318,
            "cache_write_tokens": 0,
        },
    }
}, provider="openrouter")

print(stats.cache_read_input_tokens)
print(stats.to_openai_usage())
print(stats.to_otel_attributes())
```

LangChain usage extraction:

```python
from prompt_cache_kit import extract_langchain_usage

response = cached_langchain_model.invoke(messages)
usage = extract_langchain_usage(response)
print(usage.to_otel_attributes())
```

## LangChain And LangGraph

Use `wrap_langchain_model()` for response caching and `apply_cache_points()` for
prompt cache markers:

```python
from prompt_cache_kit import (
    CachePolicy,
    MemoryCacheBackend,
    StablePrefixPromptCachingStrategy,
    apply_cache_points,
    wrap_langchain_model,
)

cached_model = wrap_langchain_model(
    chat_model,
    backend=MemoryCacheBackend(),
    policy=CachePolicy(namespace="langgraph-node", ttl_seconds=300),
)

strategy = StablePrefixPromptCachingStrategy(min_tokens=1024, max_points=4)
messages = apply_cache_points(raw_messages, plan=strategy, provider="bedrock")
response = cached_model.invoke(messages)
```

For Bedrock via LangChain, the message content follows the documented pattern of
text blocks plus a `cachePoint` block.

## CrewAI

```python
from prompt_cache_kit import CachePolicy, MemoryCacheBackend, cache_until, apply_cache_points, wrap_crewai_llm

cached_llm = wrap_crewai_llm(
    llm,
    backend=MemoryCacheBackend(),
    policy=CachePolicy(namespace="crew", ttl_seconds=300),
)

messages = apply_cache_points(
    [
        {"role": "system", "content": "Long crew instructions...", "stable": True},
        {"role": "user", "content": "Current task", "stable": False},
    ],
    plan=cache_until(0, id="crew-system-v1"),
    provider="anthropic",
)

result = cached_llm.call(messages)
```

## vLLM And LMCache

For plain vLLM prefix caching:

```python
from prompt_cache_kit import VLLMConfig

cfg = VLLMConfig(model="Qwen/Qwen3-8B", enable_prefix_caching=True)
print("vllm serve " + " ".join(cfg.to_cli_args()))
```

For current LMCache multiprocess mode:

```python
mp = cfg.with_lmcache_mp(host="127.0.0.1", port=5555)
print("vllm serve " + " ".join(mp.to_cli_args()))
print(mp.to_engine_kwargs())
```

This emits a `kv_transfer_config` using `LMCacheMPConnector` and
`kv_connector_extra_config` with `lmcache.mp.host` and `lmcache.mp.port`.

For deployments using the simpler `LMCacheConnectorV1` shape:

```python
classic = cfg.with_lmcache_v1()
print(classic.kv_transfer_config)
```

LMCache HTTP control:

```python
from prompt_cache_kit import LMCacheClient

client = LMCacheClient("http://localhost:8080")
print(client.health())
print(client.status())
```

## Examples

See the [examples](examples) folder:

- [custom_inference.py](examples/custom_inference.py): response caching for a custom model function.
- [custom_strategy.py](examples/custom_strategy.py): extend `PromptCachingStrategy`.
- [redis_backend.py](examples/redis_backend.py): Redis response-cache backend.
- [langgraph_usage.py](examples/langgraph_usage.py): LangGraph-style node integration.
- [crewai_usage.py](examples/crewai_usage.py): CrewAI LLM wrapping.
- [langchain_bedrock_cachepoint.py](examples/langchain_bedrock_cachepoint.py): LangChain Bedrock `cachePoint` content.
- [vllm_lmcache_config.py](examples/vllm_lmcache_config.py): vLLM prefix caching and LMCache connector config.

## Package Boundaries

Prompt Cache Kit can:

- Decide where cache points should go.
- Render cache points for known provider formats.
- Cache complete LLM responses.
- Normalize token/cache usage.
- Generate vLLM/LMCache configuration snippets.

Prompt Cache Kit cannot:

- Read or write arbitrary model KV tensors.
- Make non-deterministic model calls deterministic.
- Guarantee provider-side prompt cache hits if the provider silently rejects a breakpoint.
- Replace LMCache, vLLM, SGLang, or provider-native prompt caches.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```
