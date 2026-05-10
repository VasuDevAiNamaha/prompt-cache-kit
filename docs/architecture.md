# Architecture

Prompt Cache Kit is organized around small extension points rather than one large
module. The layout follows a few SOLID-oriented boundaries:

## Package Layout

- `types/`: public protocols and value objects shared by other modules.
- `core/`: cache policy and deterministic key generation.
- `backends/`: cache storage implementations such as memory and Redis.
- `serializers/`: payload serializers used by storage backends.
- `wrappers/`: callable and model-object response-cache wrappers.
- `prompt_caching/`: provider-neutral strategy planning and provider-specific cache-point compilers.
- `prompt_layout.py`: high-level prompt builder for stable/dynamic content.
- `telemetry/`: usage normalization and OpenTelemetry-style exports.
- `integrations/`: framework and engine helpers such as LangChain, CrewAI, and vLLM.
- `clients/`: external service clients such as LMCache.

## Core Design

### Strategy

`PromptCachingStrategy` is the main extension point for deciding where cache
boundaries belong. Built-ins include manual, cache-until, stable-prefix, and
rolling strategies. User strategies only need to implement:

```python
def select(self, context: PromptCachingContext) -> Sequence[CacheDirective]:
    ...
```

### Compiler

`CachePointCompiler` turns provider-neutral cache points into provider-specific
message payloads. Built-ins support Anthropic, Bedrock/LangChain, and generic
metadata. New providers can be added without changing planning strategies.

### Backend

`CacheBackend` is a protocol. Wrappers depend on the protocol rather than concrete
storage classes, so users can add SQLite, DynamoDB, S3, or custom internal stores.

### Serializer

`Serializer` is injectable into backends that need byte payloads. Redis uses a
pickle serializer by default and can optionally wrap it with signature checking.

## Compatibility

The old single-file modules such as `prompt_cache_kit.backend` and
`prompt_cache_kit.cache_points` remain as thin compatibility exports where that
does not conflict with the new package layout.
