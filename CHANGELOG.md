# Changelog

All notable changes to Prompt Cache Kit will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic
versioning once it reaches a stable API.

## [0.1.0] - 2026-05-10

### Added

- Response caching wrapper for sync and async Python callables.
- `CachedModel` duck-typed wrapper for model-like objects.
- In-memory and Redis cache backends.
- Prompt caching Strategy pattern with built-in manual, cache-until, rolling, and stable-prefix strategies.
- Provider compilers for Anthropic `cache_control`, Bedrock/LangChain `cachePoint`, and generic metadata.
- Prompt layout helper, cache-point instrumentation, and prompt linting.
- Usage normalization for OpenAI/OpenRouter, Anthropic, and LangChain-style metadata.
- OpenTelemetry GenAI-style usage attribute export.
- vLLM prefix-caching and LMCache connector configuration helpers.
- Examples for custom inference, Redis, LangGraph, CrewAI, LangChain Bedrock, custom strategy, and vLLM/LMCache.
