"""Generate vLLM/LMCache configuration snippets.

This does not start vLLM or LMCache. It prints the CLI args that should be used
with those services.
"""

from prompt_cache_kit import LMCacheClient, VLLMConfig


prefix_only = VLLMConfig(model="Qwen/Qwen3-8B", enable_prefix_caching=True)
print("vLLM prefix caching:")
print("vllm serve " + " ".join(prefix_only.to_cli_args()))

mp = prefix_only.with_lmcache_mp(host="127.0.0.1", port=5555)
print("\nvLLM + LMCache multiprocess connector:")
print("vllm serve " + " ".join(mp.to_cli_args()))
print(mp.to_engine_kwargs())

classic = prefix_only.with_lmcache_v1()
print("\nvLLM + LMCacheConnectorV1 style:")
print("vllm serve " + " ".join(classic.to_cli_args()))

client = LMCacheClient("http://localhost:8080")
print("\nLMCache health result if server is running:")
print(client.health())
