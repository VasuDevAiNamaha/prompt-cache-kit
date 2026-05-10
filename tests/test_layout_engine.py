from prompt_cache_kit import PromptLayout, VLLMConfig


def test_prompt_layout_renders_stable_blocks_first():
    layout = PromptLayout().dynamic_user("question").stable_system("rules", name="rules").stable_context("manual")

    messages = layout.to_openai_messages()

    assert [message["content"] for message in messages] == ["rules", "manual", "question"]
    assert layout.lint()[0].code == "stable-after-dynamic"


def test_prompt_layout_anthropic_cache_control():
    layout = PromptLayout().stable_system("rules").dynamic_user("question")

    messages = layout.to_anthropic_messages()

    assert messages[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[1]


def test_provider_messages_do_not_include_internal_instrumentation():
    layout = PromptLayout().stable_system("rules").cache_point("rules-v1").dynamic_user("question")

    messages = layout.to_openai_messages()
    instrumented = layout.to_instrumented_messages()

    assert "cache_point" not in messages[0]
    assert "stable" not in messages[0]
    assert instrumented[0]["cache_point"]["id"] == "rules-v1"


def test_lint_detects_dynamic_marker_in_stable_block():
    layout = PromptLayout().stable_system("request_id: 123")

    assert layout.lint()[0].code == "dynamic-marker-in-stable-block"


def test_vllm_config_cli_and_kwargs():
    cfg = VLLMConfig(model="test-model", extra_args={"tensor_parallel_size": 2}).with_lmcache()

    args = cfg.to_cli_args()
    kwargs = cfg.to_engine_kwargs()

    assert "--enable-prefix-caching" in args
    assert "--kv-transfer-config" in args
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["kv_transfer_config"]["kv_connector"] == "LMCacheConnectorV1"


def test_vllm_config_lmcache_mp_connector():
    cfg = VLLMConfig(model="test-model").with_lmcache_mp(host="10.0.0.5", port=6000)

    args = cfg.to_cli_args()
    kwargs = cfg.to_engine_kwargs()

    assert "--kv-transfer-config" in args
    assert kwargs["kv_transfer_config"]["kv_connector"] == "LMCacheMPConnector"
    assert kwargs["kv_transfer_config"]["kv_connector_extra_config"]["lmcache.mp.host"] == "10.0.0.5"
    assert kwargs["kv_transfer_config"]["kv_connector_extra_config"]["lmcache.mp.port"] == 6000
