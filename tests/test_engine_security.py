import pytest

from prompt_cache_kit import LMCacheClient


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "   ",
        "file:///etc/passwd",
        "ftp://localhost:8080",
        "http:///missing-host",
        "https://user:pass@example.com",
        "https://example.com/#fragment",
    ],
)
def test_lmcache_client_rejects_unsafe_base_urls(base_url):
    with pytest.raises(ValueError):
        LMCacheClient(base_url)


def test_lmcache_client_accepts_http_urls():
    client = LMCacheClient("http://127.0.0.1:8080")
    assert client.base_url == "http://127.0.0.1:8080"
