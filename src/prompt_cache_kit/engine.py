from __future__ import annotations

import json
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class VLLMConfig:
    """Configuration helper for vLLM prefix caching and LMCache connectors."""

    model: str
    enable_prefix_caching: bool = True
    kv_transfer_config: Optional[Mapping[str, Any]] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def with_lmcache(self, *, role: str = "kv_both", connector: str = "LMCacheConnectorV1") -> "VLLMConfig":
        """Attach the classic LMCache connector shape used by many vLLM examples."""

        config = {"kv_connector": connector, "kv_role": role}
        return replace(self, kv_transfer_config=config)

    def with_lmcache_v1(self, *, role: str = "kv_both") -> "VLLMConfig":
        """Attach `LMCacheConnectorV1`.

        This remains useful for deployments and docs that configure LMCache via
        environment variables or non-MP integration paths.
        """

        return self.with_lmcache(role=role, connector="LMCacheConnectorV1")

    def with_lmcache_mp(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        role: str = "kv_both",
        connector: str = "LMCacheMPConnector",
        extra_config: Optional[Mapping[str, Any]] = None,
    ) -> "VLLMConfig":
        """Attach the LMCache multiprocess connector used by `lmcache server`.

        LMCache MP runs as a standalone service and vLLM connects over ZMQ. The
        current LMCache configuration docs use `LMCacheMPConnector` with
        `kv_connector_extra_config` keys for the MP host and port.
        """

        connector_config: Dict[str, Any] = {
            "lmcache.mp.host": host,
            "lmcache.mp.port": port,
        }
        connector_config.update(dict(extra_config or {}))
        return replace(
            self,
            kv_transfer_config={
                "kv_connector": connector,
                "kv_role": role,
                "kv_connector_extra_config": connector_config,
            },
        )

    def to_cli_args(self) -> List[str]:
        args = ["--model", self.model]
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.kv_transfer_config:
            args.extend(["--kv-transfer-config", json.dumps(dict(self.kv_transfer_config), sort_keys=True)])
        for key, value in sorted(self.extra_args.items()):
            cli_key = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(cli_key)
            else:
                args.extend([cli_key, str(value)])
        return args

    def to_engine_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": self.model, "enable_prefix_caching": self.enable_prefix_caching}
        if self.kv_transfer_config:
            kwargs["kv_transfer_config"] = dict(self.kv_transfer_config)
        kwargs.update(self.extra_args)
        return kwargs


class LMCacheClient:
    """Tiny HTTP client for LMCache server health and control endpoints."""

    def __init__(self, base_url: str, *, timeout: float = 3.0) -> None:
        validated = _validate_http_base_url(base_url)
        self.base_url = validated.rstrip("/")
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def health(self) -> Dict[str, Any]:
        return self._request("GET", ("/api/healthcheck", "/healthcheck", "/health"))

    def status(self) -> Dict[str, Any]:
        return self._request("GET", ("/api/status", "/status"))

    def clear_cache(self) -> Dict[str, Any]:
        return self._request("POST", ("/api/clear-cache", "/clear-cache"))

    def _request(self, method: str, paths: tuple[str, ...]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for path in paths:
            request = urllib.request.Request(f"{self.base_url}{path}", method=method)
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                    if not body:
                        return {"ok": True, "status": response.status}
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = {"body": body}
                    parsed.setdefault("ok", 200 <= response.status < 300)
                    parsed.setdefault("status", response.status)
                    return parsed
            except urllib.error.HTTPError as exc:
                # Treat HTTP errors as responses (common when probing multiple endpoints).
                try:
                    body = exc.read().decode("utf-8")
                except Exception:
                    body = ""
                if body:
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = {"body": body}
                else:
                    parsed = {}
                parsed.setdefault("ok", False)
                parsed.setdefault("status", getattr(exc, "code", None))
                parsed.setdefault("error", getattr(exc, "reason", None) or str(exc))
                return parsed
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
        return {"ok": False, "error": str(last_error) if last_error else "request failed"}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, "redirect blocked", headers, fp)


def _validate_http_base_url(base_url: str) -> str:
    """Prevent unexpected URL schemes (e.g. file://) and missing hosts."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base_url must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("base_url must include a host")
    # Avoid embedded credentials leaking via logs/tracebacks.
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include credentials")
    if parsed.fragment:
        raise ValueError("base_url must not include a URL fragment")
    return urllib.parse.urlunsplit(parsed)
