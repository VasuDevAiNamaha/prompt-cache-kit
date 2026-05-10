from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


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
