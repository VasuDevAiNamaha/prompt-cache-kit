from __future__ import annotations

import json
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
        """Attach the LMCache multiprocess connector used by `lmcache server`."""

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
