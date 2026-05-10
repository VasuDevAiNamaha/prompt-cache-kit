# Security Policy

Prompt Cache Kit can cache sensitive prompts and LLM responses. Treat cache
backends as application data stores.

## Reporting A Vulnerability

Please open a private security advisory on GitHub or contact the maintainers
through the repository owner profile.

## Security Notes

- `RedisCacheBackend` uses pickle to support arbitrary Python response objects.
  Use it only with trusted Redis instances and trusted application data.
- Prefer short TTLs for sensitive LLM workloads.
- Use separate namespaces or Redis databases per application or tenant.
- Do not cache secrets, API keys, authorization headers, or per-user private data
  unless your storage and access controls are designed for it.
