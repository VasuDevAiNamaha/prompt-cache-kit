# Contributing

Thanks for helping make Prompt Cache Kit better.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,redis,langchain]"
pre-commit install
```

## Checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m bandit -q -r src
```

## Pull Requests

- Keep changes focused.
- Add tests for new behavior.
- Update README/examples when public APIs change.
- Do not claim provider or engine KV-cache behavior that the package cannot enforce.

## Release Checklist

1. Update `CHANGELOG.md`.
2. Run the full test and build checks.
3. Build with `python -m build`.
4. Check metadata with `python -m twine check dist/*`.
5. Publish through the GitHub release workflow using PyPI Trusted Publishing.
