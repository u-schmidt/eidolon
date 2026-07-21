# AGENTS.md — Eidolon Coding Conventions

## Architecture
- Prefer strict OOP: classes over free functions
- Each class in its own file
- Interfaces/ABCs for provider abstractions (`typing.Protocol` or `abc.ABC`)
- Private attributes with a `_` prefix, `@property` for access
- No free functions that mutate objects — if a function changes an object, it belongs on the class as a method
- `@dataclass` for pure data containers (like a `record` in C#)
- Use `@staticmethod` and `@classmethod` deliberately, not as the default

## Core ≠ Adapter
- `core/` never imports from `adapters/` — only the other way around
- Providers are defined as Protocols; concrete implementations are injected via config
- The core is stateless — all session information comes in as parameters

## Testing
- pytest for unit tests in the `tests/` folder
- Every new class needs at least one test
- Integration tests only when necessary (database, API, filesystem)
- `uv run pytest` before every commit
- Test files mirror the module structure: `tests/test_<module>.py`

## Documentation
- After larger implementation cycles (new features, larger refactorings): check README.md for accuracy — architecture tree, config examples, roadmap/phase status, test count

## Code Style
- Type hints are mandatory (including for `self` and `cls`)
- No wild `**kwargs` dictionaries — explicit parameters
- Docstrings only for public APIs, not for trivial getters
- No free functions that mutate imported objects (preserve encapsulation)
- `from __future__ import annotations` at the top of every file

## Avoiding Errors
- No circular imports — architecture runs outside in: Adapter → Core → Providers → Models
- No God classes — if a class exceeds 300 lines, it's too big
- No `from x import *` — explicit imports
- No mutable default arguments — `def foo(items: list = None)` instead of `def foo(items: [])`
