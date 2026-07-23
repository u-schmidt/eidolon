"""Eidolon Hermes Memory Provider Plugin — entry point.

Full Mode: Replaces Honcho as the active memory provider.
Provides topic-based, associative memory with perspective attribution.

Directory structure (inside Hermes plugins/):
    plugins/eidolon/
    ├── __init__.py           # register() entry point (this file)
    ├── memory_provider.py    # EidolonMemoryProvider implementation
    ├── plugin.yaml           # Metadata
    └── README.md             # Setup instructions
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the plugin root is on sys.path so core/, providers/ etc. are importable
# __file__ is adapters/hermes/__init__.py, so parent.parent.parent is the plugin root
_plugin_root = str(Path(__file__).resolve().parent.parent.parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

from adapters.hermes.memory_provider import EidolonMemoryProvider

# ── Plugin Entry Point ──────────────────────────────────


def register(ctx) -> None:
    """Register the Eidolon memory provider with Hermes.

    Called by the Hermes plugin system when the plugin is loaded.
    """
    ctx.register_memory_provider(EidolonMemoryProvider())
