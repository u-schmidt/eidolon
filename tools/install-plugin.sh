#!/usr/bin/env bash
# Eidolon Plugin Installer — copies the project as a Hermes plugin.
#
# Usage:
#   HERMES_HOME=/path/to/hermes ./tools/install-plugin.sh
#   HERMES_HOME=/path/to/hermes PROFILE=eidolon-test ./tools/install-plugin.sh
#
# Environment variables:
#   HERMES_HOME  (required) — Hermes data directory (e.g. ~/.hermes or .../hermes-data)
#   PROFILE      (optional) — profile name, default: eidolon-test

set -euo pipefail

# ── Config ────────────────────────────────────────────────
HERMES_HOME="${HERMES_HOME:?❌ HERMES_HOME is not set. Usage: HERMES_HOME=/path/to/hermes ./tools/install-plugin.sh}"
PROFILE="${PROFILE:-eidolon-test}"
PLUGIN_DIR="$HERMES_HOME/profiles/$PROFILE/plugins/eidolon"
CONFIG_FILE="$HERMES_HOME/profiles/$PROFILE/eidolon.yaml"
EXAMPLE_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/eidolon.example.yaml"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔══════════════════════════════════════════════╗"
echo "║     Eidolon Plugin Installer                 ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Profile:      $PROFILE"
echo "  Plugin target: $PLUGIN_DIR"
echo "  Config:       $CONFIG_FILE"
echo ""

# ── Check ───────────────────────────────────────────────
if [ ! -d "$HERMES_HOME/profiles/$PROFILE" ]; then
    echo "❌ Profile '$PROFILE' does not exist under $HERMES_HOME/profiles/$PROFILE"
    echo "   Create it first: hermes setup --profile $PROFILE"
    exit 1
fi

# ── Copy plugin ──────────────────────────────────────────
# rsync instead of cp -r + rm -rf: data/ under the running plugin at
# $PLUGIN_DIR holds the real eidolon_memory.db from live operation! A
# previous rm -rf + cp -r of the project root (which has its own local
# data/) once overwrote it. rsync with --exclude data leaves the target
# directory untouched there; --delete only cleans up deleted/renamed
# code files, never the excluded paths.
echo "📦 Syncing project to $PLUGIN_DIR …"
mkdir -p "$PLUGIN_DIR"
# tests/ is no longer needed at runtime: core/provider_factory.py imports
# SimulatedLLM/SimulatedEmbedder from core/providers/simulated.py, not
# from tests/mocks.py anymore — no runtime dependency from
# core/adapters/providers on tests/ remains.
#
# tools/ is pure dev tooling (CLI, trace TUI, this installer itself) —
# core/ and adapters/ don't import anything from it, and cli.py/trace_tui.py
# take the DB path via --storage-path/--db arguments instead of assuming
# it relative to their own directory. Run them from the dev checkout
# against the deployed DB, not from the plugin directory.
#
# AGENTS.md/CLAUDE.md are coding conventions for working on this
# project — no runtime relevance, don't belong in a deployed plugin.
#
# eidolon.example.yaml is read separately below from $PROJECT_ROOT (see
# $EXAMPLE_CONFIG), not from the rsync target — the copy in the plugin
# directory wouldn't be read by anything at runtime.
rsync -a --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.claude' \
    --exclude='.gitignore' \
    --exclude='data' \
    --exclude='tests' \
    --exclude='tools' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='.mypy_cache' \
    --exclude='eidolon.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='uv.lock' \
    --exclude='pyproject.toml' \
    --exclude='AGENTS.md' \
    --exclude='CLAUDE.md' \
    --exclude='eidolon.example.yaml' \
    "$PROJECT_ROOT/" "$PLUGIN_DIR/"
echo "   ✅ Done (data/, .git/, .venv/ untouched at the target)"

# ── Copy plugin.yaml to root ─────────────────────────────
# Hermes' discovery only reads plugin.yaml from $PLUGIN_DIR/plugin.yaml
# (for the description). The original lives in the project under
# adapters/hermes/plugin.yaml.
ROOT_PLUGIN_YAML="$PLUGIN_DIR/plugin.yaml"
ADAPTER_PLUGIN_YAML="$PLUGIN_DIR/adapters/hermes/plugin.yaml"
if [ -f "$ADAPTER_PLUGIN_YAML" ] && [ ! -f "$ROOT_PLUGIN_YAML" ]; then
    echo "📝 Copying plugin.yaml to root position …"
    cp "$ADAPTER_PLUGIN_YAML" "$ROOT_PLUGIN_YAML"
    echo "   ✅ plugin.yaml copied to $ROOT_PLUGIN_YAML"
elif [ -f "$ROOT_PLUGIN_YAML" ]; then
    echo "   ✅ plugin.yaml already present at root position"
else
    echo "   ⚠  No plugin.yaml found under adapters/hermes/"
fi

# ── Create __init__.py shim (if missing) ─────────────────
# Hermes' plugin discovery looks for $HERMES_HOME/plugins/<name>/__init__.py
# with a register() function. The actual implementation lives in
# adapters/hermes/__init__.py — this shim forwards to it.
SHIM="$PLUGIN_DIR/__init__.py"
if [ ! -f "$SHIM" ]; then
    echo "📝 Creating __init__.py shim for Hermes plugin discovery …"
    cat > "$SHIM" << 'PYEOF'
"""Eidolon — topic memory provider for Hermes Agent.

Re-exports the register() entry point from the Hermes adapter.
The actual implementation lives in adapters/hermes/__init__.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# MemoryProvider / register_memory_provider  # HERMES MEMORY PLUGIN MARKER — DO NOT REMOVE
_plugin_root = str(Path(__file__).resolve().parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

from adapters.hermes import register  # noqa: E402, F401
PYEOF
    echo "   ✅ __init__.py shim created"
else
    echo "   ✅ __init__.py shim already present"
fi

# ── Create config (if missing) ───────────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo "📝 No eidolon.yaml found — creating example config …"
    cp "$EXAMPLE_CONFIG" "$CONFIG_FILE"
    echo "   ⚠  Please adjust the values in $CONFIG_FILE:"
    echo "      - models.llm.base_url"
    echo "      - models.llm.model"
    echo "      - models.embedder.base_url"
    echo "      - models.embedder.model"
else
    echo ""
    echo "✅ eidolon.yaml present: $CONFIG_FILE"
fi

# ── Check memory.provider ────────────────────────────────
CONFIG_YAML="$HERMES_HOME/profiles/$PROFILE/config.yaml"
if [ -f "$CONFIG_YAML" ]; then
    if grep -q "provider: eidolon" "$CONFIG_YAML" 2>/dev/null; then
        echo "✅ memory.provider is already set to eidolon"
    else
        echo ""
        echo "⚠  memory.provider is NOT set to eidolon."
        echo "   Set in $CONFIG_YAML:"
        echo "     memory:"
        echo "       provider: eidolon"
    fi
else
    echo ""
    echo "⚠  No config.yaml found under $CONFIG_YAML"
fi

echo ""
echo "🚀 Installation complete!"
echo "   Restart the gateway (Dashboard → Restart) to activate the plugin."
