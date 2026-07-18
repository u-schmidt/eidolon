#!/usr/bin/env bash
# Eidolon Plugin-Installer — kopiert das Projekt als Hermes-Plugin.
#
# Usage:
#   HERMES_HOME=/pfad/zu/hermes ./tools/install-plugin.sh
#   HERMES_HOME=/pfad/zu/hermes PROFILE=eidolon-test ./tools/install-plugin.sh
#
# Environment-Variablen:
#   HERMES_HOME  (Pflicht) — Hermes-Datenverzeichnis (z.B. ~/.hermes oder .../hermes-data)
#   PROFILE      (Optional) — Profilname, Standard: eidolon-test
#
# Beispiel (Alex):
#   HERMES_HOME=/Users/udo/Documents/AILocal/hermes/hermes-data ./tools/install-plugin.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────
HERMES_HOME="${HERMES_HOME:?❌ HERMES_HOME ist nicht gesetzt. Usage: HERMES_HOME=/pfad/zu/hermes ./tools/install-plugin.sh}"
PROFILE="${PROFILE:-eidolon-test}"
PLUGIN_DIR="$HERMES_HOME/profiles/$PROFILE/plugins/eidolon"
CONFIG_FILE="$HERMES_HOME/profiles/$PROFILE/eidolon.yaml"
EXAMPLE_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/eidolon.example.yaml"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔══════════════════════════════════════════════╗"
echo "║     Eidolon Plugin-Installer                 ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Profil:      $PROFILE"
echo "  Plugin-Ziel: $PLUGIN_DIR"
echo "  Config:      $CONFIG_FILE"
echo ""

# ── Prüfen ──────────────────────────────────────────────
if [ ! -d "$HERMES_HOME/profiles/$PROFILE" ]; then
    echo "❌ Profil '$PROFILE' existiert nicht unter $HERMES_HOME/profiles/$PROFILE"
    echo "   Lege es zuerst an: hermes setup --profile $PROFILE"
    exit 1
fi

# ── Plugin kopieren ─────────────────────────────────────
# rsync statt cp -r + rm -rf: data/ liegt beim laufenden Plugin unter
# $PLUGIN_DIR (die echte eidolon_memory.db aus dem Live-Betrieb!). Ein
# vorheriges rm -rf + cp -r des Projekt-Roots (der sein eigenes lokales
# data/ hat) hat das schon mal überschrieben. rsync mit --exclude data
# lässt das Zielverzeichnis dort unangetastet; --delete räumt nur
# gelöschte/umbenannte Code-Dateien auf, nie die ausgeschlossenen Pfade.
echo "📦 Synchronisiere Projekt nach $PLUGIN_DIR …"
mkdir -p "$PLUGIN_DIR"
# tests/ wird nicht mehr benötigt: core/provider_factory.py importiert
# SimulatedLLM/SimulatedEmbedder aus providers/simulated.py, nicht mehr
# aus tests/mocks.py — kein Laufzeit-Bezug von core/adapters/providers
# auf tests/ mehr vorhanden.
rsync -a --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.claude' \
    --exclude='.gitignore' \
    --exclude='data' \
    --exclude='tests' \
    --exclude='.pytest_cache' \
    --exclude='eidolon.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='uv.lock' \
    --exclude='pyproject.toml' \
    "$PROJECT_ROOT/" "$PLUGIN_DIR/"
echo "   ✅ Erledigt (data/, .git/, .venv/ am Ziel unangetastet)"

# ── Config anlegen (falls nicht vorhanden) ──────────────
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo "📝 Keine eidolon.yaml gefunden — lege Beispiel-Config an …"
    cp "$EXAMPLE_CONFIG" "$CONFIG_FILE"
    echo "   ⚠  Bitte passe die Werte in $CONFIG_FILE an:"
    echo "      - models.llm.base_url"
    echo "      - models.llm.model"
    echo "      - models.embedder.base_url"
    echo "      - models.embedder.model"
else
    echo ""
    echo "✅ eidolon.yaml vorhanden: $CONFIG_FILE"
fi

# ── memory.provider prüfen ──────────────────────────────
CONFIG_YAML="$HERMES_HOME/profiles/$PROFILE/config.yaml"
if [ -f "$CONFIG_YAML" ]; then
    if grep -q "provider: eidolon" "$CONFIG_YAML" 2>/dev/null; then
        echo "✅ memory.provider ist bereits auf eidolon gesetzt"
    else
        echo ""
        echo "⚠  memory.provider ist NICHT auf eidolon gesetzt."
        echo "   Setze in $CONFIG_YAML:"
        echo "     memory:"
        echo "       provider: eidolon"
    fi
else
    echo ""
    echo "⚠  Keine config.yaml gefunden unter $CONFIG_YAML"
fi

echo ""
echo "🚀 Installation abgeschlossen!"
echo "   Starte das Gateway neu (Dashboard → Restart), um das Plugin zu aktivieren."
