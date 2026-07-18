#!/usr/bin/env bash
# Eidolon DB-Reset — löscht die SQLite-Datenbank, sodass sie beim nächsten Start neu angelegt wird.
#
# Usage:
#   ./tools/reset-db.sh                    # Löscht data/memory.db im Projekt-Root
#   ./tools/reset-db.sh /pfad/zur/db.db    # Löscht eine beliebige DB
#
# Die DB wird beim nächsten Start des Cores/Adapters automatisch neu angelegt
# (SqliteStore.connect() legt das Verzeichnis an + Migrationen laufen durch).

set -euo pipefail

DB_PATH="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/memory.db}"

if [ ! -f "$DB_PATH" ] && [ ! -f "${DB_PATH}-shm" ] && [ ! -f "${DB_PATH}-wal" ]; then
    echo "⚠  Keine Eidolon-DB gefunden unter: $DB_PATH"
    exit 0
fi

echo "🧹 Lösche Eidolon-Datenbank …"
rm -vf "$DB_PATH" "${DB_PATH}-shm" "${DB_PATH}-wal" 2>/dev/null || true
echo "✅ Erledigt. Die DB wird beim nächsten Start neu angelegt."
