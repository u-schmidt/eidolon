#!/usr/bin/env bash
# Eidolon DB reset — deletes the SQLite database so it gets recreated on next start.
#
# Usage:
#   ./tools/reset-db.sh                    # deletes data/memory.db in the project root
#   ./tools/reset-db.sh /path/to/db.db     # deletes an arbitrary DB
#
# The DB gets recreated automatically on the core/adapter's next start
# (SqliteStore.connect() creates the directory and runs migrations).

set -euo pipefail

DB_PATH="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/memory.db}"

if [ ! -f "$DB_PATH" ] && [ ! -f "${DB_PATH}-shm" ] && [ ! -f "${DB_PATH}-wal" ]; then
    echo "⚠  No Eidolon DB found at: $DB_PATH"
    exit 0
fi

echo "🧹 Deleting Eidolon database …"
rm -vf "$DB_PATH" "${DB_PATH}-shm" "${DB_PATH}-wal" 2>/dev/null || true
echo "✅ Done. The DB will be recreated on next start."
