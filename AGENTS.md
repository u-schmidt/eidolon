# AGENTS.md — Eidolon Coding Conventions

## Architecture
- Bevorzuge strikte OOP: Klassen statt freie Funktionen
- Jede Klasse in eigener Datei
- Interfaces/ABCs für Provider-Abstraktionen (`typing.Protocol` oder `abc.ABC`)
- Private Attribute mit `_`-Prefix, `@property` für Zugriff
- Keine freien Funktionen die Objekte mutieren — wenn eine Funktion ein Objekt verändert, gehört sie als Methode in die Klasse
- `@dataclass` für reine Datencontainer (wie `record` in C#)
- `@staticmethod` und `@classmethod` bewusst einsetzen, nicht als Standard

## Core ≠ Adapter
- `core/` importiert niemals aus `adapters/` — nur umgekehrt
- Provider sind als Protocols definiert, konkrete Implementierungen werden via Config injiziert
- Der Core ist stateless — alle Session-Informationen kommen als Parameter

## Testing
- pytest für Unit-Tests im `tests/`-Ordner
- Jede neue Klasse braucht mindestens einen Test
- Integration-Tests nur wenn nötig (Datenbank, API, Dateisystem)
- `uv run pytest` vor jedem Commit
- Test-Dateien spiegeln die Modul-Struktur: `tests/test_<modul>.py`

## Dokumentation
- Nach größeren Implementierungs-Zyklen (neue Features, größere Refactorings): README.md auf Aktualität prüfen — Architektur-Baum, Config-Beispiele, Roadmap/Phasen-Status, Testanzahl

## Code Style
- Type Hints sind Pflicht (auch für `self` und `cls`)
- Keine wilden `**kwargs`-Dictionaries — explizite Parameter
- Docstrings nur bei öffentlichen APIs, nicht bei trivialen Gettern
- Keine freien Funktionen die importierte Objekte mutieren (Kapselung wahren)
- `from __future__ import annotations` am Anfang jeder Datei

## Fehlervermeidung
- Keine circular imports — Architektur von aussen nach innen: Adapter → Core → Providers → Models
- Keine God Classes — wenn eine Klasse >300 Zeilen, ist sie zu gross
- Keine `from x import *` — explizite Imports
- Keine mutable Default-Argumente — `def foo(items: list = None)` statt `def foo(items: [])`
