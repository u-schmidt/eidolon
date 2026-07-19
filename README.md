# Eidolon

> A topic memory system: human-like memory for AI agents.
> Segments conversations into topics, extracts memory fragments (not transcripts),
> and retrieves them associatively across multiple dimensions.

Eidolon models how humans actually remember — not a perfect database, but a system that feels natural, associative, and "human enough." It segments conversations into topics, extracts the essence (not the full transcript), and retrieves related memories across semantic, tag, person, and temporal dimensions.

## Architecture

```
eidolon/
├── core/                     # Shared Core (stateless, provider-abstracted)
│   ├── detection/            #   Topic Detector (CONTINUE / NEW_TOPIC)
│   ├── extraction/           #   Fragmentary Extractor (essence, facts, surprises, feelings)
│   ├── retrieval/            #   Associative Retriever (semantic + tag + person + time)
│   ├── consolidation/        #   Fact Consolidator — periodic durable-facts profile reconciliation
│   ├── stores/               #   SQLite store with auto-migration
│   ├── trace/                #   Debug tracer (NullTracer / SqliteTracer), off by default
│   ├── providers/            #   Provider implementations (one class per service)
│   │   ├── lm_studio.py      #     LM Studio: complete() + embed()
│   │   ├── ollama.py         #     Ollama: complete() + embed() (OpenAI-compatible)
│   │   ├── sentence_transformers.py  # Local PyTorch embeddings (fallback)
│   │   └── simulated.py      #     Deterministic fake provider for tests/CLI dry-runs
│   └── context_formatter.py  #   Renders a ContextBundle into the prompt-injection text
├── adapters/                 # Platform-specific adapters
│   └── hermes/               #   Hermes MemoryProvider plugin
├── tools/                    # Standalone tools
│   ├── cli.py                #   CLI for testing without Hermes
│   ├── trace.py              #   Terminal viewer for debug trace events
│   ├── trace_tui.py          #   Interactive drill-down TUI over trace.db (needs `tui` extra)
│   ├── tui/                  #   Textual screens for trace_tui.py
│   └── simulate_injection.py #   Replays a turn's prefetch() injection string from trace.db
├── tests/                    # Test suite
│   ├── core/                 #   Core tests (incl. core/trace/)
│   ├── adapters/hermes/      #   Adapter tests
│   └── tools/                #   Trace viewer + TUI tests
├── eidolon.example.yaml      # Example configuration
├── AGENTS.md                 # Coding conventions
├── plan.md                   # Phase 1 implementation plan
└── README.md                 # This file
```

### Core Principles

1. **Core ≠ Adapter** — The core library (`core/`) is standalone and stateless. Hermes is just the first adapter. The core must be usable without Hermes.
2. **Provider Abstraction** — Storage, LLM, and Embedder are swappable via configuration. No hardcoded providers.
3. **"Human Enough" Over Perfection** — Errors and omissions are acceptable. Graceful degradation is a feature.

### Three Memory Layers

| Layer | Description | Persistence |
|---|---|---|
| **Facts** | A small, curated profile of durable identity-level facts about a peer (name, location, relationships, health, work, stable preferences) — reconciled periodically, not per-turn; see [Facts Consolidation](#facts-consolidation) | Persistent across sessions |
| **Topics** | Episodic conversation segments with summaries, tags, embeddings | Persistent across sessions |
| **Memory Items** | Extracted fragments (essence, fact, surprise, feeling) with perspective attribution | Persistent across sessions |

## Per-Turn Flow

```
User Message
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Topic Detection (LLM, ~50 tok)   │  ← SYNC (before LLM response)
│    → CONTINUE or NEW_TOPIC:<label>  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Embedding (local, ~5ms)          │
│ 3. Associative Retrieval (SQL+vec)  │
│    → Facts + Related Topics + Items │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. LLM generates response           │  ← Hermes does this
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Fragmentary Extraction (LLM)     │  ← ASYNC (after LLM response)
│ 6. Topic Summary + Tags (LLM)       │
│ 7. Embedding storage                │
│ 8. Facts Consolidation (LLM)        │  ← every N turns, not every turn
└─────────────────────────────────────┘  see Facts Consolidation below
```

Every step above (prompts, responses, durations, retrieval score breakdowns, and otherwise-silent failures) can be recorded to a separate trace database for inspection — see [Debug Tracing](#debug-tracing) below. Off by default; zero behavior/performance impact unless enabled.

## Facts Consolidation

The facts profile is intentionally small (~50-100 entries) and curated, not a growing log — every `item_type="fact"` memory item is still stored and embedded as usual (searchable via associative retrieval, same as any other memory item), but converting a fact-shaped statement into a permanent profile row happens in a periodic **batch reconciliation pass** instead of immediately on every turn.

Every `facts.consolidation_interval` user turns (default 20, see [Configuration](#configuration)), `FactConsolidator` ([`core/consolidation/fact_consolidator.py`](core/consolidation/fact_consolidator.py)) sends the LLM the *entire current profile* plus everything fact-shaped learned since the last pass, and asks for a diff — only what changed:

```json
{"upserts": [{"category": "location", "key": "city", "value": "Berlin"}], "deletes": [{"category": "lifestyle", "key": "smoking"}]}
```

Seeing the full existing profile each time lets the LLM reuse an existing `(category, key)` when the same concept is being confirmed or corrected — moving to Berlin updates the existing `location` entry in place instead of adding a duplicate — while omitted entries are left untouched. A checkpoint (the highest `memory_items.id` already incorporated) ensures a failed pass gets retried at the next trigger rather than silently dropping those facts.

Trigger it manually any time — useful for testing without waiting N turns:

```
You > /consolidate
```

## Tool Calls

The Hermes adapter exposes two tools to the LLM (`EidolonMemoryProvider.get_tool_schemas()`):

| Tool | Effect |
|---|---|
| `eidolon_search` | Read-only. Runs associative retrieval for an explicit query, returning facts, related topics, and memory items as JSON. |
| `eidolon_correct_fact` | Writes directly. Upserts or deletes one `(category, key)` fact immediately, bypassing consolidation — for when the user explicitly says a remembered fact is wrong or should be forgotten. Omitting `value` deletes the fact. |

`eidolon_correct_fact` is a deliberately narrow write surface: it can only touch the curated facts profile, not `memory_items` or topics. A manual correction can still be silently revised or reverted by the next automatic consolidation pass, since `FactConsolidator` has no concept of "this fact was manually pinned" — an accepted tradeoff, not a bug.

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` (recommended) or `pip`

### Setup

```bash
# Clone / enter the project
cd eidolon

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run tests
uv run pytest tests/ -v
```

### CLI (standalone, without Hermes)

```bash
# With simulated providers (no external dependencies):
python tools/cli.py

# With real providers — provider, base_url, AND model are all required;
# an incomplete role fails fast with a clear error instead of guessing
# an endpoint (see Configuration below):
python tools/cli.py \
  --llm-provider ollama --llm-base-url http://localhost:11434 --llm-model deepseek-v4-flash:cloud \
  --embedder-provider lm_studio --embedder-base-url http://localhost:1234 --embedder-model text-embedding-bge-m3

# With a YAML config file:
python tools/cli.py --config ~/.eidolon/config.yaml

# With debug tracing enabled:
python tools/cli.py --debug --trace-db /tmp/trace.db
```

### Hermes Adapter

See [`adapters/hermes/README.md`](adapters/hermes/README.md) for installation and configuration instructions.

## Configuration

Eidolon uses a YAML-based configuration system with three sources (highest wins):

1. **CLI arguments** (e.g. `--llm-provider ollama`)
2. **Environment variables** (e.g. `EIDOLON_LLM_PROVIDER=ollama`)
3. **YAML config file** (e.g. `~/.eidolon/config.yaml`)

See [`eidolon.example.yaml`](eidolon.example.yaml) for all available options.

### Key Settings

Each model role (`llm`, `embedder`, and any future role like `judge`) is a **self-contained** endpoint — provider, `base_url`, and `model` together, no shared registry, no cross-references by name. A role that's actually used but not fully specified (missing `base_url`, for instance) fails immediately with a clear error at startup rather than silently falling back to a guessed endpoint — this used to silently misroute requests to whatever happened to be listening on a default port.

```yaml
models:
  llm:
    provider: ollama
    base_url: http://192.168.178.81:11434
    model: deepseek-v4-flash:cloud

  embedder:
    provider: lm_studio
    base_url: http://192.168.178.81:1234
    model: text-embedding-bge-m3

facts:
  consolidation_interval: 20  # See Facts Consolidation above
  target_max: 100

debug:
  enabled: false             # See Debug Tracing below
  trace_db_path: ~/.eidolon/trace.db
```

`llm` and `embedder` have dedicated CLI flags/env vars for quick overrides without touching YAML: `--llm-provider`/`--llm-base-url`/`--llm-model`/`--llm-max-tokens`/`--llm-timeout` (and the `embedder-`/`EIDOLON_LLM_*`/`EIDOLON_EMBEDDER_*` equivalents). Other roles are YAML-only for now.

## Debug Tracing

Off by default, zero behavior/performance impact unless enabled. When on, every step in the [Per-Turn Flow](#per-turn-flow) — topic detection, extraction, facts consolidation, topic summarization, every embedding call, and retrieval (including per-candidate similarity scores *before and after* weighting) — is recorded as a structured event to a **separate** SQLite database, not the main memory DB. Previously-silent failures (extraction/summarization/embedding/consolidation errors that would otherwise vanish into a bare `except: pass`) are recorded as `error` events instead of disappearing.

Enable it via `eidolon.yaml`:

```yaml
debug:
  enabled: true
  trace_db_path: ~/.eidolon/trace.db
```

...or `--debug --trace-db <path>` on `tools/cli.py`, or `EIDOLON_DEBUG_ENABLED=true`.

Inspect recorded traces with the terminal viewer:

```bash
# Most recent request
python tools/trace.py --db ~/.eidolon/trace.db

# Last 5 requests for one session
python tools/trace.py --db ~/.eidolon/trace.db --session <session_id> --last 5
```

Or explore interactively with the Textual-based TUI (`uv sync --extra tui` first): a drill-down list of recent turns → prefetch/sync_turn → individual steps → for context_retrieval, the full scored candidate ranking (topics/memory items with similarity scores) behind what actually got injected into the prompt.

```bash
python tools/trace_tui.py --db ~/.eidolon/trace.db
```

Both tools share the same request_id: the Hermes adapter and `tools/cli.py` both mint one id per turn and reuse it across prefetch and sync_turn, so a turn's events all group together instead of appearing as unrelated requests.

To see exactly what a past turn's prefetch() returned for prompt injection, without a live Eidolon instance:

```bash
# Most recent turn
python tools/simulate_injection.py --db ~/.eidolon/trace.db

# A specific turn
python tools/simulate_injection.py --db ~/.eidolon/trace.db --request-id <request_id>
```

It replays the turn's recorded `context_retrieval` event through `core/context_formatter.py`'s `ContextFormatter` — the same formatter `prefetch()` itself calls — so the output can never drift from what actually happened.

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Core only
uv run pytest tests/core/ -v

# Adapter only
uv run pytest tests/adapters/ -v
```

### Project Status

| Phase | Status | What |
|---|---|---|
| **Phase 1** | ✅ Complete | Core library, providers, config, CLI, Hermes adapter, debug tracing + TUI, batch facts consolidation, 297+ tests |
| **Phase 2** | 🔜 Next | Topic-Only Mode, batch fact extraction, cluster summarization |
| **Phase 3** | ⏳ TBD | Production deployment, performance optimization |

## Roadmap

See [`plan.md`](plan.md) for the detailed Phase 1 implementation plan.

### Phase 2 (Planned)

- **Topic-Only Mode** — Run alongside Honcho via pre_llm_call hook (fallback if Full Mode isn't sufficient; scaffolded in `adapters/hermes/llm_hook.py`, not yet wired to a live Hermes hook)
- **Cluster Summarization** — Group related topics, generate cluster summaries
- **Alternative Storage** — pgvector (already running in the target deployment; migration not yet started)

### Done ahead of schedule

- **Weighted Associations** — retrieval weights (`semantic`/`tag_overlap`/`person`/`temporal`) are configurable and actually wired into scoring
- **Debug Tracing** — see [Debug Tracing](#debug-tracing) above
- **TUI** — Textual-based interactive drill-down explorer over the trace database (`tools/trace_tui.py`), see [Debug Tracing](#debug-tracing)
- **Batch Facts Consolidation** — turn-interval-triggered (not cron, since there's no scheduler concept yet) periodic reconciliation of the durable-facts profile, replacing per-turn extraction — see [Facts Consolidation](#facts-consolidation)

## License

MIT — see LICENSE for details.
