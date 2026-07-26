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
│   ├── consolidation/        #   Fact Consolidator (durable-facts profile) + Cluster Consolidator (recurring-topic threads)
│   ├── stores/               #   SQLite store (default) + Postgres/pgvector store (needs `postgres` extra)
│   │   ├── vec_index.py      #     SQLite-only: optional sqlite-vec KNN index, auto-fallback to brute-force scan
│   │   ├── postgres_store.py #     PostgresStore — native pgvector VECTOR/HNSW, no fallback needed
│   │   └── postgres_schema.sql  #  Consolidated schema (VECTOR/JSONB/TIMESTAMPTZ vs SQLite's BLOB/TEXT/naive)
│   ├── trace/                #   Debug tracer (NullTracer / SqliteTracer), off by default
│   ├── providers/            #   Provider implementations (one class per service)
│   │   ├── openai_compatible.py  # Any OpenAI-API endpoint: complete() + embed()
│   │   ├── ollama.py         #     Preset of openai_compatible (local Ollama defaults)
│   │   ├── lm_studio.py      #     Preset of openai_compatible (local LM Studio defaults)
│   │   ├── openai.py         #     Preset of openai_compatible (OpenAI's hosted API)
│   │   ├── openrouter.py     #     Preset of openai_compatible (OpenRouter, LLM only)
│   │   ├── sentence_transformers.py  # Local PyTorch embeddings (fallback)
│   │   └── simulated.py      #     Deterministic fake provider for tests/CLI dry-runs
│   ├── context_formatter.py  #   Renders a ContextBundle into the prompt-injection text
│   └── eidolon_client.py     #   EidolonClient Protocol — the contract adapters code against
├── server/                   # Core-as-a-service HTTP server (the "server" deployment mode)
│   ├── app.py                #   FastAPI app — one route per EidolonClient method
│   └── auth.py               #   Bearer-token auth dependency
├── client/                   # HTTP implementation of EidolonClient (talks to server/)
│   ├── http_client.py        #   EidolonHttpClient — uses `requests`, already a base dependency
│   └── wire.py               #   Stdlib-only (de)serialization — deliberately no pydantic
├── deploy/                   # Docker packaging for the "server + pgvector" preset
│   ├── server/Dockerfile     #   Image running tools/serve.py
│   ├── docker-compose.example.yml  # Bundles the server with a pgvector/pgvector:pg15 container
│   └── init.sql              #   Bootstraps the `vector` extension
├── adapters/                 # Platform-specific adapters
│   └── hermes/               #   Hermes MemoryProvider plugin
├── tools/                    # Standalone tools
│   ├── cli.py                #   CLI for testing without Hermes (embedded mode only)
│   ├── serve.py              #   Entry point for the core-as-a-service HTTP server
│   ├── trace.py              #   Terminal viewer for debug trace events
│   ├── trace_tui.py          #   Interactive drill-down TUI over trace.db (needs `tui` extra)
│   ├── tui/                  #   Textual screens for trace_tui.py
│   ├── simulate_injection.py #   Replays a turn's prefetch() injection string from trace.db
│   ├── run_eval.py           #   Eval harness CLI entry point
│   ├── eval/                 #   Eval harness: YAML scripts run against real LLMs, tier-1/tier-2 scoring
│   │   ├── scripts/          #     Golden eval scripts (YAML)
│   │   └── reports/          #     Retained run reports (markdown + JSON), not deleted after each run
│   ├── install-plugin.sh     #   Deploys the Hermes adapter into a live Hermes install
│   ├── migrate_sqlite_to_postgres.py  # One-time data migration between storage backends
│   └── reset-db.sh           #   Wipes the local dev memory DB (SQLite only)
├── tests/                    # Test suite
│   ├── conftest.py           #   store/store_backend + make_eidolon_client fixtures
│   ├── core/                 #   Core tests (incl. core/trace/, core/stores/)
│   ├── adapters/hermes/      #   Adapter tests
│   └── tools/                #   Trace viewer + TUI tests
├── eidolon.example.yaml      # Example configuration
├── AGENTS.md                 # Coding conventions
├── ROADMAP.md                # Project roadmap and status
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
│    + embedding per extracted item   │
│ 6. Mid-conversation Relabel (LLM)   │  ← once per topic, threshold-gated
│ 7. Topic Summary + Tags (LLM)       │  ← on topic close
│    + re-embedding from summary      │
│ 8. Recurring-topic Clustering (LLM) │  ← on topic close, if it matches
│    + cluster summary embedding      │    a prior recurring thread
│ 9. Facts Consolidation (LLM)        │  ← every N turns, not every turn
└─────────────────────────────────────┘  see Facts Consolidation below
```

Every step above always records an **operational-metrics row** — duration and success/failure, typed columns in the main database — regardless of debug settings (see [Operational Metrics](#operational-metrics) below). With [Debug Tracing](#debug-tracing) additionally enabled, the same steps also get a much richer trace event (full prompts, responses, retrieval score breakdowns) — off by default, zero extra behavior/performance impact unless enabled.

Whenever there's anything to inject, `ContextFormatter` prepends a short `[Memory Context]` preamble explaining how to read the sections that follow (recalled background, not something the user just said) — a static, always-on hint aimed at reducing the LLM misinterpreting stale facts or old topic summaries as part of the live conversation. How many related topics and memory items get injected per turn is configurable — see `retrieval.*_new`/`retrieval.*_continue` under [Configuration](#configuration).

## Facts Consolidation

The facts profile is intentionally small (~50-100 entries) and curated, not a growing log — every `item_type="fact"` memory item is still stored and embedded as usual (searchable via associative retrieval, same as any other memory item), but converting a fact-shaped statement into a permanent profile row happens in a periodic **batch reconciliation pass** instead of immediately on every turn.

Every `facts.consolidation_interval` user turns (default 20, see [Configuration](#configuration)), `FactConsolidator` ([`core/consolidation/fact_consolidator.py`](core/consolidation/fact_consolidator.py)) sends the LLM the *entire current profile* — each entry tagged with who contributed it (`source`: `ai`-inferred vs. explicitly `user`-confirmed) and its `importance` — plus everything fact-shaped learned since the last pass, and asks for a diff — only what changed:

```json
{"upserts": [{"category": "work", "key": "employer", "value": "Acme Corp", "importance": "high"}], "deletes": [{"category": "routine", "key": "old_gym"}]}
```

Seeing the full existing profile each time lets the LLM reuse an existing `(category, key)` when the same concept is being confirmed or corrected — updating a job title updates the existing `work: employer` entry in place instead of adding a duplicate — while omitted entries are left untouched. `category` is drawn from a closed, configurable list (`facts.categories`, see [Configuration](#configuration)) — soft-enforced, so an off-list category is logged rather than dropped. The prompt explicitly steers multi-valued categories (`relationship`, `technology`, `preference`, `possession`) toward one key per distinct fact instead of accumulating clauses into a single key. `importance` (`low`/`medium`/`high`, set by the LLM per fact) determines injection order — highest first — and gives the "trim toward `target_max`" instruction a concrete signal instead of an unguided guess. A checkpoint (the highest `memory_items.id` already incorporated) ensures a failed pass gets retried at the next trigger rather than silently dropping those facts.

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

# Maintenance: gracefully close topics left 'active' too long (a crashed
# session, an adapter that stopped calling store_turn) — lists candidates,
# closes them via the same summarize-then-close path as a normal topic
# transition, then exits without entering the REPL. --dry-run lists only.
python tools/cli.py --close-stale-topics --stale-older-than-hours 12 --dry-run
python tools/cli.py --close-stale-topics --stale-older-than-hours 12 --stale-peer-id user
```

### Hermes Adapter

See [`adapters/hermes/README.md`](adapters/hermes/README.md) for installation and configuration instructions.

## Deployment Modes

Two presets, picked via `deployment.mode` in `eidolon.yaml`:

**`embedded` (default) — try it in five minutes.** Core runs in-process inside the adapter — nothing extra to start, works with `sqlite` storage out of the box. This is what the Quick Start and CLI sections above use; no `deployment:` section needed at all.

**`server` — running this for real.** Core runs as its own standalone HTTP process ([`tools/serve.py`](tools/serve.py)), and the adapter becomes a thin HTTP client ([`client/http_client.py`](client/http_client.py)) instead of importing core directly. Decouples a core code change or restart from the adapter's own process, and avoids installing core's dependencies (e.g. `psycopg` for Postgres) into a platform adapter's own environment. Pairs naturally with `postgres` storage — [`deploy/docker-compose.example.yml`](deploy/docker-compose.example.yml) bundles both in one stack:

```bash
cp eidolon.example.yaml deploy/eidolon.yaml   # then edit models: for your LLM/embedder
EIDOLON_DEPLOYMENT_TOKEN=... EIDOLON_STORAGE_PASSWORD=... \
  docker compose -f deploy/docker-compose.example.yml up -d
```

Already have a Postgres instance running (e.g. migrating an existing `embedded` + `postgres` deployment to `server` mode, rather than starting fresh)? [`deploy/docker-compose.external-db.example.yml`](deploy/docker-compose.external-db.example.yml) runs just the server, pointed at your existing instance via `EIDOLON_STORAGE_*` env vars, instead of bundling a new Postgres container.

The adapter's own `eidolon.yaml` then only needs the `deployment:` block — everything else (models/storage/facts/...) is the separately-run server's own config, not the adapter's:

```yaml
deployment:
  mode: server
  server_url: http://<server-host>:8420
  token: ...          # or EIDOLON_DEPLOYMENT_TOKEN — must match the server's own token
  timeout: 30
```

`tools/cli.py` stays `embedded`-only (its `/items`/`/score` debug commands read storage directly, which a remote server doesn't expose) — run it against a local config for testing regardless of what your Hermes deployment uses. Any other combination (e.g. `embedded` + `postgres`, already the case if you just set `storage.provider: postgres` without touching `deployment:`) works too; these two are simply what's documented as the headline path, to keep the decision simple for a first deployment.

## Configuration

Eidolon uses a YAML-based configuration system with three sources (highest wins):

1. **CLI arguments** (e.g. `--llm-provider ollama`)
2. **Environment variables** (e.g. `EIDOLON_LLM_PROVIDER=ollama`)
3. **YAML config file** (e.g. `~/.eidolon/config.yaml`)

See [`eidolon.example.yaml`](eidolon.example.yaml) for all available options.

### Key Settings

Each model role (`llm`, `embedder`, and any future role like `judge`) is a **self-contained** endpoint — provider, `base_url`, and `model` together, no shared registry, no cross-references by name. A role that's actually used but not fully specified (missing `base_url`, for instance) fails immediately with a clear error at startup rather than silently falling back to a guessed endpoint — this used to silently misroute requests to whatever happened to be listening on a default port.

**Storage.** `sqlite` (default) needs zero setup and is what the CLI/tests use. `postgres` — recommended for a real deployment — needs `pip install eidolon[postgres]` and a reachable Postgres with the `vector` extension available; discrete `host`/`port`/`dbname`/`user`/`password` fields, same "self-contained, explicit fields" philosophy as models (`password` also settable via `EIDOLON_STORAGE_PASSWORD`, kept out of the YAML file). pgvector's `VECTOR(n)` column width is fixed from `models.embedder.dimension` the first time the schema is created — changing the embedder's dimension later needs a real migration, not just a config edit. Moving existing data between backends: [`tools/migrate_sqlite_to_postgres.py`](tools/migrate_sqlite_to_postgres.py).

```yaml
storage:
  provider: postgres    # or: sqlite
  host: 127.0.0.1
  port: 5432
  dbname: eidolon
  user: eidolon
  password: ...         # or EIDOLON_STORAGE_PASSWORD
```

Why a second storage backend at all: SQLite's concurrency model has every client process take OS-level advisory file locks directly on the `.db` file, which doesn't reliably work over network/bridged filesystems (Docker Desktop's virtiofs bind mounts on macOS, NFS, SMB) — Postgres's single-writer-process-plus-wire-protocol model doesn't hit that failure class at all. See [`core/stores/postgres_store.py`](core/stores/postgres_store.py)'s module docstring for the full story.

**Providers.** `openai_compatible` targets any endpoint speaking the OpenAI HTTP API — hosted (Groq, Together, DeepInfra), self-hosted (vLLM, LiteLLM, a private gateway), or local (Ollama 0.5+, LM Studio). Set `api_key` for authenticated endpoints (sent as a Bearer token; also settable via `EIDOLON_LLM_API_KEY`/`EIDOLON_EMBEDDER_API_KEY`). `ollama`, `lm_studio`, `openai`, and `openrouter` are thin presets of it with sensible defaults baked in — `openai` targets OpenAI's own hosted API (LLM + embedder), `openrouter` targets OpenRouter (LLM only; it has no embeddings endpoint). `sentence_transformers` provides local PyTorch embeddings; `simulated` is a deterministic, network-free fake for tests and CLI dry-runs.

```yaml
models:
  llm:
    provider: openai          # or: openai_compatible | ollama | lm_studio | openrouter | simulated
    api_key: sk-...           # only for authenticated endpoints

  embedder:
    provider: lm_studio       # a local preset of openai_compatible
    base_url: http://192.168.1.100:1234
    model: text-embedding-bge-m3

retrieval:
  related_topics_new: 6       # Related topics injected on a NEW_TOPIC turn (default: 6)
  related_topics_continue: 3  # Related topics injected on a CONTINUE turn (default: 3)
  memory_items_new: 4         # Memory items injected on a NEW_TOPIC turn (default: 4)
  memory_items_continue: 2    # Memory items injected on a CONTINUE turn (default: 2)
  recent_topics: 5            # Most-recently-closed topics injected regardless of relevance,
                              # excluding duplicates of related_topics_* above (default: 5) — never
                              # filtered by min_topic_messages below, by design
  min_topic_messages: 3       # Soft-hide topics at/below this message count from related-topic
                              # search only, not recent_topics — "mini-topics" like one-off asides
                              # (default: 3)

facts:
  consolidation_interval: 20  # See Facts Consolidation above
  target_max: 100

debug:
  enabled: false              # See Debug Tracing below
  trace_db_path: ~/.eidolon/trace.db

timezone: Europe/Berlin       # IANA zone for displaying injected time (default: UTC)
```

`llm` and `embedder` have dedicated CLI flags/env vars for quick overrides without touching YAML: `--llm-provider`/`--llm-base-url`/`--llm-model`/`--llm-max-tokens`/`--llm-timeout` (and the `embedder-`/`EIDOLON_LLM_*`/`EIDOLON_EMBEDDER_*` equivalents). Other roles are YAML-only for now.

`timezone` only affects how the `[Environment]` time is *displayed* to the LLM — stored timestamps (facts, topics, memory items) stay UTC internally, so relative-age suffixes like "3 weeks ago" are unaffected. Defaults to `UTC` for backward compatibility; set it to your own IANA zone (e.g. `Europe/Berlin`) so the LLM sees an unambiguous, correctly-labeled local time instead of an unlabeled UTC one.

## Operational Metrics

Always on, no config flag — every [Per-Turn Flow](#per-turn-flow) step writes a small, typed row to the main database the moment it runs, regardless of `debug.enabled`. Three tables, in the same database as facts/topics/memory items (SQLite: same file; Postgres: same instance):

| Table | One row per | Key columns |
|---|---|---|
| `events` | An error or warning from any step | `severity`, `step`, `message`, plus `peer_id`/`request_id`/`session_id` |
| `llm_calls` | An LLM completion call | `call_type` (`topic_detection`, `extraction`, `topic_summarization`, `topic_relabel`, `fact_consolidation`, `cluster_consolidation`), `duration_ms`, `succeeded` |
| `embedding_calls` | An embedding call | `purpose` (`retrieval_query`, `memory_item`, `topic_summary`, `cluster_summary`), `duration_ms`, `dimension`, `succeeded`, `entity_type`/`entity_id` |

This is deliberately the lightweight, always-on counterpart to [Debug Tracing](#debug-tracing) below — plain typed columns (directly queryable with `AVG(duration_ms)`, `COUNT(*) WHERE succeeded = false`, no JSON parsing needed), not full prompts/responses. `MetricsRecorder` (`core/interfaces.py`, implementations in `core/metrics/`) is a separate collaborator from `Tracer`, wired into every instrumented class the same way `tracer` already is, and — unlike `Tracer` — `ProviderFactory.create_metrics_recorder()` never returns a no-op: it always builds a real backend, wrapped in `SafeMetricsRecorder` so a broken metrics backend can never take down real extraction/storage.

Retrieval (`context_retrieval`, the per-candidate similarity scoring) is the one step with no row here — it carries full candidate/score payloads, not just a duration, so it stays exclusively in the richer, debug-gated trace below. Its query-embedding cost is still visible via `embedding_calls` (`purpose = 'retrieval_query'`).

## Debug Tracing

Off by default, zero behavior/performance impact unless enabled. When on, every step in the [Per-Turn Flow](#per-turn-flow) — topic detection, extraction, facts consolidation, topic summarization, every embedding call, and retrieval (including per-candidate similarity scores *before and after* weighting) — is recorded as a rich structured event (full prompts/responses, not just duration — see [Operational Metrics](#operational-metrics) above for the always-on counterpart) to a trace store **separate from the main memory tables**: `storage.provider: sqlite` gets a separate SQLite file (`debug.trace_db_path`); `storage.provider: postgres` gets a `trace_events` table in that same Postgres instance instead (`PostgresTracer` — avoids the reliability problems of a SQLite file on a mounted/bind-mounted volume, the same reasoning `PostgresStore` itself exists for). Previously-silent failures (extraction/summarization/embedding/consolidation errors that would otherwise vanish into a bare `except: pass`) are recorded as `error` events instead of disappearing.

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

## Turn Detail Dashboard

A small read-only web page (server-rendered Jinja2 + [htmx](https://htmx.org), vendored locally — no build step, no CDN dependency) that shows *everything* that happened during one turn, for the "why didn't this work as expected" case: which topic was active before/after, the user message and AI response, then a step-by-step account of what fired — topic detection, extraction, every embedding call, retrieval + scoring, and, conditionally, whatever the turn triggered (fact consolidation, topic close/summarize, relabel, cluster consolidation).

It works at two levels, degrading gracefully rather than requiring debug mode:

- **Skeleton** (always available): real message text and topic linkage (`messages.request_id`, always-on — see [Operational Metrics](#operational-metrics)), plus which steps ran with their timing/success from `events`/`llm_calls`/`embedding_calls`. No `debug.enabled` required.
- **Rich** (only for turns that ran while `debug.enabled: true`): the full [Debug Tracing](#debug-tracing) payload per step — prompts, responses, retrieval score breakdowns — shown instead of the skeleton whenever it exists for that turn.

`request_id` is the single correlation key across both tiers and every table — minted once per turn (`core/__init__.py`'s `store_turn`) and threaded into everything it triggers, so a step's mere presence or absence for that id already answers "did this happen."

Part of `eidolon-server` (see [Deployment Modes](#deployment-modes)) rather than a separate process or container: it's mounted under `/dashboard` on the same `tools/serve.py`, sharing that one process's `Eidolon`/storage instance instead of opening a second, independent connection to the database. That's deliberate — the moment write actions land here (correcting a fact, closing a topic on demand), they need to go through core's real orchestration (`TopicSummarizer`, `ClusterConsolidator`, ...), not a raw database write, and the only way to guarantee that without a second live writer racing the first is to have exactly one process ever touch the database through core. Read-only for now, `server`-deployment-mode only (there's no separately-addressable core process to mount it on in `embedded` mode).

Reachable at `http://<server-host>:8420/dashboard/` once `eidolon-server` is running — nothing extra to start. Gated by its own optional HTTP Basic auth, independent of the RPC API's bearer token (skipped entirely if unset — same "binds to localhost by default" threat model `tools/serve.py` already uses for its own default host):

```bash
EIDOLON_DASHBOARD_PASSWORD=... uv run python tools/serve.py --config eidolon.yaml
```

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

The storage-provider contract suite ([`tests/core/test_storage_provider_contract.py`](tests/core/test_storage_provider_contract.py)) runs against both SQLite and a real Postgres — `tests/conftest.py` spins up an ephemeral `pgvector/pgvector:pg15` container via `testcontainers` (session-scoped, truncated between tests) — so the full suite needs Docker running. `dev` extra installs `testcontainers[postgres]`/`psycopg`/`pgvector` for this; no separate setup needed beyond having Docker available.

### Linting & Type Checking

```bash
uv run ruff check .     # lint (E/W/F/I/UP/B/C4/SIM), 120-col
uv run ruff check . --fix
uv run mypy             # type-check core, adapters, tools
```

Both are configured in `pyproject.toml` and installed via the `dev` extra
(`uv sync --extra dev`).

### Project Status

| Phase | Status | What |
|---|---|---|
| **Phase 1** | ✅ Complete | Core library, providers, config, CLI, Hermes adapter, debug tracing + TUI, batch facts consolidation, 419+ tests |
| **Phase 2** | 🚧 In progress | Quality/eval infrastructure ✅, recurring-topic clustering ✅, storage acceleration ✅, Postgres/pgvector storage backend ✅, core-as-a-service deployment mode ✅, topic detection & summarization review, fact consolidation hardening — see [`ROADMAP.md`](ROADMAP.md); 744+ tests |
| **Phase 3** | ⏳ TBD | Topic-Only Mode, production deployment, performance optimization |

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for current phase status, what's shipped,
and what's planned next.

## License

MIT — see LICENSE for details.
