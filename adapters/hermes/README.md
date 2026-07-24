# Eidolon — Hermes Memory Provider Plugin

> A topic memory system: human-like memory for AI agents.
> Segments conversations into topics, extracts memory fragments (not transcripts),
> and retrieves them associatively across multiple dimensions.

## Directory Structure

```
plugins/eidolon/              ← Hermes plugin root (flat, no memory/ subdir!)
├── __init__.py               # MemoryProvider implementation + register()
├── plugin.yaml               # Metadata (name, description, hooks)
├── core/                     # Eidolon core library (incl. core/providers/: LM Studio, Ollama, etc.)
├── adapters/hermes/          # Hermes adapter implementation
├── tools/                    # CLI for standalone testing
├── eidolon.example.yaml      # Example configuration
└── README.md                 # This file
```

## Installation

### Recommended: `install-plugin.sh`

From the Eidolon project root:

```bash
HERMES_HOME=/path/to/hermes PROFILE=<profile> ./tools/install-plugin.sh
```

This rsyncs the project into `$HERMES_HOME/profiles/<profile>/plugins/eidolon` — excluding dev-only files, and never touching the plugin's own `data/` directory (a plain `cp -r` here can clobber a live `eidolon_memory.db`) — scaffolds `eidolon.yaml` from `eidolon.example.yaml` if one doesn't exist yet, and checks whether `memory.provider: eidolon` is set in the profile's `config.yaml`. Safe to re-run any time to redeploy code changes.

After it finishes: edit `eidolon.yaml` (see [Configuration](#configuration) below), make sure `memory.provider: eidolon` is set in the profile's `config.yaml`, then restart the gateway (Dashboard → Restart, or `s6-svc`).

### Manual installation

If you'd rather not use the script, or want to understand what it does:

1. **Copy the project** into the Hermes plugin directory. The plugin root must be `plugins/eidolon/` (flat), **not** `plugins/memory/eidolon/`. Hermes discovers user plugins at `$HERMES_HOME/plugins/<name>/` — there is no `memory/` subdirectory for user plugins.

   ```bash
   cp -r /path/to/eidolon /path/to/hermes/profiles/<profile>/plugins/eidolon
   ```

2. **Create the configuration file** — copy `eidolon.example.yaml` to the profile root and adjust the values:

   ```bash
   cp eidolon.example.yaml /path/to/hermes/profiles/<profile>/eidolon.yaml
   ```

   Edit `eidolon.yaml` to configure your models (see [Configuration](#configuration) below for the full shape):

   ```yaml
   models:
     llm:
       provider: ollama                # openai_compatible | ollama | lm_studio | openai | openrouter | simulated
       base_url: http://192.168.1.100:11434
       model: deepseek-v4-flash:cloud

     embedder:
       provider: lm_studio             # a local preset of openai_compatible
       base_url: http://192.168.1.100:1234
       model: text-embedding-bge-m3
   ```

3. **Enable the plugin** in the profile's `config.yaml`:

   ```yaml
   memory:
     provider: eidolon
   ```

4. **Restart the gateway** (via Dashboard or `s6-svc`).

## Configuration

Eidolon loads its configuration from `eidolon.yaml` in the profile root directory
(`$HERMES_HOME/profiles/<profile>/eidolon.yaml`). See `eidolon.example.yaml` for all available options.

### Key settings

Each model role (`llm`, `embedder`, and any future role) is a **self-contained** endpoint — provider, `base_url`, and `model` together, no shared registry. A role that's actually used but not fully specified fails immediately with a clear error at startup instead of silently guessing an endpoint.

| Section | Key | Description |
|---|---|---|
| `models.llm.provider` | string | `openai_compatible` \| `ollama` \| `lm_studio` \| `openai` \| `openrouter` \| `simulated` |
| `models.llm.base_url` | URL | Base URL of the LLM endpoint |
| `models.llm.model` | string | Model name for chat completions |
| `models.llm.api_key` | string | Bearer token for authenticated endpoints (optional) |
| `models.embedder.provider` | string | `openai_compatible` \| `ollama` \| `lm_studio` \| `openai` \| `sentence_transformers` \| `simulated` |
| `models.embedder.base_url` | URL | Base URL of the embedder endpoint |
| `models.embedder.model` | string | Model name for embeddings |
| `storage.provider` | string | `sqlite` (default, zero setup) \| `postgres` (recommended for production — see below) |
| `storage.path` | string | Path to the SQLite database file (`provider: sqlite` only) |
| `storage.host`/`port`/`dbname`/`user`/`password` | string/int | Postgres connection (`provider: postgres` only); `password` also settable via `EIDOLON_STORAGE_PASSWORD` |
| `deployment.mode` | string | `embedded` (default, in-process, uses the sections above) \| `server` (thin HTTP client instead — see below) |
| `deployment.server_url`/`token`/`timeout` | string/float | Remote server connection (`mode: server` only); `token` also settable via `EIDOLON_DEPLOYMENT_TOKEN` |
| `retrieval.semantic` | float | Embedding similarity weight (0-1) |
| `retrieval.tag_overlap` | float | Tag overlap weight (0-1) |
| `retrieval.person` | float | Same peer weight (0-1) |
| `retrieval.temporal` | float | Temporal closeness weight (0-1) |
| `retrieval.related_topics_new` | int | Related topics injected on a NEW_TOPIC turn (default 6) |
| `retrieval.related_topics_continue` | int | Related topics injected on a CONTINUE turn (default 3) |
| `retrieval.memory_items_new` | int | Memory items injected on a NEW_TOPIC turn (default 4) |
| `retrieval.memory_items_continue` | int | Memory items injected on a CONTINUE turn (default 2) |
| `facts.consolidation_interval` | int | Reconcile the durable-facts profile every N user turns (default 20) |
| `facts.target_max` | int | Soft guidance to the LLM for the target profile size (default 100), not hard-enforced |
| `facts.categories` | list of strings | Closed category list offered to the LLM — soft-enforced, see the main [README's Facts Consolidation section](../../README.md#facts-consolidation) |
| `maintenance.enabled` | bool | Auto-close stale-open topics (default `false`) — see [Stale-topic maintenance](#stale-topic-maintenance) below |
| `maintenance.interval_hours` | float | Minimum time between sweep attempts (default 1) |
| `maintenance.stale_after_hours` | float | How long a topic must be idle to count as stale (default 24) |
| `maintenance.peer_id` | string | Restrict the sweep to one peer (default: every peer) |
| `language` | string | Language code for stop word filtering (`en`, `de`) |
| `timezone` | string | IANA zone name (e.g. `Europe/Berlin`) for displaying the injected `[Environment]` time (default `UTC`); stored timestamps stay UTC internally |

### Deployment mode

Two presets, picked via `deployment.mode` — full rationale in the main
[README's Deployment Modes section](../../README.md#deployment-modes).
Here's specifically what each means for *this plugin's* `eidolon.yaml`:

- **`embedded` (default)** — `eidolon.yaml` needs the full config shown
  in [Installation](#installation) above (`models:`/`storage:`/
  `retrieval:`/`facts:`/...). Core runs in-process inside the adapter.
- **`server`** — `eidolon.yaml` needs *only* the `deployment:` block:

  ```yaml
  deployment:
    mode: server
    server_url: http://<server-host>:8420
    token: ...          # or EIDOLON_DEPLOYMENT_TOKEN
    timeout: 30
  ```

  Everything else (`models:`/`storage:`/`retrieval:`/`facts:`/...)
  belongs to the separately-run server's own `eidolon.yaml` instead
  (see [`deploy/`](../../deploy/)) — the adapter no longer reads or
  needs those sections at all. This is also the permanent fix for the
  Postgres/psycopg fragility issue below.

### Stale-topic maintenance

Hermes has no session-end hook, so a topic a peer just abandons
mid-conversation stays `active` in storage forever unless something
closes it — see ROADMAP.md's "Topics never close on their own" entry.
Off by default (`maintenance.enabled: false`); when enabled, the
adapter piggybacks a cleanup sweep on real chat activity — checked no
more than once every `maintenance.interval_hours`, closing anything
idle past `maintenance.stale_after_hours` — rather than running on a
fixed timer independent of activity. That distinction matters: closing
a topic while its owner is still away risks fragmenting a conversation
they'd have naturally continued (topic detection has nothing "active"
left to compare against once a topic is closed, so a returning peer's
continuation gets forced into a new topic instead of resuming the old
one). Piggybacking on activity means a topic only ever closes once
some turn — anyone's, not necessarily its owner's — proves enough idle
time has actually passed.

### Postgres storage — deployed-environment dependency

`install-plugin.sh` only rsyncs Eidolon's own code — it does **not** install
Python packages into the Hermes container's environment. `provider: sqlite`
needs nothing extra (stdlib `sqlite3`), but `provider: postgres` needs
`psycopg`/`pgvector` importable from wherever the plugin actually runs,
which is a separate Python environment from your dev checkout's `.venv`.
Confirmed missing by default in a Docker-based Hermes deployment; installed
into the container's own venv with:

```bash
docker exec <container> uv pip install --python <hermes-venv>/bin/python3 "psycopg[binary]>=3.1" "pgvector>=0.3"
```

(substitute `pip install` directly if `uv` isn't available in the
container). **This does not survive a container recreation** — only an
in-place restart of the gateway process. Recreating the container (e.g.
after changing `docker-compose.yml`, such as adding
`EIDOLON_STORAGE_PASSWORD`) wipes the ephemeral install and it needs
redoing. Two permanent fixes, depending on which deployment mode this
adapter runs in:

- **Staying in `embedded` mode**: bake `psycopg`/`pgvector` into a
  custom image layer over the upstream Hermes image (`RUN uv pip
  install ... "psycopg[binary]>=3.1" "pgvector>=0.3"` in a `Dockerfile`,
  built once, reused across recreations) instead of installing them by
  hand after every recreation.
- **Switching to `deployment.mode: server`** (see the main
  [README's Deployment Modes section](../../README.md#deployment-modes)):
  this problem disappears entirely, since the adapter no longer imports
  `core` at all in this mode — `psycopg`/`pgvector` only need to be
  installed in the separately-run server process (see
  [`deploy/`](../../deploy/)), never in the Hermes container's venv.

## How It Works

### Per-turn flow

1. **`prefetch(query, session_id)`** — Called **before** each LLM call (sync).
   - Detects topic (CONTINUE / NEW_TOPIC)
   - Retrieves relevant context (facts, related topics, memory items)
   - Returns formatted text for prompt injection, prefixed with a short `[Memory Context]` preamble explaining how to read it — see the main [README's Per-Turn Flow section](../../README.md#per-turn-flow)

2. **`sync_turn(user, assistant, session_id, messages)`** — Called **after** each turn (async).
   - Stores user + assistant messages
   - Extracts memory fragments (essence, facts, surprises, feelings) — fact-type items are stored and embedded like any other memory item
   - Manages topic lifecycle (close old topic, create new topic, generate summary + tags)
   - Every `facts.consolidation_interval` turns, reconciles the durable-facts profile in one batch LLM call against everything fact-shaped learned since the last pass — see the main [README's Facts Consolidation section](../../README.md#facts-consolidation)

### Memory layers

| Layer | Description | Persistence |
|---|---|---|
| **Facts** | A small, curated profile of durable identity-level facts about a peer (name, location, relationships, health, work, stable preferences) — reconciled periodically, not per-turn | Persistent across sessions |
| **Topics** | Episodic conversation segments with summaries, tags, embeddings | Persistent across sessions |
| **Memory Items** | Extracted fragments (essence, fact, surprise, feeling) with perspective attribution | Persistent across sessions |
| **Messages** | Raw conversation turns, linked to topics and memory items | Persistent across sessions |

### LLM calls per turn

| Step | Tokens | Timing |
|---|---|---|
| Topic Detection | ~50 | Sync (before LLM) |
| Embedding | — | Sync (local) |
| Associative Retrieval | — | Sync (SQL + vector search) |
| Fragmentary Extraction | ~100 | Async (after LLM) |
| Topic Summary + Tags | ~200 | Async (on topic close) |
| Facts Consolidation | ~200-2000 | Async, only every `facts.consolidation_interval` turns |

**Total per turn:** ~150-250 tokens + 2 embedding calls, plus one larger consolidation call every N turns.

### Tool calls

`get_tool_schemas()` exposes two tools the LLM can call explicitly:

| Tool | Effect |
|---|---|
| `eidolon_search` | Read-only associative retrieval for an explicit query. |
| `eidolon_correct_fact` | Upserts or deletes one fact immediately, bypassing consolidation. Use when the user says a remembered fact is wrong or should be forgotten. Omit `value` to delete. See the main [README's Tool Calls section](../../README.md#tool-calls). |

## Development

### Running tests

```bash
cd /path/to/eidolon
uv run pytest tests/ -v
```

### Running the CLI (standalone, without Hermes)

```bash
# With simulated providers (no external dependencies):
python tools/cli.py

# With real providers — provider, base_url, AND model are all required;
# an incomplete role fails fast with a clear error instead of guessing
# an endpoint (see Configuration above):
python tools/cli.py \
  --llm-provider ollama --llm-base-url http://localhost:11434 --llm-model deepseek-v4-flash:cloud \
  --embedder-provider lm_studio --embedder-base-url http://localhost:1234 --embedder-model text-embedding-bge-m3

# With a YAML config file:
python tools/cli.py --config ~/.eidolon/config.yaml
```

## Known Issues

- The plugin must be **copied** (not symlinked) into the Hermes plugins directory,
  because Hermes needs the `core/` and `providers/` packages on `sys.path`.
  A symlink makes the files visible but the imports won't resolve.
- The `eidolon.yaml` config file must exist in the profile root before the
  gateway starts, otherwise `initialize()` raises a `RuntimeError`.
