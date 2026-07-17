# Eidolon — Hermes Memory Provider Plugin

> A topic memory system: human-like memory for AI agents.
> Segments conversations into topics, extracts memory fragments (not transcripts),
> and retrieves them associatively across multiple dimensions.

## Directory Structure

```
plugins/eidolon/              ← Hermes plugin root (flat, no memory/ subdir!)
├── __init__.py               # MemoryProvider implementation + register()
├── plugin.yaml               # Metadata (name, description, hooks)
├── core/                     # Eidolon core library
├── providers/                # Provider implementations (LM Studio, Ollama, etc.)
├── adapters/hermes/          # Hermes adapter implementation
├── tools/                    # CLI for standalone testing
├── eidolon.example.yaml      # Example configuration
└── README.md                 # This file
```

## Installation

### 1. Copy the project into the Hermes plugin directory

```bash
cp -r /path/to/eidolon /path/to/hermes/profiles/<profile>/plugins/eidolon
```

**Important:** The plugin root must be `plugins/eidolon/` (flat), **not** `plugins/memory/eidolon/`. Hermes discovers user plugins at `$HERMES_HOME/plugins/<name>/` — there is no `memory/` subdirectory for user plugins.

### 2. Create the configuration file

Copy `eidolon.example.yaml` to the profile root and adjust the values:

```bash
cp eidolon.example.yaml /path/to/hermes/profiles/<profile>/eidolon.yaml
```

Edit `eidolon.yaml` to configure your providers:

```yaml
providers:
  ollama:
    base_url: http://192.168.178.81:11434
    timeout: 60
  lm_studio:
    base_url: http://192.168.178.81:1234
    timeout: 30

llm:
  provider: ollama
  model: deepseek-v4-flash:cloud

embedder:
  provider: lm_studio
  model: text-embedding-bge-m3
```

### 3. Enable the plugin in Hermes config

In the profile's `config.yaml`, set:

```yaml
memory:
  provider: eidolon
```

### 4. Restart the gateway

Restart the profile's gateway (via Dashboard or `s6-svc`).

## Configuration

Eidolon loads its configuration from `eidolon.yaml` in the profile root directory
(`$HERMES_HOME/eidolon.yaml`). See `eidolon.example.yaml` for all available options.

### Key settings

| Section | Key | Description |
|---|---|---|
| `providers` | — | Map of named provider services (ollama, lm_studio, etc.) |
| `providers.<name>.base_url` | URL | Base URL of the provider service |
| `providers.<name>.timeout` | int | Request timeout in seconds |
| `llm.provider` | string | Which provider to use for LLM calls (must match a key in `providers`) |
| `llm.model` | string | Model name for chat completions |
| `embedder.provider` | string | Which provider to use for embeddings (must match a key in `providers`) |
| `embedder.model` | string | Model name for embeddings |
| `storage.path` | string | Path to the SQLite database file |
| `retrieval.semantic` | float | Embedding similarity weight (0-1) |
| `retrieval.tag_overlap` | float | Tag overlap weight (0-1) |
| `retrieval.person` | float | Same peer weight (0-1) |
| `retrieval.temporal` | float | Temporal closeness weight (0-1) |
| `language` | string | Language code for stop word filtering (`en`, `de`) |

## How It Works

### Per-turn flow

1. **`prefetch(query, session_id)`** — Called **before** each LLM call (sync).
   - Detects topic (CONTINUE / NEW_TOPIC)
   - Retrieves relevant context (facts, related topics, memory items)
   - Returns formatted text for prompt injection

2. **`sync_turn(user, assistant, session_id, messages)`** — Called **after** each turn (async).
   - Stores user + assistant messages
   - Extracts memory fragments (essence, facts, surprises, feelings)
   - Extracts structured facts from fact-type items
   - Manages topic lifecycle (close old topic, create new topic, generate summary + tags)

### Memory layers

| Layer | Description | Persistence |
|---|---|---|
| **Facts** | Structured, (near-)constant info about peers (identity, preferences) | Persistent across sessions |
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
| Structured Fact Extraction | ~100 | Async (after LLM) |
| Topic Summary + Tags | ~200 | Async (on topic close) |

**Total per turn:** ~250-350 tokens + 2 embedding calls.

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

# With real providers:
python tools/cli.py \
  --llm-provider ollama --llm-model deepseek-v4-flash:cloud \
  --embedder-provider lm_studio --embedder-model text-embedding-bge-m3

# With a YAML config file:
python tools/cli.py --config ~/.eidolon/config.yaml
```

## Known Issues

- The plugin must be **copied** (not symlinked) into the Hermes plugins directory,
  because Hermes needs the `core/` and `providers/` packages on `sys.path`.
  A symlink makes the files visible but the imports won't resolve.
- The `eidolon.yaml` config file must exist in the profile root before the
  gateway starts, otherwise `initialize()` raises a `RuntimeError`.
