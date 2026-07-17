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
│   └── stores/               #   SQLite store with auto-migration
├── adapters/                 # Platform-specific adapters
│   └── hermes/               #   Hermes MemoryProvider plugin
├── providers/                # Provider implementations (one class per service)
│   ├── lm_studio.py          #   LM Studio: complete() + embed()
│   ├── ollama.py             #   Ollama: complete() + embed() (OpenAI-compatible)
│   └── sentence_transformers.py  #   Local PyTorch embeddings (fallback)
├── tools/                    # Standalone tools
│   └── cli.py                #   CLI for testing without Hermes
├── tests/                    # Test suite
│   ├── core/                 #   Core tests
│   └── adapters/hermes/      #   Adapter tests
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
| **Facts** | Structured, (near-)constant info about peers (identity, preferences) | Persistent across sessions |
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
│ 6. Structured Fact Extraction (LLM) │
│ 7. Topic Summary + Tags (LLM)       │
│ 8. Embedding storage                │
└─────────────────────────────────────┘
```

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

# With real providers:
python tools/cli.py \
  --llm-provider ollama --llm-model deepseek-v4-flash:cloud \
  --embedder-provider lm_studio --embedder-model text-embedding-bge-m3

# With a YAML config file:
python tools/cli.py --config ~/.eidolon/config.yaml
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

```yaml
providers:
  ollama:
    base_url: http://192.168.178.81:11434
  lm_studio:
    base_url: http://192.168.178.81:1234

llm:
  provider: ollama
  model: deepseek-v4-flash:cloud

embedder:
  provider: lm_studio
  model: text-embedding-bge-m3
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

### Project Status

| Phase | Status | What |
|---|---|---|
| **Phase 1** | ✅ Complete | Core library, providers, config, CLI, Hermes adapter, 118+ tests |
| **Phase 2** | 🔜 Next | Topic-Only Mode, batch fact extraction, cluster summarization, TUI |
| **Phase 3** | ⏳ TBD | Production deployment, performance optimization |

## Roadmap

See [`plan.md`](plan.md) for the detailed Phase 1 implementation plan.

### Phase 2 (Planned)

- **Topic-Only Mode** — Run alongside Honcho via pre_llm_call hook
- **Batch Fact Extraction (Cron)** — Collect fact-type items over multiple turns, extract in a single batch LLM call (cost optimization)
- **Cluster Summarization** — Group related topics, generate cluster summaries
- **Alternative Storage** — pgvector, MongoDB
- **Weighted Associations** — Tune retrieval weights per use case
- **TUI** — Textual-based debug/analysis interface (experimental)

## License

MIT — see LICENSE for details.
