# Roadmap

## Phase 1 — Complete

Core library, provider abstractions (LLM/embedder/storage), YAML-based
config, standalone CLI, the Hermes adapter (`EidolonMemoryProvider`),
debug tracing with a terminal viewer and a Textual TUI, and batch facts
consolidation. 400+ tests.

## Phase 2 — In progress

- ✅ **Quality/eval infrastructure** — `tools/eval/`: YAML-scripted
  golden conversations run against real, configured LLMs (never a
  simulated provider), scored with tier-1 hard asserts (topic-change
  decisions) and tier-2 embedding-similarity scoring (topic labels,
  extracted memory items, consolidated facts — never exact string
  match, since a model's own wording isn't what's under test). Reports
  are markdown + JSON, retained across runs for comparison.
- ✅ **Storage acceleration** — retrieval originally did a brute-force
  Python cosine scan over every closed topic and memory item per
  query, with no vector index and a cross-peer filtering gap on top.
  Fixed the peer-isolation bug first, then added an optional
  [sqlite-vec](https://github.com/asg017/sqlite-vec)-backed KNN index
  (`pip install eidolon[vec]`) partitioned by peer, with automatic
  fallback to the original scan if the extension isn't available.
  Roughly 56x faster at 60k items in benchmarking, with identical
  retrieval-quality scores before and after on the eval harness.
- ✅ **Postgres/pgvector storage backend** — `PostgresStore`
  (`core/stores/postgres_store.py`) implements `StorageProvider`
  alongside `SqliteStore`, config-selectable via `storage.provider:
  postgres`. Originally scoped as a scale escalation
  ([pgvector](https://github.com/pgvector/pgvector) "for deployments
  that outgrow an embedded SQLite file"), it shipped for reliability
  instead: SQLite's per-process advisory file locking doesn't hold up
  over Docker Desktop's virtiofs bind mount (the same class of problem
  as NFS), which cost a production deployment a WAL-mode corruption and
  several stuck-write-lock incidents inside a few days. Postgres's
  single-writer-process-plus-wire-protocol model doesn't hit that
  failure class. Native `VECTOR`/HNSW replaces the BLOB+optional-
  sqlite-vec-shadow-table duality entirely — one search path, not two.
  Includes a data-migration tool
  (`tools/migrate_sqlite_to_postgres.py`) and a storage-provider
  contract test suite (`tests/core/test_storage_provider_contract.py`,
  parametrized over both backends via a real `testcontainers`-managed
  Postgres) so the two backends can't silently drift apart. `SqliteStore`
  stays the zero-config default for the CLI and tests.
- ✅ **Core-as-a-service deployment mode** — a second, opt-in way to run
  Eidolon: `core` as its own standalone HTTP process
  ([`tools/serve.py`](tools/serve.py), [`server/app.py`](server/app.py))
  instead of an adapter importing it in-process. Motivated by two
  concrete pains from running the Postgres backend above inside
  Hermes' container: installing core's own dependencies (`psycopg`) into
  the adapter's venv is fragile and doesn't survive a container
  recreation, and a core code change requires restarting the whole
  Hermes gateway to pick it up. `EidolonClient`
  ([`core/eidolon_client.py`](core/eidolon_client.py)) is the shared
  contract — `core.Eidolon` (in-process) and
  [`client.http_client.EidolonHttpClient`](client/http_client.py) (over
  HTTP, using `requests` — already a base dependency, so the "server"
  mode needs zero new packages in an adapter's venv) both satisfy it
  structurally. Same escalation-path shape as `SqliteStore`/
  `PostgresStore`, one layer up: **embedded** (in-process, `sqlite-vec`,
  nothing to run) stays the default, **server** (its own process,
  `pgvector`, decoupled lifecycle —
  [`deploy/docker-compose.example.yml`](deploy/docker-compose.example.yml)
  bundles both, mirroring Honcho's own API+Postgres compose pattern) is
  opt-in via one `deployment.mode` config field. Nothing about core's
  actual logic is duplicated — the server is a thin FastAPI translation
  layer over the facade's ~9 methods, reusing `core.models` dataclasses
  directly as the wire format (no parallel DTO layer). Deliberately
  scoped to what a conversational adapter needs, not a general-purpose
  browsing API — see [`tests/core/test_eidolon_client_contract.py`](tests/core/test_eidolon_client_contract.py)
  (parity coverage between the two implementations, mirroring the
  storage-provider contract suite) and
  [`tests/core/test_eidolon_http_smoke.py`](tests/core/test_eidolon_http_smoke.py)
  (a real server subprocess over real sockets). `tools/cli.py` stays
  `embedded`-only (its debug commands read storage directly).
- ✅ **Stale-topic auto-close** — Hermes has no session-end hook, so a
  topic a peer just abandons mid-conversation used to stay `active`
  indefinitely (`close_stale_topics`/`get_stale_active_topics`,
  Phase 2, existed but nothing called them automatically). Fixed with
  an activity-gated sweep in `EidolonMemoryProvider` (`adapters/hermes/
  memory_provider.py`, `maintenance.*` config, off by default) —
  deliberately *not* a fixed timer, since closing a topic while its
  peer is away risks fragmenting a conversation they'd have naturally
  continued once they return. Piggybacks on real per-turn activity
  instead, checked no more than once every `maintenance.interval_hours`
  and run after `store_turn()` so a turn's own topic can never be
  caught by the sweep it just triggered.
- ✅ **Recurring-topic clustering** — `ClusterConsolidator` merges
  related closed topics (e.g. several separate mentions of "buying a
  new bike" over days or weeks) into a single, continuously-rewritten
  narrative, so a recurring thread doesn't quietly fall out of context
  once it has more instances than retrieval's per-turn limit.
- 🚧 **Topic detection & summarization review** — the detection prompt
  has held up well across three structurally different local/cloud
  models in eval runs so far, which lowers the urgency of a dedicated
  review pass, but it hasn't had one yet.
- ✅ **Stuck topic labels fixed** — a topic's label was guessed from
  just the first message and never revisited, not even at close
  (`close_topic()` only ever wrote `status`/`summary`/`closed_at`) — a
  message that's mostly small talk stayed labeled "Greeting" forever,
  including live in-conversation display, even once the conversation
  turned into something substantive. New topics now start pending
  (`label=""`), finalized by whichever comes first: `TopicRelabeler`
  (`core/summarization/topic_relabeler.py`, new) firing mid-conversation
  from `store_turn`'s CONTINUE branch once a still-pending topic
  accumulates `RelabelConfig.message_threshold` messages (default 4,
  on by default) — deliberately activity-gated rather than
  timer-driven, since closing/relabeling a topic while its peer is
  away risks fragmenting a conversation they'd have naturally
  continued — or `TopicSummarizer`'s existing close-time pass, now
  also producing a label in the same LLM call, as the fallback for
  topics that close before ever reaching the threshold.
- ✅ **Mini-topic soft-hide** — roughly 25% of topics in live usage
  were 1-2 message asides too thin to be worth surfacing as their own
  unit, cluttering `[Related topics]`/`[Recently discussed]` alongside
  genuinely substantial topics. Fixed with a persisted
  `topics.message_count` column (kept in sync via an explicit `UPDATE`
  in `store_message()`, no DB trigger), filtered at query time in
  `search_topics`/`get_recent_topics` (`retrieval.min_topic_messages`,
  default 3) — a raw stored count rather than a boolean flag baked in
  at topic-close time, so changing the threshold later reclassifies
  every topic, old and new, for free, with no backfill. Nothing is
  ever deleted or merged (B1 "soft-hide"): a thin topic's row,
  messages, and any extracted memory items stay fully intact, they
  just stop being *returned* by those two queries below the threshold.
  Scoped to standalone one-offs only — see the "interrupting tangents"
  case still open below.
- 🚧 **Fact consolidation hardening** — `FactConsolidator` already ships
  and works (Phase 1); ongoing work is about improving edge cases like
  the boundary between "essence" and "fact" classification during
  extraction. Early tuning attempts here ran into a real methodology
  problem: single before/after eval runs aren't a reliable signal, since
  run-to-run variance in cloud-hosted models can swamp the sample size
  of a short golden script. Next step is a more statistically sound way
  to compare runs (e.g. averaging several repeated runs per config)
  before trying further prompt/logic changes.

## Phase 3 — Planned

- **Topic-Only Mode** — an alternative integration path that runs
  alongside another memory provider via Hermes' generic
  `pre_llm_call`/`post_llm_call` hooks instead of replacing it outright
  (scaffolded in `adapters/hermes/llm_hook.py`, not yet wired to a live
  hook) — for cases where the primary `EidolonMemoryProvider` full-mode
  integration isn't sufficient on its own.
- Production deployment guidance, performance optimization beyond the
  storage work above.
- **OpenClaw adapter** — a second platform adapter alongside
  `adapters/hermes/`, following the same `core`-never-imports-`adapters`
  boundary and provider-injection pattern. Not implemented yet, but a
  real reconnaissance pass was done (2026-07-24) to size the effort —
  findings below so a future session doesn't have to rediscover them.

  **Runtime mismatch, and why it doesn't block this**: OpenClaw is a
  Node.js/TypeScript gateway (its global npm install's real source —
  `dist/`, `docs/plugins/`, etc. — is what this reconnaissance pass
  actually read); native plugins run in-process, in JS — there is
  no way to embed `core.Eidolon` directly the way Hermes does. That
  turns out not to matter: **`eidolon-server`** (this session's
  "Core-as-a-service" work) already exposes exactly the HTTP contract
  a JS plugin would need (`/prepare-context`, `/store-turn`, etc.,
  the same surface `EidolonHttpClient` speaks) — an OpenClaw adapter
  would be a thin TypeScript HTTP client, no changes to `core/` at all.

  **Real precedent found and inspected**: `@honcho-ai/openclaw-honcho`
  — the OpenClaw plugin for Honcho, the same memory provider already
  running in this user's live Hermes deployment — is cached locally at
  `~/.openclaw/npm/projects/honcho-ai-openclaw-honcho-*/node_modules/
  @honcho-ai/openclaw-honcho/` (path will shift on reinstall/update;
  re-find via `find ~/.openclaw/npm -iname "*openclaw-honcho*"` if
  gone). Its `dist/` source was read in full — the single best
  reference for what this port actually needs, since it solves the
  identical problem (an external AI-native memory service plugged into
  OpenClaw's memory slot) that an Eidolon plugin would solve.

  **What the port would actually involve**, per that reference:
  - Packaging: a real npm package — `package.json` with an `openclaw`
    metadata block (`plugin`, `extensions`, `build.openclawVersion`,
    `compat.pluginApi`) plus an `openclaw.plugin.json` manifest
    (`id`, `kind: "memory"`, `contracts.tools`, JSON-schema
    `configSchema`, `uiHints`), built via `tsc`. New tooling this repo
    has never needed — no JS/TS build pipeline exists here today.
  - Hook mapping is clean 1:1: `before_prompt_build` ≈
    `prepare_context`/`retrieve_context`; `agent_end` ≈ `store_turn`.
  - `plugins.slots.memory` is an **exclusive slot** — an Eidolon
    plugin would contend with whatever currently owns it (today:
    OpenClaw's builtin, since nothing is set explicitly), same
    single-active-provider constraint Hermes already has.
  - The real extra weight, not present in `adapters/hermes/
    session_state.py` today: OpenClaw's capture hook does per-session
    watermarking (`lastSavedIndex`, so a re-fired hook never re-sends
    already-persisted messages), chunks at 100 messages/request,
    resolves a sender id per message for group-chat multi-party
    conversations, filters noise patterns, tracks sub-agent parent/
    child sessions, and flushes on three separate triggers
    (`agent_end`, `before_compaction`, `before_reset`) so nothing is
    lost when OpenClaw truncates a transcript. None of that exists in
    Hermes' simpler one-call-per-turn model — this is where the actual
    size of the port lives, not in talking to `eidolon-server` itself.

  **Verdict**: doable without dramatic changes to `core/` — real work,
  concentrated in a new TS session-bookkeeping layer heavier than
  `session_state.py`'s Hermes equivalent, plus adopting TS build/test
  tooling from scratch. Estimate multi-day, not a quick afternoon
  adapter; worth its own scoping pass (own plan, own session) rather
  than folding into an existing one. Revisit once a concrete need
  arises (e.g. wanting Eidolon alongside/instead of Honcho on the
  OpenClaw side), not before.

## Ideas / open questions (not yet scoped)

Raw observations from live-testing the deployed Hermes integration —
not committed to Phase 3 yet, just captured here so they don't get
lost. Pick up and scope properly when one of these becomes a priority.

- **Interrupting tangents fragmenting a real conversation** — the
  other half of the mini-topics problem (see the ✅ shipped soft-hide
  entry above for the standalone-one-off half). Re-checked against live
  data before scoping this as new work, and it turned out to already be
  handled: the exact example that motivated this entry (an image-prompt/
  LoRA discussion split by two tangents) is fully re-linked in the live
  DB today — `ClusterConsolidator` (embedding/tag/temporal-scored
  candidates, LLM match judgment, runs on every topic close) correctly
  grouped all the image-prompt fragments into one cluster while keeping
  the two interrupting tangents in a separate cluster of their own, and
  left a genuine standalone one-off unclustered. Between that and the
  stuck-topic-labels fix (better labels reduce how often a resumed
  subject even reads as "new" to begin with), both halves of the
  original observation may already be covered by mechanisms that
  shipped for other reasons. Deferring further action here — not
  reopening unless real data later shows a case clustering actually
  misses (e.g. a longer gap or weaker semantic overlap between
  fragments than this example had).
- **More explicit temporal modeling** — when did what happen, and how
  do topics relate to each other in time, not just via the embedding/
  tag similarity retrieval already weighs (`retrieval.temporal`).
- **Artificial salience + associative (non-relevance) recall** — a much
  bigger, genuinely uncertain idea, explicitly not committed to and put
  here only so it isn't lost. Motivation: human memory doesn't retrieve
  purely by topical relevance — it's shaped by emotional impact (an AI
  has none, so this would have to be faked with an artificial score) and
  by arbitrary co-occurrence. Two distinct mechanisms were named, and
  they're genuinely different problems:
  - **Temporal-proximity binding** — two topics linked purely because
    they happened close together in time, regardless of subject (e.g. a
    house move forever bound to a vacation two weeks earlier). Cheap to
    detect (a time-window join), but raw proximity alone over-generates
    — most things that happen in the same two weeks aren't bound
    together, so there's usually a second ingredient (significance)
    that makes proximity memorable, not proximity by itself.
  - **Salience/vividness triggers** — a sensory or emotional detail
    (heavy rain, a TV show playing in the background) acting as a
    retrieval key regardless of elapsed time. This needs an artificial
    "vividness" marker on a topic/memory (parallel to `Fact.importance`)
    plus a trigger-detector on new messages, closer to an associative
    index than anything time-based.

  Explored but deliberately not decided: how such a score gets
  estimated (LLM-judged at close time vs. inferred from language cues
  like intensity words), and how it would surface at retrieval — folding
  it into the existing scored `search_topics`/`search_memory_items`
  pipeline risks degrading precision on ordinary "help me debug this"
  turns, so a small, separately-gated, possibly probabilistic injection
  channel (distinct from `[Related topics]`, easy to disable) seems
  safer than blending it into the main scoring path. The real open
  question isn't mechanism, it's whether occasional non-relevant recall
  reads as "natural"/human or as noise — worth prototyping the smallest
  possible version (an LLM-judged vividness score at topic-close + one
  trigger-matching retrieval call) and actually sitting with it in a
  real conversation before designing further, rather than speccing this
  fully upfront.
- **Direct topic-to-topic relations** — beyond the existing cluster
  grouping, could topics reference each other directly (e.g. "this is
  a follow-up to topic X")?
- **Explicit "current status" line in cluster summaries** — questionable,
  low-conviction, try only when in the mood for an experiment. Motivated
  by wanting the LLM to volunteer a thread's latest resolution (e.g.
  "want to use Z Image Turbo again?") without having to infer it from
  narrative prose. `_CLUSTER_SYSTEM_PROMPT` already trends toward ending
  with the resolution in real live output, so the actual gap (if any) is
  unclear — the idea was floated, then explicitly walked back as
  not-well-thought-through rather than scoped. If revisited: a prompt
  tweak asking the rewrite to close with an unambiguous status sentence,
  not a new schema field (a separate field risks drifting out of sync
  with the narrative it summarizes) — and check against real cluster
  data before/after rather than assuming it helps.
- **Usefulness scoring for topics/facts** — some notion of which
  topics/facts actually get surfaced and used again vs. which never
  come up. Open question: how to even measure "useful" — retrieval
  frequency? explicit user correction/confirmation signals?
- **The insights/monitoring tool, revisited** — not built *for* the
  core-as-a-service work, but that work is what makes it newly
  practical: a standalone tool could talk to a running `eidolon-server`
  (only the ~9-method `EidolonClient` facade — deliberately scoped to
  adapter-lifecycle needs, see `core/eidolon_client.py`) or connect to
  Postgres directly for read-only browsing, which the storage migration
  already made safe to do. Worth a real design pass once the above
  ideas clarify what such a tool would actually need to show — several
  of them (temporal relations, usefulness scoring) are exactly the kind
  of thing a monitoring tool would want to surface.

## Shipped ahead of schedule

- **Weighted associative retrieval** — `semantic`/`tag_overlap`/`person`/
  `temporal` weights are configurable and actually wired into scoring.
- **Debug tracing** — see [Debug Tracing](README.md#debug-tracing) in
  the README; off by default, zero-overhead when disabled, records
  every step (prompts, responses, retrieval score breakdowns) to a
  separate SQLite database for inspection via a terminal viewer or a
  Textual TUI.
- **Batch facts consolidation** — turn-interval-triggered periodic
  reconciliation of the durable-facts profile, replacing naive
  per-turn extraction — see [Facts Consolidation](README.md#facts-consolidation).
