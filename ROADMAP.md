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
- ✅ **Recurring-topic clustering** — `ClusterConsolidator` merges
  related closed topics (e.g. several separate mentions of "buying a
  new bike" over days or weeks) into a single, continuously-rewritten
  narrative, so a recurring thread doesn't quietly fall out of context
  once it has more instances than retrieval's per-turn limit.
- 🚧 **Topic detection & summarization review** — the detection prompt
  has held up well across three structurally different local/cloud
  models in eval runs so far, which lowers the urgency of a dedicated
  review pass, but it hasn't had one yet.
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
  boundary and provider-injection pattern. Not scoped in detail yet;
  revisit once the Hermes adapter and Phase 2 work are settled.

## Ideas / open questions (not yet scoped)

Raw observations from live-testing the deployed Hermes integration —
not committed to Phase 3 yet, just captured here so they don't get
lost. Pick up and scope properly when one of these becomes a priority.

- **Topics never close on their own** — Hermes has no session-end
  hook, so a session's last topic stays "active" indefinitely once the
  user just stops chatting mid-topic. `close_stale_topics`/
  `get_stale_active_topics` (Phase 2, shipped) exist for exactly this,
  but nothing on the live path calls them automatically today — it's a
  manual `tools/cli.py` operation. Needs either a scheduled trigger
  (cron-style, natural now that `tools/serve.py` is a standalone
  long-running process) or a different mechanism entirely.
- **Lots of 1-3 turn "mini-topics"** — worth a merge/consolidation pass,
  distinct from `ClusterConsolidator`'s existing recurring-topic
  clustering (which groups *related* topics across time, not small
  topics by turn count). Open question: merge into the nearest related
  topic, or fold into a catch-all bucket?
- **Topic names chosen too early** — the label largely reflects the
  first turn or two; by turn 5 the real topic may look different.
  Possible fix: re-run labeling once N turns have accumulated, rather
  than only relying on the label set at creation (`TopicSummarizer`
  already runs a separate summary pass at topic *close* — the question
  is whether the early label itself needs revisiting mid-conversation).
- **More explicit temporal modeling** — when did what happen, and how
  do topics relate to each other in time, not just via the embedding/
  tag similarity retrieval already weighs (`retrieval.temporal`).
- **Direct topic-to-topic relations** — beyond the existing cluster
  grouping, could topics reference each other directly (e.g. "this is
  a follow-up to topic X")?
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
