# Eidolon Dashboard

A small read-only web app (server-rendered Jinja2 + [htmx](https://htmx.org),
vendored locally — no build step, no CDN dependency) for inspecting what
Eidolon's memory pipeline is actually doing: one turn at a time, or
aggregated over a time window.

Part of `eidolon-server` (see [Deployment Modes](../../README.md#deployment-modes)
in the main README) rather than a separate process or container: it's
mounted under `/dashboard` on the same `tools/serve.py`, sharing that
one process's `Eidolon`/storage instance instead of opening a second,
independent connection to the database. That's deliberate — the moment
write actions land here (correcting a fact, closing a topic on demand),
they need to go through core's real orchestration (`TopicSummarizer`,
`ClusterConsolidator`, ...), not a raw database write, and the only way
to guarantee that without a second live writer racing the first is to
have exactly one process ever touch the database through core.
Read-only for now, `server`-deployment-mode only (there's no
separately-addressable core process to mount it on in `embedded` mode).

Reachable at `http://<server-host>:8420/dashboard/` once `eidolon-server`
is running — nothing extra to start:

```bash
EIDOLON_DASHBOARD_PASSWORD=... uv run python tools/serve.py --config eidolon.yaml
```

Gated by its own optional HTTP Basic auth, independent of the RPC API's
bearer token (skipped entirely if unset — same "binds to localhost by
default" threat model `tools/serve.py` already uses for its own default
host).

## Two data tiers

Every page degrades gracefully across two tiers rather than requiring
debug mode outright:

- **Always-on**: sourced from `messages`/`topics`/`events`/`llm_calls`/
  `embedding_calls` — real production data regardless of settings.
- **Rich** (only for turns/windows that ran while `debug.enabled: true`):
  the full [Debug Tracing](../../README.md#debug-tracing) payload —
  prompts, responses, retrieval score breakdowns.

`request_id` is the correlation key threading both tiers together —
minted once per turn (`core/__init__.py`'s `store_turn`) and passed into
everything it triggers, so a step's mere presence or absence for that id
already answers "did this happen." The five aggregate pages below all
share the same window selector (`1h`/`24h`/`7d`).

## Pages

### Overview & Turns (`/dashboard/`, `/dashboard/turns`)

Landing page and a filterable (by `peer_id`) list of recent turns —
always-on, the entry point into turn detail.

### Turn Detail (`/dashboard/turns/{request_id}`)

Everything that happened during **one** turn: which topic was active
before/after, the user message and AI response, then a step-by-step
account of what fired — topic detection, extraction, every embedding
call, retrieval + scoring, and, conditionally, whatever the turn
triggered (fact consolidation, topic close/summarize, relabel, cluster
consolidation). Master-detail layout: a compact step list on the left,
full detail for the selected step on the right.

- **Skeleton** (always available): real message text and topic linkage,
  plus which steps ran with their timing/success.
- **Rich** (debug-gated): the full trace payload per step, shown instead
  of the skeleton whenever it exists for that turn.

### Performance & Reliability (`/dashboard/performance`) — always-on

Latency percentiles (p50/p95/p99) per LLM call type, error rate over
time, embedding latency by purpose, hourly throughput, and top error
sources by peer/session. The "cheap" insight group — needs zero debug
tracing, sourced entirely from the always-on operational-metrics tables.

### Retrieval Quality & Weighting (`/dashboard/retrieval-quality`) — debug-gated

Which topics/memory items/clusters get retrieved most often (over-used
content) vs. scored but never included (a decay signal); whether
`semantic` actually dominates every score or `tag_overlap`/`person`/
`temporal` meaningfully contribute — i.e. whether the configured
`retrieval.*` weights do anything; near-miss candidates just below the
inclusion cutoff; and which facts got injected into context most often.

### Prompt & Pipeline Quality (`/dashboard/prompt-quality`) — mostly debug-gated

How often `topic_detection` falls back to `CONTINUE` because the LLM's
response wasn't parseable; topics whose label flip-flopped across
`topic_relabel` calls; the extracted `item_type`/`attribution_type`
mix; facts whose category falls outside the configured list
(**always-on** — reflects the current facts table, not the selected
window); consolidation churn (a key upserted more than once, with
distinct-value tracking to separate genuine flip-flops from mere
reconfirmation); and manual fact-correction frequency (a trust signal).

### Topic Lifecycle (`/dashboard/topic-lifecycle`) — mostly always-on

Average/p50/p95 topic lifespan to close (too short may mean over-eager
`NEW_TOPIC` detection) and the shortest-lived topics in the window —
**always-on**, straight from `topics.created_at`/`closed_at`/
`message_count`. The one debug-gated piece: how many closes came from
the idle-timeout stale sweep vs. a natural transition (there's no
`topics.closed_reason` column, so this needs correlating against the
`stale_topic_cleanup` trace event).

### Cluster Dynamics (`/dashboard/cluster-dynamics`) — debug-gated

Whether topic clustering tends to match an existing cluster, create a
brand-new one, or find no confident match at all; and which clusters
are seeing the most consolidation activity in the window (labeled
"consolidation events", not a precise "topics added" count — a
cluster's founding event bundles two topic assignments into one).

## Adding a new page

Every aggregate page follows the same shape — worth knowing if adding
another one: a `*Reader` composition class in `tools/dashboard/` (reads
via `StorageProvider`/`TraceReader`, does the aggregation math, returns
a plain dataclass — see `retrieval_quality_reader.py` for the fullest
example), a route in `app.py` reusing the shared `_WINDOW_CUTOFFS`, a
template extending `base.html`, and a nav link. `TraceReader.
events_of_type_since(event_type, since)` is the one bulk primitive every
debug-gated page's reader calls — no new `TraceReader`/`StorageProvider`
method is needed unless the page needs data no existing method already
exposes.
