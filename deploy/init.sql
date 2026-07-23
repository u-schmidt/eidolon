-- Bootstraps the pgvector extension the pgvector/pgvector:pg15 image
-- doesn't enable by default. Mounted into Postgres's init directory by
-- docker-compose.example.yml — runs once, on first container startup
-- against an empty data directory. Same one-line bootstrap already used
-- outside this repo for the live Hermes deployment's Postgres instance;
-- core/stores/postgres_migrations.py handles all further schema setup
-- (tables, indexes) itself once the server connects.
CREATE EXTENSION IF NOT EXISTS vector;
