-- ============================================================
-- Dataset Intelligence Platform — PostgreSQL Schema
-- ============================================================

-- Enable uuid-ossp for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ──────────────────────────────────────────────────────────────
-- 1. datasets
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS datasets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL,
    filename        VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    s3_raw_key      TEXT NOT NULL,
    s3_cleaned_key  TEXT,
    status          VARCHAR(50) NOT NULL DEFAULT 'pending_upload',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_datasets_user_id    ON datasets (user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_status     ON datasets (status);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets (created_at);

-- ──────────────────────────────────────────────────────────────
-- 2. jobs
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id      UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    job_type        VARCHAR(20) NOT NULL CHECK (job_type IN ('analyze', 'clean', 'query')),
    status          VARCHAR(20) NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'degraded')),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    retry_count     INT NOT NULL DEFAULT 0,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_dataset_id  ON jobs (dataset_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at  ON jobs (created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_job_type    ON jobs (job_type);

-- ──────────────────────────────────────────────────────────────
-- 3. dataset_profiles
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_profiles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id      UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_count       BIGINT,
    column_count    INT,
    schema_snapshot JSONB,
    quality_score   DOUBLE PRECISION,
    null_stats      JSONB,
    duplicate_count BIGINT,
    llm_suggestions JSONB DEFAULT '[]'::jsonb,
    profiled_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profiles_dataset_id ON dataset_profiles (dataset_id);

-- ──────────────────────────────────────────────────────────────
-- 4. cleaned_data_preview
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cleaned_data_preview (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id              UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    job_id                  UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    preview_rows            JSONB,
    cleaned_row_count       BIGINT,
    prompt_used             TEXT,
    cleaning_steps_applied  JSONB DEFAULT '[]'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cleaned_preview_dataset_id ON cleaned_data_preview (dataset_id);
CREATE INDEX IF NOT EXISTS idx_cleaned_preview_job_id     ON cleaned_data_preview (job_id);

-- ──────────────────────────────────────────────────────────────
-- 5. query_logs
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id              UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    natural_language_query   TEXT NOT NULL,
    generated_sql           TEXT,
    validated               BOOLEAN DEFAULT FALSE,
    confidence_score        DOUBLE PRECISION,
    execution_time_ms       INT,
    result_row_count        INT,
    error_message           TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_dataset_id  ON query_logs (dataset_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at  ON query_logs (created_at);

-- ──────────────────────────────────────────────────────────────
-- 6. parquet_schema_versions
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS parquet_schema_versions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id          UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version             INT NOT NULL DEFAULT 1,
    schema_fingerprint  TEXT NOT NULL,
    column_map          JSONB NOT NULL DEFAULT '{}'::jsonb,
    parquet_metadata    JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (dataset_id, version)
);

CREATE INDEX IF NOT EXISTS idx_parquet_schema_dataset_id ON parquet_schema_versions (dataset_id);

-- ──────────────────────────────────────────────────────────────
-- Auto-update updated_at trigger for datasets
-- ──────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
