# Dataset Intelligence Platform — Example API Requests

All examples use `http://localhost:8000` as the base URL.
Replace UUIDs with actual values from your responses.

---

## 1. Initiate Upload

```bash
curl -X POST http://localhost:8000/v1/datasets/initiate-upload \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "sales_data.csv",
    "file_size_bytes": 1048576,
    "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }'
```

**Response:**
```json
{
  "dataset_id": "...",
  "job_id": "...",
  "upload_url": "http://minio:9000/datasets/raw/.../source.csv?X-Amz-...",
  "expires_at": "2026-04-01T14:15:00Z"
}
```

---

## 2. Upload File to S3 (using the presigned URL)

```bash
curl -X PUT "<upload_url_from_step_1>" \
  -H "Content-Type: text/csv" \
  --data-binary @sales_data.csv
```

---

## 3. Complete Upload

```bash
curl -X POST http://localhost:8000/v1/datasets/{dataset_id}/complete-upload
```

**Response:**
```json
{
  "dataset_id": "...",
  "status": "uploaded",
  "analyze_job_id": "..."
}
```

---

## 4. Check Job Status

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

**Response:**
```json
{
  "id": "...",
  "dataset_id": "...",
  "job_type": "analyze",
  "status": "completed",
  "started_at": "2026-04-01T14:01:00Z",
  "completed_at": "2026-04-01T14:01:15Z",
  "metadata": {"llm_assisted": true, "tier_used": 1}
}
```

---

## 5. Analyze Dataset (manual trigger)

```bash
curl -X POST http://localhost:8000/v1/datasets/{dataset_id}/analyze
```

---

## 6. Clean Dataset

```bash
curl -X POST http://localhost:8000/v1/datasets/{dataset_id}/clean \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Remove all rows where revenue is negative. Standardize country names to ISO 3166-1 alpha-2 codes. Fill missing email addresses with N/A.",
    "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }'
```

**Response:**
```json
{
  "job_id": "...",
  "status": "queued"
}
```

---

## 7. Download Cleaned File

```bash
# Parquet format (default)
curl "http://localhost:8000/v1/datasets/{dataset_id}/download?format=parquet"

# CSV format
curl "http://localhost:8000/v1/datasets/{dataset_id}/download?format=csv"
```

**Response:**
```json
{
  "download_url": "http://minio:9000/datasets/processed/.../cleaned.parquet?X-Amz-...",
  "expires_at": "2026-04-01T15:00:00Z",
  "format": "parquet",
  "row_count": null
}
```

---

## 8. Query Dataset (NL → SQL)

```bash
curl -X POST http://localhost:8000/v1/datasets/{dataset_id}/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top 10 customers by total revenue?",
    "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }'
```

**Response:**
```json
{
  "sql": "SELECT customer_name, SUM(revenue) AS total_revenue FROM dataset GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 10",
  "results": [
    {"Customer Name": "Acme Corp", "total_revenue": 150000.0},
    {"Customer Name": "Widget Inc", "total_revenue": 120000.0}
  ],
  "row_count": 10,
  "confidence": 0.92,
  "execution_time_ms": 45,
  "assumptions": ["Assumed 'revenue' refers to the 'revenue' column"]
}
```

---

## Full Workflow (sequential)

```bash
# 1. Initiate upload
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/datasets/initiate-upload \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.csv","file_size_bytes":148,"user_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}')

DATASET_ID=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['dataset_id'])")
UPLOAD_URL=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['upload_url'])")

# 2. Upload file
curl -X PUT "$UPLOAD_URL" -H "Content-Type: text/csv" --data-binary @test.csv

# 3. Complete upload (auto-triggers analysis)
COMPLETE=$(curl -s -X POST "http://localhost:8000/v1/datasets/$DATASET_ID/complete-upload")
ANALYZE_JOB=$(echo $COMPLETE | python3 -c "import sys,json; print(json.load(sys.stdin)['analyze_job_id'])")

# 4. Poll analysis job
curl -s "http://localhost:8000/v1/jobs/$ANALYZE_JOB"

# 5. Clean dataset
CLEAN=$(curl -s -X POST "http://localhost:8000/v1/datasets/$DATASET_ID/clean" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Standardize all columns and remove duplicates","user_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}')
CLEAN_JOB=$(echo $CLEAN | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# 6. Poll cleaning job
curl -s "http://localhost:8000/v1/jobs/$CLEAN_JOB"

# 7. Query the cleaned dataset
curl -s -X POST "http://localhost:8000/v1/datasets/$DATASET_ID/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Show all unique values","user_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}'

# 8. Download cleaned file
curl -s "http://localhost:8000/v1/datasets/$DATASET_ID/download?format=csv"
```
