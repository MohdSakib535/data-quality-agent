# AI CSV Cleaning Agent

A production-ready, general-purpose AI agent that safely cleans any CSV file without pre-defined schemas. It uses heuristic analysis for fundamental cleaning and standardizing, and delegates only ambiguous text normalization and category mapping to a local LLM via Ollama and LangChain, ensuring perfect data privacy and zero hallucination.

## Tech Stack
- **FastAPI**: High-performance asynchronous API endpoints
- **Pandas**: Fast matrix manipulation and column profiling
- **LangChain**: Orchestrating structured JSON requests to the LLM
- **Ollama**: Local LLM execution (Default: `llama3`)
- **Pydantic**: Deep strict validation of AI output

## How It Works

1. **Upload**: User posts a CSV file to `/api/v1/upload-csv`. The system returns a `job_id`.
2. **Clean**: User triggers `/api/v1/clean-csv/{job_id}`.
   - The cleaning request runs synchronously and returns only after processing finishes with `completed` or `failed`.
   - **Profile**: System calculates sparsity, unique ratios, string limits.
   - **Infer Types**: Safely predicts `category_text`, `date`, `float`, `identifier`, etc.
   - **Deterministic Clean**: Trims strings, homogenizes `null`/`N/A` into blanks, normalizes header text to `snake_case`. Pure pandas logic, 100% safe.
   - **AI Clean**: Batches remaining muddy textual columns (`category_text`, `name`) and asks Ollama to standardize spelling, correct casing, or expand standard acronyms, producing strict JSON arrays.
   - **Validation**: Checks AI confidence. Outputs >= 95% are adopted automatically. Lower confidence results in the row being shunted to a manual review queue.
3. **Download**: User downloads the output artifacts:
   - `/api/v1/download-cleaned/{job_id}`: The pristine CSV.
   - `/api/v1/download-review/{job_id}`: A sub-CSV containing rows the AI was unsure about.
   - `/api/v1/download-audit/{job_id}`: A deep JSON log proving *every single decision* (AI or heuristic), keeping the original value intact for traceback.

## Running Locally

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama
Ensure Ollama is installed locally (https://ollama.com). Pull the default model:
```bash
ollama run llama3
```
Ensure Ollama is running on `http://localhost:11434` (the default).

### 3. Start the API
```bash
python -m app.main
```
Or via uvicorn:
```bash
uvicorn app.main:app --reload
```

Swagger documentation will be available at: http://localhost:8000/docs

## Running With Docker Compose

### 1. Build and start the services
```bash
docker compose up -d
```

This starts:
- `api`: FastAPI application on `http://localhost:8000`

The Compose setup bind-mounts the project into the container and runs Uvicorn with `--reload`, so normal Python code changes reload automatically without rebuilding the image.

Rebuild only when dependencies or the image definition change:
```bash
docker compose up --build -d
```

### 2. Ensure local Ollama is already running
Your Docker setup expects Ollama to be running on the host at:
```bash
http://localhost:11434
```

On Docker Desktop for macOS, the container reaches that host service through `http://host.docker.internal:11434`.

Pull the model on your host machine if needed:
```bash
ollama pull llama3
```

If you want a different model, update `OLLAMA_MODEL` in `docker-compose.yml` and pull that model locally instead.

### 3. Open the API docs
Swagger documentation will be available at: http://localhost:8000/docs

### 4. Stop the stack
```bash
docker compose down
```

## An Example Scenario

Imagine you upload a `messy_users.csv`:
```csv
User ID , First   Name, Email Address       , Job Title      , Age 
1       , John        , john@example.com    , ui/ux designer , 29
2       , Jane        , jane@test.com       , UI UX Desginer , 31
3       , N/A         , null                , Manager        , -  
```

**The Pipeline:**
1. **Deterministic Clean**:
   - Headers become: `user_id`, `first_name`, `email_address`, `job_title`, `age`
   - Row 3 `N/A` and `null` string and `-` become system null values (`NaN`).
   - Extra spaces (e.g. `First   Name`) are normalized.
2. **Heuristic Inferencing**:
   - `user_id`: Inferred as `identifier`. Left alone by AI.
   - `age`: Inferred as `float`. Left alone by AI.
3. **AI Cleaning**:
   - `job_title` has `'ui/ux designer'` and `'UI UX Desginer'`.
   - Sent to Ollama in a batch.
   - Ollama returns JSON replacing both with `"UI/UX Designer"` with 0.98 confidence.
   - Applied safely.

**Result**: A clean target CSV and a JSON audit trail explaining the title consolidation.
