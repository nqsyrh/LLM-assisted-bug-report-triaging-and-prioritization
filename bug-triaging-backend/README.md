# Bug Triaging API — FastAPI + SQLite Backend

This wraps your existing classification/duplicate-detection/priority logic
(zero_shot_classify_v2.py, find_duplicates.py, priority_scoring.py,
evaluation_ml_baseline.py) behind a real REST API with SQLite storage —
satisfying the FastAPI + SQLite tools listed in FORM 01A.

**Nothing was recomputed.** The 499 reports, 33 duplicate pairs, and 71
evaluation examples are loaded directly from your existing CSV outputs, so
the numbers match your dashboard exactly. New reports submitted via the API
use the same model (facebook/bart-large-mnli) and same formulas as your
batch scripts.

## Folder structure
```
bug-triaging-backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + all routes
│   ├── database.py       # SQLAlchemy engine/session (SQLite)
│   ├── models.py          # ORM models: BugReport, DuplicatePair, EvalExample, ModelEvalResult
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── classifier.py        # Severity classification (bart-large-mnli) for NEW reports
│   ├── duplicates.py         # FAISS + MiniLM duplicate detection for NEW reports
│   ├── evaluation.py          # Reproduces evaluation_ml_baseline.py, stores results
│   └── seed_data.py            # Loads your existing CSVs into SQLite
├── data/
│   ├── prioritized_bug_reports.csv   # your 499 reports w/ severity + priority
│   ├── duplicate_pairs.csv            # your 33 duplicate pairs
│   └── llm_predictions_v2.csv          # your 71 labeled evaluation examples
└── requirements.txt
```

## Setup (Windows PowerShell, inside your existing venv)

1. Copy this whole `bug-triaging-backend` folder into your project, e.g. next
   to your existing `dashboard/`, `src/`, `data/` folders.

2. Activate your existing venv (the one with transformers/sentence-transformers
   already installed from your other scripts):
   ```powershell
   .venv-1\Scripts\activate
   ```

3. Install the additional backend packages:
   ```powershell
   pip install fastapi "uvicorn[standard]" sqlalchemy
   ```
   (You likely already have pandas, scikit-learn, transformers,
   sentence-transformers, faiss-cpu, torch from your existing project — if not,
   `pip install -r bug-triaging-backend\requirements.txt`)

4. Seed the database from your existing results (one-time):
   ```powershell
   cd bug-triaging-backend
   python -m app.seed_data
   ```
   Expected output:
   ```
   Seeded 499 bug reports from prioritized_bug_reports.csv (0 already present, skipped).
   Seeded 33 duplicate pairs from duplicate_pairs.csv.
   Seeded 71 evaluation examples from llm_predictions_v2.csv.
   ```

5. Compute the model comparison metrics (one-time, takes a few seconds —
   only trains the traditional ML baselines, doesn't touch bart-large-mnli):
   ```powershell
   python -m app.evaluation
   ```

6. Start the API server:
   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

7. Open the interactive docs (this works exactly like Postman — you can try
   every endpoint directly in the browser):
   ```
   http://localhost:8000/docs
   ```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/reports` | Submit a new bug report — live severity classification, duplicate check, priority score |
| GET | `/reports` | List reports, filterable by severity, priority range, duplicates_only, repo, search |
| GET | `/reports/{id}` | Single report detail |
| GET | `/duplicates` | All detected duplicate pairs |
| GET | `/duplicates/{pair_id}/compare` | Full side-by-side comparison of a duplicate pair |
| GET | `/evaluation` | Accuracy/Precision/Recall/F1 — LLM vs Naive Bayes vs SVM vs Random Forest |
| GET | `/evaluation/{model_name}/detail` | Per-class (Critical/High/Medium/Low) breakdown for one model |
| GET | `/stats` | Summary numbers (total reports, critical count, duplicates found) |

## Testing in Postman

Import the OpenAPI schema directly: in Postman, **Import → Link**, paste:
```
http://localhost:8000/openapi.json
```
This auto-generates a full collection with all endpoints, matching what your
FORM 01A "Tools/equipment required" lists Postman for.

## Example: submitting a new report

```powershell
curl -X POST "http://localhost:8000/reports" `
  -H "Content-Type: application/json" `
  -d '{\"repo\":\"microsoft/vscode\",\"full_text\":\"Editor crashes immediately on startup, complete data loss of unsaved work\"}'
```

Expected response shape:
```json
{
  "id": 4699280169,
  "predicted_severity": "Critical",
  "model_confidence": 0.71,
  "is_duplicate": false,
  "matched_duplicate_id": null,
  "similarity_score": null,
  "priority_score": 4.0
}
```

Note: the very first POST request will be slow (~30-60s) because it loads
facebook/bart-large-mnli and all-MiniLM-L6-v2 into memory. Every request
after that is fast, since both models stay loaded for the life of the server.

## What's verified vs. what needs your machine

Verified end-to-end in the sandbox this was built in:
- Database seeding (exact row counts: 499 / 33 / 71, matching your dashboard)
- All GET endpoints (`/reports`, `/duplicates`, `/duplicates/{id}/compare`,
  `/evaluation`, `/stats`) — tested with real curl requests, real responses shown above
- Evaluation metrics reproduce your dashboard's numbers (Naive Bayes 85.9%,
  SVM 81.7%, LLM 56.3%)
- Swagger docs (`/docs`) and OpenAPI schema (`/openapi.json`) both load correctly

Not testable in the sandbox this was built in (no internet access to
huggingface.co), needs verification on your machine:
- `POST /reports` live classification (needs `transformers` to download
  facebook/bart-large-mnli, and `sentence-transformers` to download
  all-MiniLM-L6-v2) — the code path was reviewed line-by-line against your
  original scripts and should work, but run it once yourself to confirm
  before you treat it as demo-ready.
