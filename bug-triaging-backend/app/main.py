"""
main.py

FastAPI application — REST layer over the bug triaging system.

Run:
    uvicorn app.main:app --reload --port 8000

Interactive API documentation:
    http://localhost:8000/docs
"""

import json
import os

from typing import (
    Optional,
    List,
)

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from sqlalchemy.orm import Session
from sqlalchemy import func

from . import (
    models,
    schemas,
    classifier,
    duplicates,
)

from .database import (
    engine,
    get_db,
    Base,
)


# ============================================================
# Database setup
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title=(
        "LLM-Assisted Bug Report "
        "Triaging API"
    ),
    description=(
        "REST API for automatic bug "
        "severity classification, "
        "semantic duplicate detection, "
        "priority ranking and "
        "model evaluation."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "message": (
            "Bug Triaging API "
            "is running"
        ),
        "docs": "/docs",
        "endpoints": [
            "/reports",
            "/duplicates",
            "/evaluation",
            (
                "/evaluation/"
                "confusion-matrix"
            ),
            "/stats",
        ],
    }


# ============================================================
# Reports
# ============================================================

@app.post(
    "/reports",
    response_model=(
        schemas.ClassifyResponse
    ),
)
def create_report(
    report: schemas.BugReportCreate,
    db: Session = Depends(get_db),
):
    """
    Submit a new bug report.

    Processing stages:

    1. Severity classification
       using facebook/bart-large-mnli.

    2. Semantic duplicate detection
       using SentenceTransformer +
       FAISS.

    3. Priority score calculation.

    4. Save the processed report
       into SQLite.
    """

    # --------------------------------------------------------
    # Assign report ID
    # --------------------------------------------------------

    new_id = report.id

    if new_id is None:

        max_id = (
            db.query(
                func.max(
                    models.BugReport.id
                )
            )
            .scalar()
            or 0
        )

        new_id = max_id + 1

    else:

        existing = (
            db.query(
                models.BugReport
            )
            .filter(
                models.BugReport.id
                == new_id
            )
            .first()
        )

        if existing:

            raise HTTPException(
                status_code=409,
                detail=(
                    f"Report with id "
                    f"{new_id} already "
                    f"exists."
                ),
            )

    # --------------------------------------------------------
    # 1. Severity classification
    # --------------------------------------------------------

    (
        severity,
        confidence,
    ) = classifier.classify_severity(
        report.full_text
    )

    # --------------------------------------------------------
    # 2. Duplicate detection
    # --------------------------------------------------------

    (
        is_dup,
        matched_id,
        similarity,
    ) = (
        duplicates
        .find_duplicate_for_text(
            report.full_text,
            db,
        )
    )

    # --------------------------------------------------------
    # 3. Priority calculation
    # --------------------------------------------------------

    priority_score = (
        classifier.calculate_priority(
            severity,
            is_dup,
        )
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    new_report = models.BugReport(
        id=new_id,
        repo=report.repo,
        full_text=report.full_text,
        predicted_severity=severity,
        model_confidence=confidence,
        is_duplicate=is_dup,
        priority_score=(
            priority_score
        ),
    )

    db.add(new_report)

    # --------------------------------------------------------
    # Save duplicate relationship
    # --------------------------------------------------------

    if (
        is_dup
        and matched_id
        is not None
    ):

        matched_report = (
            db.query(
                models.BugReport
            )
            .filter(
                models.BugReport.id
                == matched_id
            )
            .first()
        )

        pair = models.DuplicatePair(
            report_1_id=matched_id,
            report_1_text=(
                matched_report.full_text[
                    :100
                ]
                if matched_report
                else ""
            ),
            report_2_id=new_id,
            report_2_text=(
                report.full_text[:100]
            ),
            similarity_score=(
                similarity
            ),
        )

        db.add(pair)

    db.commit()

    return schemas.ClassifyResponse(
        id=new_id,
        predicted_severity=(
            severity
        ),
        model_confidence=(
            confidence
        ),
        is_duplicate=is_dup,
        matched_duplicate_id=(
            matched_id
        ),
        similarity_score=(
            similarity
        ),
        priority_score=(
            priority_score
        ),
    )


# ============================================================
# List reports
# ============================================================

@app.get(
    "/reports",
    response_model=(
        List[
            schemas.BugReportOut
        ]
    ),
)
def list_reports(
    severity: Optional[
        List[str]
    ] = Query(
        None,
        description=(
            "Filter by severity "
            "such as Critical, High, "
            "Medium or Low."
        ),
    ),

    min_priority: Optional[
        float
    ] = Query(None),

    max_priority: Optional[
        float
    ] = Query(None),

    duplicates_only: bool = Query(
        False
    ),

    repo: Optional[
        str
    ] = Query(None),

    search: Optional[
        str
    ] = Query(
        None,
        description=(
            "Keyword or bug ID search"
        ),
    ),

    limit: int = Query(
        100,
        le=1000,
    ),

    offset: int = Query(0),

    db: Session = Depends(get_db),
):
    """
    Return bug reports with optional
    dashboard filters.
    """

    q = db.query(
        models.BugReport
    )

    if severity:

        q = q.filter(
            models.BugReport
            .predicted_severity
            .in_(severity)
        )

    if min_priority is not None:

        q = q.filter(
            models.BugReport
            .priority_score
            >= min_priority
        )

    if max_priority is not None:

        q = q.filter(
            models.BugReport
            .priority_score
            <= max_priority
        )

    if duplicates_only:

        q = q.filter(
            models.BugReport
            .is_duplicate
            == True
        )

    if repo:

        q = q.filter(
            models.BugReport.repo
            == repo
        )

    if search:

        if search.isdigit():

            q = q.filter(
                models.BugReport.id
                == int(search)
            )

        else:

            q = q.filter(
                models.BugReport
                .full_text
                .ilike(
                    f"%{search}%"
                )
            )

    q = q.order_by(
        models.BugReport
        .priority_score
        .desc()
    )

    return (
        q.offset(offset)
        .limit(limit)
        .all()
    )


# ============================================================
# Get one report
# ============================================================

@app.get(
    "/reports/{report_id}",
    response_model=(
        schemas.BugReportOut
    ),
)
def get_report(
    report_id: int,
    db: Session = Depends(get_db),
):
    report = (
        db.query(
            models.BugReport
        )
        .filter(
            models.BugReport.id
            == report_id
        )
        .first()
    )

    if not report:

        raise HTTPException(
            status_code=404,
            detail=(
                "Report not found"
            ),
        )

    return report


# ============================================================
# Duplicates
# ============================================================

@app.get(
    "/duplicates",
    response_model=(
        List[
            schemas.DuplicatePairOut
        ]
    ),
)
def list_duplicates(
    min_similarity: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
    ),

    limit: int = Query(
        100,
        le=1000,
    ),

    db: Session = Depends(get_db),
):
    """
    Return semantic duplicate
    candidate pairs.

    Duplicate candidates are produced
    using:
        all-MiniLM-L6-v2
        + FAISS
        + cosine similarity

    Backend decision threshold:
        0.80
    """

    q = (
        db.query(
            models.DuplicatePair
        )
        .filter(
            models.DuplicatePair
            .similarity_score
            >= min_similarity
        )
        .order_by(
            models.DuplicatePair
            .similarity_score
            .desc()
        )
    )

    return (
        q.limit(limit)
        .all()
    )


# ============================================================
# Duplicate pair comparison
# ============================================================

@app.get(
    "/duplicates/"
    "{pair_id}/compare",
    response_model=(
        schemas.DuplicateCompareOut
    ),
)
def compare_duplicate_pair(
    pair_id: int,
    db: Session = Depends(get_db),
):
    """
    Display the complete report text
    for both reports in one detected
    duplicate pair.
    """

    pair = (
        db.query(
            models.DuplicatePair
        )
        .filter(
            models.DuplicatePair.id
            == pair_id
        )
        .first()
    )

    if not pair:

        raise HTTPException(
            status_code=404,
            detail=(
                "Duplicate pair "
                "not found"
            ),
        )

    report_1 = (
        db.query(
            models.BugReport
        )
        .filter(
            models.BugReport.id
            == pair.report_1_id
        )
        .first()
    )

    report_2 = (
        db.query(
            models.BugReport
        )
        .filter(
            models.BugReport.id
            == pair.report_2_id
        )
        .first()
    )

    if (
        not report_1
        or not report_2
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "One or both reports "
                "in this pair no "
                "longer exist."
            ),
        )

    return (
        schemas.DuplicateCompareOut(
            report_1=report_1,
            report_2=report_2,
            similarity_score=(
                pair.similarity_score
            ),
        )
    )


# ============================================================
# Evaluation
# ============================================================

@app.get(
    "/evaluation",
    response_model=(
        List[
            schemas.ModelEvalOut
        ]
    ),
)
def get_evaluation(
    db: Session = Depends(get_db),
):
    """
    Return final severity
    classification evaluation results.

    Models:
    - Naive Bayes
    - SVM
    - Random Forest
    - LLM Zero-Shot

    Metrics:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1

    Results are generated by:

        python -m app.evaluation
    """

    results = (
        db.query(
            models.ModelEvalResult
        )
        .all()
    )

    if not results:

        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation results "
                "available. Run "
                "`python -m "
                "app.evaluation` first."
            ),
        )

    return results


# ============================================================
# Evaluation detail
# ============================================================

@app.get(
    "/evaluation/"
    "{model_name}/detail"
)
def get_evaluation_detail(
    model_name: str,
    db: Session = Depends(get_db),
):
    """
    Return per-class:
    - precision
    - recall
    - F1-score
    - support

    for one evaluated model.
    """

    result = (
        db.query(
            models.ModelEvalResult
        )
        .filter(
            models.ModelEvalResult
            .model_name
            == model_name
        )
        .first()
    )

    if not result:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No evaluation "
                f"result for "
                f"'{model_name}'"
            ),
        )

    try:

        per_class = json.loads(
            result.per_class_report
        )

    except (
        TypeError,
        json.JSONDecodeError,
    ):

        per_class = {}

    return {
        "model_name": (
            result.model_name
        ),
        "accuracy": (
            result.accuracy
        ),
        "macro_precision": (
            result.macro_precision
        ),
        "macro_recall": (
            result.macro_recall
        ),
        "macro_f1": (
            result.macro_f1
        ),
        "per_class": (
            per_class
        ),
    }


# ============================================================
# Confusion matrix
# ============================================================

@app.get(
    "/evaluation/"
    "confusion-matrix"
)
def get_confusion_matrix():
    """
    Return the final LLM zero-shot
    confusion matrix.

    The file is generated by:

        python -m app.evaluation
    """

    path = (
        "llm_confusion_matrix.json"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail=(
                "Confusion matrix "
                "not found. Run "
                "`python -m "
                "app.evaluation` first."
            ),
        )

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read "
                "confusion matrix: "
                f"{error}"
            ),
        )

    return data


# ============================================================
# Duplicate evaluation
# ============================================================

@app.get(
    "/evaluation/duplicates"
)
def get_duplicate_evaluation():
    """
    Duplicate-detection precision,
    recall and F1 require labelled
    duplicate/non-duplicate ground
    truth pairs.

    These metrics are intentionally
    not fabricated.

    The current duplicate detector
    uses:
        all-MiniLM-L6-v2
        FAISS
        similarity threshold = 0.80
    """

    raise HTTPException(
        status_code=404,
        detail=(
            "Duplicate evaluation "
            "requires labelled "
            "ground-truth duplicate "
            "pairs. The current "
            "semantic similarity "
            "threshold is 0.80, but "
            "precision, recall and "
            "F1 cannot be calculated "
            "until labelled duplicate "
            "relationships are "
            "available."
        ),
    )


# ============================================================
# Statistics
# ============================================================

@app.get(
    "/stats",
    response_model=(
        schemas.StatsOut
    ),
)
def get_stats(
    db: Session = Depends(get_db),
):
    """
    Return the summary statistics
    displayed on the dashboard.
    """

    total = (
        db.query(
            models.BugReport
        )
        .count()
    )

    critical = (
        db.query(
            models.BugReport
        )
        .filter(
            models.BugReport
            .predicted_severity
            == "Critical"
        )
        .count()
    )

    dup_pairs = (
        db.query(
            models.DuplicatePair
        )
        .count()
    )

    breakdown = {}

    for severity in [
        "Critical",
        "High",
        "Medium",
        "Low",
    ]:

        breakdown[
            severity
        ] = (
            db.query(
                models.BugReport
            )
            .filter(
                models.BugReport
                .predicted_severity
                == severity
            )
            .count()
        )

    return schemas.StatsOut(
        total_reports=total,
        critical_count=critical,
        duplicates_found=(
            dup_pairs
        ),
        currently_showing=(
            total
        ),
        severity_breakdown=(
            breakdown
        ),
    )