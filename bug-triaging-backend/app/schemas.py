"""
schemas.py
Pydantic request/response models. Separate from the SQLAlchemy ORM models
so the API contract (what Postman/the dashboard sees) is explicit and
doesn't silently change if the DB schema changes.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


# ---------- Bug Reports ----------

class BugReportBase(BaseModel):
    repo: str
    full_text: str


class BugReportCreate(BugReportBase):
    """Body for POST /reports — submitting a brand new bug report for live classification."""
    id: Optional[int] = None  # if omitted, server assigns one


class BugReportOut(BugReportBase):
    id: int
    state: Optional[str] = None
    created_at: Optional[str] = None
    closed_at: Optional[str] = None
    predicted_severity: Optional[str] = None
    model_confidence: Optional[float] = None
    is_duplicate: bool = False
    priority_score: Optional[float] = None

    class Config:
        from_attributes = True


class ClassifyResponse(BaseModel):
    """What POST /reports returns — the live classification result."""
    id: int
    predicted_severity: str
    model_confidence: float
    is_duplicate: bool
    matched_duplicate_id: Optional[int] = None
    similarity_score: Optional[float] = None
    priority_score: float


# ---------- Duplicates ----------

class DuplicatePairOut(BaseModel):
    report_1_id: int
    report_1_text: str
    report_2_id: int
    report_2_text: str
    similarity_score: float

    class Config:
        from_attributes = True


class DuplicateCompareOut(BaseModel):
    """Full side-by-side comparison — matches your dashboard's expandable 'Reference comparison' rows."""
    report_1: BugReportOut
    report_2: BugReportOut
    similarity_score: float


# ---------- Evaluation ----------

class ModelEvalOut(BaseModel):
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    class Config:
        from_attributes = True


# ---------- Stats ----------

class StatsOut(BaseModel):
    total_reports: int
    critical_count: int
    duplicates_found: int
    currently_showing: int
    severity_breakdown: dict
