"""
models.py
SQLAlchemy ORM models. Column names deliberately mirror your existing
CSV schemas (prioritized_bug_reports.csv, duplicate_pairs.csv,
llm_predictions_v2.csv) so nothing gets renamed or reinterpreted
between your thesis scripts and this API layer.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base


class BugReport(Base):
    """
    Mirrors prioritized_bug_reports.csv (which itself extends
    cleaned_bug_reports.csv with predicted_severity / model_confidence /
    is_duplicate / priority_score).
    """
    __tablename__ = "bug_reports"

    # GitHub issue IDs are large ints (e.g. 4398280052) — not the SQLite
    # autoincrement PK, but the real-world bug ID, so it stays the primary key.
    id = Column(Integer, primary_key=True, index=True)
    repo = Column(String, index=True)
    full_text = Column(Text)
    state = Column(String, nullable=True)
    created_at = Column(String, nullable=True)
    closed_at = Column(String, nullable=True)

    # populated by zero_shot_classify (facebook/bart-large-mnli)
    predicted_severity = Column(String, index=True, nullable=True)
    model_confidence = Column(Float, nullable=True)

    # populated by find_duplicates.py / cross-referenced against DuplicatePair
    is_duplicate = Column(Boolean, default=False, index=True)

    # populated by priority_scoring.py: SEVERITY_SCORE[severity], halved if duplicate
    priority_score = Column(Float, index=True, nullable=True)

    duplicate_pairs_as_first = relationship(
        "DuplicatePair", foreign_keys="DuplicatePair.report_1_id", back_populates="report_1"
    )
    duplicate_pairs_as_second = relationship(
        "DuplicatePair", foreign_keys="DuplicatePair.report_2_id", back_populates="report_2"
    )


class DuplicatePair(Base):
    """Mirrors duplicate_pairs.csv exactly — output of find_duplicates.py (FAISS + MiniLM)."""
    __tablename__ = "duplicate_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_1_id = Column(Integer, ForeignKey("bug_reports.id"), index=True)
    report_1_text = Column(Text)
    report_2_id = Column(Integer, ForeignKey("bug_reports.id"), index=True)
    report_2_text = Column(Text)
    similarity_score = Column(Float, index=True)

    report_1 = relationship("BugReport", foreign_keys=[report_1_id], back_populates="duplicate_pairs_as_first")
    report_2 = relationship("BugReport", foreign_keys=[report_2_id], back_populates="duplicate_pairs_as_second")


class EvalExample(Base):
    """
    Mirrors labeled_training_data.csv / llm_predictions_v2.csv — the 71
    ground-truth-labeled Eclipse bugs used by evaluation_ml_baseline.py.
    Kept separate from BugReport since this is a different dataset
    (Eclipse, labeled) from the live triage set (VS Code, unlabeled).
    """
    __tablename__ = "eval_examples"

    id = Column(Integer, primary_key=True, index=True)  # the Eclipse bug ID
    clean_text = Column(Text)
    mapped_severity = Column(String, index=True)   # ground truth
    component = Column(String, nullable=True)
    priority = Column(String, nullable=True)        # Eclipse's own P1-P5, NOT priority_score
    predicted_severity = Column(String, nullable=True)  # LLM zero-shot prediction
    confidence = Column(Float, nullable=True)


class ModelEvalResult(Base):
    """
    Stores the precomputed accuracy/precision/recall/F1 per model
    (LLM, Naive Bayes, SVM, Random Forest) so the Evaluation tab doesn't
    need to re-run 5-fold CV on every page load.
    """
    __tablename__ = "model_eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, unique=True, index=True)
    accuracy = Column(Float)
    macro_precision = Column(Float)
    macro_recall = Column(Float)
    macro_f1 = Column(Float)
    # per-class breakdown stored as JSON-ish text for simplicity (SQLite has no native JSON requirement)
    per_class_report = Column(Text)
