"""
seed_data.py
One-time load of your EXISTING computed results (prioritized_bug_reports.csv,
duplicate_pairs.csv, llm_predictions_v2.csv) into SQLite.

Deliberately does NOT recompute severity/priority/duplicates — you already
ran those pipelines and got results that match your dashboard screenshots
(499 reports, 4 critical, 16 duplicate-flagged, 33 pairs). Re-deriving them
here would risk producing slightly different numbers (zero-shot models can
have minor nondeterminism) and breaking that consistency.

Run with: python -m app.seed_data
Expects these files in backend/data/ (copy them there before running):
  - prioritized_bug_reports.csv
  - duplicate_pairs.csv
  - llm_predictions_v2.csv
"""
import os
import pandas as pd

from .database import engine, SessionLocal, Base
from . import models

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # ---------- Bug reports (prioritized_bug_reports.csv) ----------
        prioritized_path = os.path.join(DATA_DIR, "prioritized_bug_reports.csv")
        if os.path.exists(prioritized_path):
            df = pd.read_csv(prioritized_path)
            existing_ids = {r.id for r in db.query(models.BugReport.id).all()}

            inserted = 0
            for _, row in df.iterrows():
                if int(row["id"]) in existing_ids:
                    continue
                report = models.BugReport(
                    id=int(row["id"]),
                    repo=row["repo"],
                    full_text=row["full_text"],
                    predicted_severity=row["predicted_severity"],
                    model_confidence=float(row["model_confidence"]) if pd.notna(row.get("model_confidence")) else None,
                    is_duplicate=bool(row["is_duplicate"]),
                    priority_score=float(row["priority_score"]),
                )
                db.add(report)
                inserted += 1
            db.commit()
            print(f"Seeded {inserted} bug reports from prioritized_bug_reports.csv "
                  f"({len(df) - inserted} already present, skipped).")
        else:
            print(f"WARNING: {prioritized_path} not found — skipping bug report seed.")

        # ---------- Duplicate pairs (duplicate_pairs.csv) ----------
        dup_path = os.path.join(DATA_DIR, "duplicate_pairs.csv")
        if os.path.exists(dup_path):
            df = pd.read_csv(dup_path)
            existing_pairs = {
                (p.report_1_id, p.report_2_id) for p in db.query(
                    models.DuplicatePair.report_1_id, models.DuplicatePair.report_2_id
                ).all()
            }

            inserted = 0
            for _, row in df.iterrows():
                key = (int(row["report_1_id"]), int(row["report_2_id"]))
                if key in existing_pairs:
                    continue
                pair = models.DuplicatePair(
                    report_1_id=int(row["report_1_id"]),
                    report_1_text=row["report_1_text"],
                    report_2_id=int(row["report_2_id"]),
                    report_2_text=row["report_2_text"],
                    similarity_score=float(row["similarity_score"]),
                )
                db.add(pair)
                inserted += 1
            db.commit()
            print(f"Seeded {inserted} duplicate pairs from duplicate_pairs.csv.")
        else:
            print(f"WARNING: {dup_path} not found — skipping duplicate pair seed.")

        # ---------- Evaluation examples (llm_predictions_v2.csv) ----------
        eval_path = os.path.join(DATA_DIR, "llm_predictions_v2.csv")
        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path)
            existing_eval_ids = {e.id for e in db.query(models.EvalExample.id).all()}

            inserted = 0
            for _, row in df.iterrows():
                if int(row["ID"]) in existing_eval_ids:
                    continue
                ex = models.EvalExample(
                    id=int(row["ID"]),
                    clean_text=row["clean_text"],
                    mapped_severity=row["mapped_severity"],
                    component=row.get("Component"),
                    priority=row.get("Priority"),
                    predicted_severity=row.get("predicted_severity"),
                    confidence=float(row["confidence"]) if pd.notna(row.get("confidence")) else None,
                )
                db.add(ex)
                inserted += 1
            db.commit()
            print(f"Seeded {inserted} evaluation examples from llm_predictions_v2.csv.")
        else:
            print(f"WARNING: {eval_path} not found — skipping evaluation seed.")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
    print("\nSeed complete.")
