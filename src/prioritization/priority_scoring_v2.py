import pandas as pd
import os
from transformers import pipeline

CLEANED_DATA_PATH = "data/processed/cleaned_bug_reports.csv"
DUPLICATES_PATH = "data/processed/duplicate_pairs.csv"
OUTPUT_PATH = "data/processed/prioritized_bug_reports.csv"

LABEL_DESCRIPTIONS = {
    "a critical bug causing system crash, data loss, or complete failure": "Critical",
    "a high severity bug causing major functionality to break or work incorrectly": "High",
    "a medium severity bug causing a minor feature to misbehave but the system still works": "Medium",
    "a low severity bug that is a small cosmetic or trivial issue": "Low",
}
CANDIDATE_LABELS = list(LABEL_DESCRIPTIONS.keys())

SEVERITY_SCORE = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
DUPLICATE_PENALTY = 0.5


def main():
    print(f"Loading {CLEANED_DATA_PATH}...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["full_text"] = df["full_text"].fillna("")
    print(f"Loaded {len(df)} bug reports.\n")

    duplicate_ids = set()
    if os.path.exists(DUPLICATES_PATH):
        dup_df = pd.read_csv(DUPLICATES_PATH)
        duplicate_ids = set(dup_df["report_2_id"].tolist())
        print(f"Loaded {len(duplicate_ids)} reports flagged as duplicates.\n")

    print("Loading pre-trained LLM for severity classification...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Model loaded.\n")

    print(f"Classifying severity for all {len(df)} reports — this will take a while, please be patient...\n")

    severities = []
    confidences = []

    for i, row in df.iterrows():
        text_for_model = row["full_text"][:1000]

        if not text_for_model.strip():
            severities.append("Medium")
            confidences.append(0.0)
            continue

        result = classifier(text_for_model, CANDIDATE_LABELS)
        predicted_description = result["labels"][0]
        severities.append(LABEL_DESCRIPTIONS[predicted_description])
        confidences.append(round(float(result["scores"][0]), 3))

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"  Processed {i+1}/{len(df)} reports...")

    df["predicted_severity"] = severities
    df["model_confidence"] = confidences
    df["is_duplicate"] = df["id"].isin(duplicate_ids)

    def calculate_priority(row):
        base_score = SEVERITY_SCORE.get(row["predicted_severity"], 2)
        if row["is_duplicate"]:
            return base_score * DUPLICATE_PENALTY
        return base_score

    df["priority_score"] = df.apply(calculate_priority, axis=1)
    df = df.sort_values("priority_score", ascending=False)

    output_columns = ["id", "repo", "full_text", "predicted_severity", "model_confidence", "is_duplicate", "priority_score"]
    final_df = df[output_columns]
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone! Saved prioritized list (with confidence scores) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
