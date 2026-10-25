import pandas as pd
import os
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score

INPUT_PATH = "data/processed/labeled_training_data.csv"
OUTPUT_PATH = "data/processed/llm_predictions_v2.csv"

# Descriptive labels instead of bare words — gives the model real context
# to work with instead of guessing what "Critical" or "Low" means on its own.
LABEL_DESCRIPTIONS = {
    "a critical bug causing system crash, data loss, or complete failure": "Critical",
    "a high severity bug causing major functionality to break or work incorrectly": "High",
    "a medium severity bug causing a minor feature to misbehave but the system still works": "Medium",
    "a low severity bug that is a small cosmetic or trivial issue": "Low",
}

CANDIDATE_LABELS = list(LABEL_DESCRIPTIONS.keys())


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} labeled bug reports.")

    print("\nLoading pre-trained LLM for zero-shot classification...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    print("Model loaded.\n")

    predictions = []

    for i, row in df.iterrows():
        text = row["clean_text"]
        text_for_model = text[:1000]

        result = classifier(text_for_model, CANDIDATE_LABELS)
        predicted_description = result["labels"][0]
        predicted_label = LABEL_DESCRIPTIONS[predicted_description]
        confidence = result["scores"][0]

        predictions.append({
            "predicted_severity": predicted_label,
            "confidence": round(confidence, 3)
        })

        print(f"[{i+1}/{len(df)}] Actual: {row['mapped_severity']:9s} | Predicted: {predicted_label:9s} | Confidence: {confidence:.2f}")

    pred_df = pd.DataFrame(predictions)
    results_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved predictions to {OUTPUT_PATH}")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (v2 - descriptive labels)")
    print("=" * 50)

    accuracy = accuracy_score(results_df["mapped_severity"], results_df["predicted_severity"])
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")

    print("Detailed report (Precision / Recall / F1 per class):")
    print(classification_report(
        results_df["mapped_severity"],
        results_df["predicted_severity"],
        labels=["Critical", "High", "Medium", "Low"],
        zero_division=0
    ))


if __name__ == "__main__":
    main()
