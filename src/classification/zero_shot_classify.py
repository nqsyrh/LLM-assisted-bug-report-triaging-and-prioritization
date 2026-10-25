import pandas as pd
import os
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score

INPUT_PATH = "data/processed/labeled_training_data.csv"
OUTPUT_PATH = "data/processed/llm_predictions.csv"

# The severity categories your project uses
CANDIDATE_LABELS = ["Critical", "High", "Medium", "Low"]


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} labeled bug reports.")

    print("\nLoading pre-trained LLM for zero-shot classification...")
    print("(This downloads a ~1.6GB model the first time you run this — may take a few minutes.)")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    print("Model loaded.\n")

    predictions = []

    for i, row in df.iterrows():
        text = row["clean_text"]

        # Truncate very long text — the model only reads the first ~1000 characters efficiently
        text_for_model = text[:1000]

        result = classifier(text_for_model, CANDIDATE_LABELS)
        predicted_label = result["labels"][0]  # highest-scoring label
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

    # Evaluate performance against your real labels
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    accuracy = accuracy_score(results_df["mapped_severity"], results_df["predicted_severity"])
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")

    print("Detailed report (Precision / Recall / F1 per class):")
    print(classification_report(
        results_df["mapped_severity"],
        results_df["predicted_severity"],
        labels=CANDIDATE_LABELS,
        zero_division=0
    ))


if __name__ == "__main__":
    main()
