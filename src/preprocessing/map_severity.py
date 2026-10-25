import pandas as pd
import re
import os

INPUT_PATH = "data/raw/eclipse_labeled_bugs.csv"
OUTPUT_PATH = "data/processed/labeled_training_data.csv"

# Maps Bugzilla's raw severity values into the 4 categories your project uses
SEVERITY_MAP = {
    "blocker": "Critical",
    "critical": "Critical",
    "major": "High",
    "normal": "Medium",
    "minor": "Low",
    "trivial": "Low",
}


def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[*_#>`~]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    # Drop feature requests — we only want actual bugs
    before = len(df)
    df = df[df["Severity"].str.lower() != "enhancement"]
    print(f"Removed {before - len(df)} 'enhancement' rows (not real bugs).")

    # Map severity to our 4 categories
    df["mapped_severity"] = df["Severity"].str.lower().map(SEVERITY_MAP)

    # Drop any rows whose severity didn't match our map (just in case)
    before = len(df)
    df = df.dropna(subset=["mapped_severity"])
    print(f"Removed {before - len(df)} rows with unrecognized severity values.")

    # Build the combined text field
    df["Summary"] = df["Summary"].fillna("")
    df["Description"] = df["Description"].fillna("")
    df["clean_text"] = (df["Summary"].apply(clean_text) + ". " + df["Description"].apply(clean_text)).str.strip()

    # Remove rows with empty text
    before = len(df)
    df = df[df["clean_text"].str.len() > 0]
    print(f"Removed {before - len(df)} rows with empty text.")

    # Keep only what we need: text + severity label (+ Component as a bonus defect-type proxy)
    final_df = df[["ID", "clean_text", "mapped_severity", "Component", "Priority"]]

    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone! Saved {len(final_df)} labeled bug reports to {OUTPUT_PATH}")
    print("\nFinal severity distribution:")
    print(final_df["mapped_severity"].value_counts())


if __name__ == "__main__":
    main()
