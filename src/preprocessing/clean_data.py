import pandas as pd
import re
import os

INPUT_PATH = "data/raw/github_bug_reports.csv"
OUTPUT_PATH = "data/processed/cleaned_bug_reports.csv"


def clean_text(text):
    """
    Cleans a single piece of text:
    - removes code blocks (```...```)
    - removes HTML tags
    - removes markdown formatting (**bold**, _italic_, etc.)
    - removes URLs
    - collapses extra whitespace/newlines
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # Remove fenced code blocks ```like this```
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove inline code `like this`
    text = re.sub(r"`[^`]*`", " ", text)

    # Remove HTML tags like <div>, <br>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove markdown image/link syntax ![alt](url) and [text](url)
    text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", text)

    # Remove raw URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove markdown bold/italic symbols
    text = re.sub(r"[*_#>`~]", " ", text)

    # Collapse multiple newlines/spaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print(f"Loading raw data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    # Drop rows with no title at all (can't use these)
    df = df.dropna(subset=["title"])

    # Fill missing body text with empty string instead of dropping the row
    df["body"] = df["body"].fillna("")

    print("Cleaning title and body text...")
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_body"] = df["body"].apply(clean_text)

    # Combine title + body into one text field — this is what the model will read
    df["full_text"] = (df["clean_title"] + ". " + df["clean_body"]).str.strip()

    # Remove rows where the combined text ended up empty after cleaning
    before = len(df)
    df = df[df["full_text"].str.len() > 0]
    after = len(df)
    print(f"Removed {before - after} rows with empty text after cleaning.")

    # Remove exact duplicate reports (same cleaned text appearing twice)
    before = len(df)
    df = df.drop_duplicates(subset=["full_text"])
    after = len(df)
    print(f"Removed {before - after} exact duplicate reports.")

    # Keep only the columns we actually need going forward
    final_df = df[["id", "repo", "full_text", "labels", "state", "created_at", "closed_at"]]

    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(final_df)} cleaned bug reports to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
