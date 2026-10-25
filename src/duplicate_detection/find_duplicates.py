import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/processed/cleaned_bug_reports.csv"
OUTPUT_PATH = "data/processed/duplicate_pairs.csv"

# Similarity threshold: how close two reports must be (0 to 1) to count as duplicates.
# 1.0 = identical meaning, 0.0 = completely unrelated. 0.80 is a reasonably strict starting point.
SIMILARITY_THRESHOLD = 0.80


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df["full_text"] = df["full_text"].fillna("")
    print(f"Loaded {len(df)} bug reports.\n")

    print("Loading sentence embedding model...")
    print("(Downloads a small ~80MB model the first time — quick.)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    print("Generating embeddings for all bug reports...")
    embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize vectors so we can use inner product as cosine similarity
    faiss.normalize_L2(embeddings)

    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product = cosine similarity (since normalized)
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors.\n")

    print("Searching for duplicates (top 3 nearest neighbors per report)...")
    k = 3  # look at each report + its 2 closest neighbors
    similarities, indices = index.search(embeddings, k)

    duplicate_pairs = []
    seen_pairs = set()

    for i in range(len(df)):
        for rank in range(1, k):  # skip rank 0, since that's the report itself
            neighbor_idx = indices[i][rank]
            similarity = similarities[i][rank]

            if similarity >= SIMILARITY_THRESHOLD:
                pair_key = tuple(sorted([i, neighbor_idx]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                duplicate_pairs.append({
                    "report_1_id": df.iloc[i]["id"],
                    "report_1_text": df.iloc[i]["full_text"][:100],
                    "report_2_id": df.iloc[neighbor_idx]["id"],
                    "report_2_text": df.iloc[neighbor_idx]["full_text"][:100],
                    "similarity_score": round(float(similarity), 3)
                })

    result_df = pd.DataFrame(duplicate_pairs)
    result_df = result_df.sort_values("similarity_score", ascending=False)

    os.makedirs("data/processed", exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFound {len(result_df)} likely duplicate pairs (threshold: {SIMILARITY_THRESHOLD})")
    print(f"Saved to {OUTPUT_PATH}\n")

    if len(result_df) > 0:
        print("Top 5 most similar pairs found:")
        print("=" * 70)
        for _, row in result_df.head(5).iterrows():
            print(f"\nSimilarity: {row['similarity_score']}")
            print(f"  Report A: {row['report_1_text']}...")
            print(f"  Report B: {row['report_2_text']}...")
    else:
        print("No duplicates found above the threshold. Try lowering SIMILARITY_THRESHOLD.")


if __name__ == "__main__":
    main()
