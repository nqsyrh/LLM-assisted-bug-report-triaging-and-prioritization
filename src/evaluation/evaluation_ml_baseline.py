import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

INPUT_PATH = "data/processed/labeled_training_data.csv"
LLM_RESULTS_PATH = "data/processed/llm_predictions_v2.csv"

LABEL_ORDER = ["Critical", "High", "Medium", "Low"]


def evaluate_model(name, model, X, y, cv):
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")

    # cross_val_predict trains/tests across folds and gives one prediction per row
    predictions = cross_val_predict(model, X, y, cv=cv)

    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy:.2%}\n")
    print(classification_report(y, predictions, labels=LABEL_ORDER, zero_division=0))

    return accuracy


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} labeled bug reports.\n")

    X_text = df["clean_text"]
    y = df["mapped_severity"]

    # Convert text into numeric features the way traditional ML expects (TF-IDF)
    print("Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(X_text)

    # 5-fold cross-validation (not stratified, since some classes have <5 examples)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    results["Naive Bayes"] = evaluate_model(
        "Naive Bayes", MultinomialNB(), X, y, cv
    )

    results["SVM"] = evaluate_model(
        "SVM (Support Vector Machine)",
        SVC(class_weight="balanced", kernel="linear"),
        X, y, cv
    )

    results["Random Forest"] = evaluate_model(
        "Random Forest",
        RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
        X, y, cv
    )

    # Final side-by-side comparison, including your LLM result
    print(f"\n{'='*50}")
    print("FINAL COMPARISON: LLM vs Traditional ML")
    print(f"{'='*50}\n")

    try:
        llm_df = pd.read_csv(LLM_RESULTS_PATH)
        llm_accuracy = accuracy_score(llm_df["mapped_severity"], llm_df["predicted_severity"])
        results["LLM (Zero-Shot, descriptive labels)"] = llm_accuracy
    except FileNotFoundError:
        print(f"Note: could not find {LLM_RESULTS_PATH} to include LLM comparison.")

    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:40s}: {acc:.2%}")


if __name__ == "__main__":
    main()
