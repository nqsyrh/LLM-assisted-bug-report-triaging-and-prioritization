import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Bug Triaging Dashboard",
    page_icon="🐛",
    layout="wide"
)

DATA_PATH = "data/processed/prioritized_bug_reports.csv"
DUPLICATES_PATH = "data/processed/duplicate_pairs.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_duplicates():
    if os.path.exists(DUPLICATES_PATH):
        return pd.read_csv(DUPLICATES_PATH)
    return pd.DataFrame()


st.title("🐛 LLM-Assisted Bug Report Triaging Dashboard")
st.caption("Automatically classified, prioritized, and deduplicated bug reports")

# Load the data
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run the priority scoring script first.")
    st.stop()

duplicates_df = load_duplicates()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")

severity_options = ["Critical", "High", "Medium", "Low"]
selected_severities = st.sidebar.multiselect(
    "Severity Level",
    options=severity_options,
    default=severity_options
)

show_duplicates_only = st.sidebar.checkbox("Show only duplicates", value=False)

search_term = st.sidebar.text_input("Search bug text", "")

repo_options = sorted(df["repo"].dropna().unique().tolist())
selected_repos = st.sidebar.multiselect(
    "Repository",
    options=repo_options,
    default=repo_options
)

# Apply filters
filtered_df = df[df["predicted_severity"].isin(selected_severities)]
filtered_df = filtered_df[filtered_df["repo"].isin(selected_repos)]

if show_duplicates_only:
    filtered_df = filtered_df[filtered_df["is_duplicate"] == True]

if search_term:
    filtered_df = filtered_df[filtered_df["full_text"].str.contains(search_term, case=False, na=False)]

# ---------- TOP METRICS ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Bug Reports", len(df))

with col2:
    critical_count = len(df[df["predicted_severity"] == "Critical"])
    st.metric("Critical Bugs", critical_count)

with col3:
    duplicate_count = int(df["is_duplicate"].sum())
    st.metric("Duplicates Detected", duplicate_count)

with col4:
    st.metric("Showing (filtered)", len(filtered_df))

st.divider()

# ---------- SEVERITY DISTRIBUTION CHART ----------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Severity Distribution")
    severity_counts = df["predicted_severity"].value_counts()
    st.bar_chart(severity_counts)

with right_col:
    st.subheader("Top 10 Highest Priority Bugs")
    top_10 = df.sort_values("priority_score", ascending=False).head(10)
    for _, row in top_10.iterrows():
        dup_label = "🔁 DUPLICATE" if row["is_duplicate"] else ""
        severity_color = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢"
        }.get(row["predicted_severity"], "⚪")

        st.markdown(f"{severity_color} **{row['predicted_severity']}** (score: {row['priority_score']}) {dup_label}")
        st.caption(row["full_text"][:150] + "...")
        st.markdown("---")

st.divider()

# ---------- MAIN TABLE ----------
st.subheader(f"All Bug Reports ({len(filtered_df)} shown)")

display_df = filtered_df[["repo", "predicted_severity", "is_duplicate", "priority_score", "full_text"]].copy()
display_df.columns = ["Repository", "Severity", "Duplicate?", "Priority Score", "Bug Report Text"]

st.dataframe(
    display_df.sort_values("Priority Score", ascending=False),
    use_container_width=True,
    height=500
)

# ---------- DUPLICATE PAIRS SECTION ----------
if not duplicates_df.empty:
    st.divider()
    st.subheader(f"Detected Duplicate Pairs ({len(duplicates_df)} pairs)")
    st.dataframe(
        duplicates_df.sort_values("similarity_score", ascending=False),
        use_container_width=True,
        height=300
    )
