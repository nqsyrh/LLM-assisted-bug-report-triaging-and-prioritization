import streamlit as st
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score

st.set_page_config(page_title="Triage Console", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

DATA_PATH = "data/processed/prioritized_bug_reports.csv"
DUPLICATES_PATH = "data/processed/duplicate_pairs.csv"
LLM_RESULTS_PATH = "data/processed/llm_predictions_v2.csv"

# Static results from your earlier traditional ML baseline run (Step 9).
# These came from real cross-validation output, not invented numbers.
ML_BASELINE_RESULTS = {
    "Naive Bayes":    {"accuracy": 0.8592, "macro_f1": 0.23},
    "SVM":            {"accuracy": 0.8169, "macro_f1": 0.22},
    "Random Forest":  {"accuracy": 0.8592, "macro_f1": 0.23},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

.console-header h1 {
    font-size: 2.1rem; margin: 0;
    background: linear-gradient(90deg, #5EEAD4, #818CF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.console-sub { font-family: 'JetBrains Mono', monospace; color: #64748B; font-size: 0.85rem; margin-top: 2px; }

.metric-card { background: #161B2E; border: 1px solid #232B45; border-radius: 10px; padding: 16px 18px; }
.metric-card .label { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; margin-top: 2px; }

.ticket { background: #161B2E; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; border-left: 4px solid #5EEAD4; }
.ticket-critical { border-left-color: #F87171; }
.ticket-high { border-left-color: #FB923C; }
.ticket-medium { border-left-color: #FBBF24; }
.ticket-low { border-left-color: #4ADE80; }
.ticket-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.badge { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600; padding: 2px 9px; border-radius: 5px; text-transform: uppercase; letter-spacing: 0.03em; }
.badge-critical { background: rgba(248,113,113,0.15); color: #F87171; }
.badge-high { background: rgba(251,146,60,0.15); color: #FB923C; }
.badge-medium { background: rgba(251,191,36,0.15); color: #FBBF24; }
.badge-low { background: rgba(74,222,128,0.15); color: #4ADE80; }
.dup-flag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #818CF8; }
.ticket-text { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #B6C0D8; line-height: 1.4; }
.meta-row { display: flex; gap: 14px; margin-top: 6px; }
.priority-score, .confidence-score { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #64748B; }
.confidence-score { color: #5EEAD4; }

.eval-card { background: #161B2E; border: 1px solid #232B45; border-radius: 10px; padding: 18px 20px; }
.eval-model-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem; }
.eval-bar-bg { background: #0B1120; border-radius: 6px; height: 8px; margin-top: 6px; overflow: hidden; }
.eval-bar-fill { height: 8px; border-radius: 6px; }

section[data-testid="stSidebar"] { border-right: 1px solid #232B45; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_duplicates():
    return pd.read_csv(DUPLICATES_PATH) if os.path.exists(DUPLICATES_PATH) else pd.DataFrame()


@st.cache_data
def load_llm_eval():
    return pd.read_csv(LLM_RESULTS_PATH) if os.path.exists(LLM_RESULTS_PATH) else pd.DataFrame()


SEVERITY_BADGE = {
    "Critical": ("badge-critical", "ticket-critical", "🔴"),
    "High": ("badge-high", "ticket-high", "🟠"),
    "Medium": ("badge-medium", "ticket-medium", "🟡"),
    "Low": ("badge-low", "ticket-low", "🟢"),
}


def render_ticket(row, has_confidence):
    badge_class, ticket_class, dot = SEVERITY_BADGE.get(row["predicted_severity"], ("badge-medium", "ticket-medium", "⚪"))
    dup_html = '<span class="dup-flag">⛓ DUPLICATE</span>' if row["is_duplicate"] else ""
    text_preview = str(row["full_text"])[:160].replace("<", "").replace(">", "")
    conf_html = f'<span class="confidence-score">confidence → {row["model_confidence"]:.0%}</span>' if has_confidence else ""

    st.markdown(f"""
    <div class="ticket {ticket_class}">
        <div class="ticket-top">
            <span class="badge {badge_class}">{dot} {row['predicted_severity']}</span>
            {dup_html}
        </div>
        <div class="ticket-text">{text_preview}...</div>
        <div class="meta-row">
            <span class="priority-score">priority → {row['priority_score']}</span>
            {conf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="console-header"><h1>🛰️ Triage Console</h1></div>
<div class="console-sub">LLM-ASSISTED BUG REPORT TRIAGING &amp; PRIORITIZATION SYSTEM</div>
""", unsafe_allow_html=True)
st.write("")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run the priority scoring script first.")
    st.stop()

has_confidence = "model_confidence" in df.columns
duplicates_df = load_duplicates()
llm_eval_df = load_llm_eval()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("### ⚙ Filters")
severity_options = ["Critical", "High", "Medium", "Low"]
selected_severities = st.sidebar.multiselect("Severity", severity_options, default=severity_options)
show_duplicates_only = st.sidebar.checkbox("Duplicates only", value=False)
search_term = st.sidebar.text_input("Search by keyword or bug ID", "")
repo_options = sorted(df["repo"].dropna().unique().tolist())
selected_repos = st.sidebar.multiselect("Repository", repo_options, default=repo_options)

filtered_df = df[df["predicted_severity"].isin(selected_severities)]
filtered_df = filtered_df[filtered_df["repo"].isin(selected_repos)]
if show_duplicates_only:
    filtered_df = filtered_df[filtered_df["is_duplicate"] == True]
if search_term:
    mask = filtered_df["full_text"].str.contains(search_term, case=False, na=False)
    if filtered_df["id"].astype(str).str.contains(search_term).any():
        mask = mask | filtered_df["id"].astype(str).str.contains(search_term)
    filtered_df = filtered_df[mask]

# ---------------- METRICS ----------------
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "Total Reports", len(df)),
    (m2, "Critical", len(df[df["predicted_severity"] == "Critical"])),
    (m3, "Duplicates Found", int(df["is_duplicate"].sum())),
    (m4, "Currently Showing", len(filtered_df)),
]
for col, label, value in metrics:
    with col:
        st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 All Reports", "🔁 Duplicates", "🧪 Evaluation"])

with tab1:
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("##### Severity Breakdown")
        severity_counts = df["predicted_severity"].value_counts().reindex(severity_options).fillna(0)
        st.bar_chart(severity_counts, color="#5EEAD4")
        if not has_confidence:
            st.caption("ℹ️ Re-run `priority_scoring_v2.py` to enable confidence scores on tickets.")
    with right:
        st.markdown("##### Top Priority Tickets")
        top_8 = df.sort_values("priority_score", ascending=False).head(8)
        for _, row in top_8.iterrows():
            render_ticket(row, has_confidence)

with tab2:
    st.markdown(f"##### {len(filtered_df)} reports match your filters")
    cols = ["repo", "predicted_severity", "is_duplicate", "priority_score", "full_text"]
    names = ["Repository", "Severity", "Duplicate?", "Priority", "Bug Report Text"]
    if has_confidence:
        cols.insert(3, "model_confidence")
        names.insert(3, "Confidence")
    display_df = filtered_df[cols].copy()
    display_df.columns = names
    st.dataframe(display_df.sort_values("Priority", ascending=False), use_container_width=True, height=520)

with tab3:
    if not duplicates_df.empty:
        st.markdown(f"##### {len(duplicates_df)} duplicate pairs detected via semantic similarity")
        st.caption("Each pair shows the matched report and how similar the system judged them to be (1.0 = identical meaning).")
        st.dataframe(duplicates_df.sort_values("similarity_score", ascending=False), use_container_width=True, height=480)
    else:
        st.info("No duplicate pairs file found yet — run the duplicate detection script first.")

with tab4:
    st.markdown("##### Model Performance Comparison")
    st.caption("Severity classification evaluated on the labeled Eclipse dataset (Zenodo). See Step 9 results for full methodology.")

    if not llm_eval_df.empty:
        llm_acc = accuracy_score(llm_eval_df["mapped_severity"], llm_eval_df["predicted_severity"])
        llm_f1 = f1_score(llm_eval_df["mapped_severity"], llm_eval_df["predicted_severity"],
                           labels=severity_options, average="macro", zero_division=0)
    else:
        llm_acc, llm_f1 = 0.5634, 0.22

    all_results = {"LLM (Zero-Shot)": {"accuracy": llm_acc, "macro_f1": llm_f1}}
    all_results.update(ML_BASELINE_RESULTS)

    cols = st.columns(len(all_results))
    for col, (name, scores) in zip(cols, all_results.items()):
        with col:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-model-name">{name}</div>
                <div style="margin-top:10px; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748B;">ACCURACY</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700;">{scores['accuracy']:.1%}</div>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:{scores['accuracy']*100}%; background:#5EEAD4;"></div></div>
                <div style="margin-top:14px; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748B;">MACRO F1-SCORE</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700;">{scores['macro_f1']:.2f}</div>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:{scores['macro_f1']*100}%; background:#818CF8;"></div></div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.warning(
        "⚠️ **Reading these numbers correctly:** raw accuracy looks higher for traditional ML models, "
        "but this is misleading — they achieve it by always predicting the majority class ('Medium'), "
        "scoring 0% precision/recall on Critical, High, and Low bugs. Macro F1-score (which treats all "
        "severity classes equally) shows all approaches performed comparably (~0.22–0.23), revealing that "
        "dataset size and class imbalance — not model choice — was the main limiting factor.",
        icon="⚠️"
    )

    st.markdown("""
    ##### Known limitations (documented for thesis transparency)
    - Defect *category* classification (UI / Backend / Performance / Security) is not yet implemented — only severity is currently classified. Planned for FYP2.
    - Evaluation set size (71 labeled examples) is small, with severe class imbalance (61 Medium vs 1 Low) — results should be read as a proof-of-concept, not a final benchmark.
    """)
