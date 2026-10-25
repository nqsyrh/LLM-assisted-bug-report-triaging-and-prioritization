import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Triage Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "data/processed/prioritized_bug_reports.csv"
DUPLICATES_PATH = "data/processed/duplicate_pairs.csv"

# ---------------------------------------------------------------------------
# CUSTOM STYLING — dark "dev console" theme with severity color coding
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

.console-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 0px;
}
.console-header h1 {
    font-size: 2.1rem;
    margin: 0;
    background: linear-gradient(90deg, #5EEAD4, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.console-sub {
    font-family: 'JetBrains Mono', monospace;
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 2px;
}

/* Metric cards */
.metric-card {
    background: #161B2E;
    border: 1px solid #232B45;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: left;
}
.metric-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    margin-top: 2px;
}

/* Ticket-style bug cards */
.ticket {
    background: #161B2E;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    border-left: 4px solid #5EEAD4;
}
.ticket-critical { border-left-color: #F87171; }
.ticket-high { border-left-color: #FB923C; }
.ticket-medium { border-left-color: #FBBF24; }
.ticket-low { border-left-color: #4ADE80; }

.ticket-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.badge-critical { background: rgba(248,113,113,0.15); color: #F87171; }
.badge-high { background: rgba(251,146,60,0.15); color: #FB923C; }
.badge-medium { background: rgba(251,191,36,0.15); color: #FBBF24; }
.badge-low { background: rgba(74,222,128,0.15); color: #4ADE80; }

.dup-flag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #818CF8;
}
.ticket-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #B6C0D8;
    line-height: 1.4;
}
.priority-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748B;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid #232B45;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_duplicates():
    if os.path.exists(DUPLICATES_PATH):
        return pd.read_csv(DUPLICATES_PATH)
    return pd.DataFrame()


SEVERITY_BADGE = {
    "Critical": ("badge-critical", "ticket-critical", "🔴"),
    "High": ("badge-high", "ticket-high", "🟠"),
    "Medium": ("badge-medium", "ticket-medium", "🟡"),
    "Low": ("badge-low", "ticket-low", "🟢"),
}


def render_ticket(row):
    badge_class, ticket_class, dot = SEVERITY_BADGE.get(row["predicted_severity"], ("badge-medium", "ticket-medium", "⚪"))
    dup_html = '<span class="dup-flag">⛓ DUPLICATE</span>' if row["is_duplicate"] else ""
    text_preview = str(row["full_text"])[:160].replace("<", "").replace(">", "")

    st.markdown(f"""
    <div class="ticket {ticket_class}">
        <div class="ticket-top">
            <span class="badge {badge_class}">{dot} {row['predicted_severity']}</span>
            {dup_html}
        </div>
        <div class="ticket-text">{text_preview}...</div>
        <div class="priority-score">priority score → {row['priority_score']}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="console-header">
    <h1>🛰️ Triage Console</h1>
</div>
<div class="console-sub">LLM-ASSISTED BUG REPORT TRIAGING &amp; PRIORITIZATION SYSTEM</div>
""", unsafe_allow_html=True)
st.write("")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run the priority scoring script first.")
    st.stop()

duplicates_df = load_duplicates()

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙ Filters")

severity_options = ["Critical", "High", "Medium", "Low"]
selected_severities = st.sidebar.multiselect("Severity", severity_options, default=severity_options)

show_duplicates_only = st.sidebar.checkbox("Duplicates only", value=False)
search_term = st.sidebar.text_input("Search report text", "")

repo_options = sorted(df["repo"].dropna().unique().tolist())
selected_repos = st.sidebar.multiselect("Repository", repo_options, default=repo_options)

filtered_df = df[df["predicted_severity"].isin(selected_severities)]
filtered_df = filtered_df[filtered_df["repo"].isin(selected_repos)]
if show_duplicates_only:
    filtered_df = filtered_df[filtered_df["is_duplicate"] == True]
if search_term:
    filtered_df = filtered_df[filtered_df["full_text"].str.contains(search_term, case=False, na=False)]

# ---------------------------------------------------------------------------
# METRIC ROW
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "Total Reports", len(df)),
    (m2, "Critical", len(df[df["predicted_severity"] == "Critical"])),
    (m3, "Duplicates Found", int(df["is_duplicate"].sum())),
    (m4, "Currently Showing", len(filtered_df)),
]
for col, label, value in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 All Reports", "🔁 Duplicates"])

with tab1:
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("##### Severity Breakdown")
        severity_counts = df["predicted_severity"].value_counts().reindex(severity_options).fillna(0)
        st.bar_chart(severity_counts, color="#5EEAD4")

    with right:
        st.markdown("##### Top Priority Tickets")
        top_10 = df.sort_values("priority_score", ascending=False).head(8)
        for _, row in top_10.iterrows():
            render_ticket(row)

with tab2:
    st.markdown(f"##### {len(filtered_df)} reports match your filters")
    display_df = filtered_df[["repo", "predicted_severity", "is_duplicate", "priority_score", "full_text"]].copy()
    display_df.columns = ["Repository", "Severity", "Duplicate?", "Priority", "Bug Report Text"]
    st.dataframe(
        display_df.sort_values("Priority", ascending=False),
        use_container_width=True,
        height=520
    )

with tab3:
    if not duplicates_df.empty:
        st.markdown(f"##### {len(duplicates_df)} duplicate pairs detected via semantic similarity")
        st.dataframe(
            duplicates_df.sort_values("similarity_score", ascending=False),
            use_container_width=True,
            height=480
        )
    else:
        st.info("No duplicate pairs file found yet — run the duplicate detection script first.")
