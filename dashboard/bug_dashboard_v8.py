import html

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Triage Console",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# FastAPI backend used by v7. Keep the backend running with:
#   cd bug-triaging-backend
#   uvicorn app.main:app --reload --port 8000
API_BASE = "http://127.0.0.1:8000"

# Fallback values from the traditional ML baseline run.
ML_BASELINE_RESULTS = {
    "Naive Bayes": {
        "accuracy": 0.8592,
        "macro_f1": 0.23,
        "macro_precision": 0.21,
        "macro_recall": 0.25,
    },
    "SVM": {
        "accuracy": 0.8169,
        "macro_f1": 0.22,
        "macro_precision": 0.21,
        "macro_recall": 0.24,
    },
    "Random Forest": {
        "accuracy": 0.8592,
        "macro_f1": 0.23,
        "macro_precision": 0.21,
        "macro_recall": 0.25,
    },
}

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_COLORS = {
    "Critical": "#DC2626",
    "High": "#EA580C",
    "Medium": "#2563EB",
    "Low": "#16A34A",
}


# ============================================================
# UI theme — v8 presentation pass
# ============================================================

st.markdown(
    """
<style>
:root {
    --bg: #F7F8FA;
    --surface: #FFFFFF;
    --surface-soft: #F9FAFB;
    --text: #111827;
    --muted: #667085;
    --border: #E4E7EC;
    --border-strong: #D0D5DD;
    --accent: #4F46E5;
    --accent-hover: #4338CA;
    --success: #15803D;
    --success-bg: #F0FDF4;
    --danger: #B42318;
    --danger-bg: #FEF3F2;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1520px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
}

/* Keep Streamlit usable, but remove presentation chrome such as Deploy. */
[data-testid="stToolbar"] {
    visibility: hidden !important;
}
#MainMenu, footer {
    visibility: hidden !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

p, label, .stCaption {
    color: var(--muted);
}

/* ---------- Header ---------- */
.app-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 22px;
}

.app-title {
    color: var(--text);
    font-size: 2rem;
    line-height: 1.15;
    font-weight: 700;
    letter-spacing: -0.035em;
    margin: 0;
}

.app-subtitle {
    color: var(--muted);
    font-size: 0.92rem;
    margin-top: 6px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
    border: 1px solid #BBF7D0;
    background: var(--success-bg);
    color: #166534;
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22C55E;
    display: inline-block;
}

.status-pill.offline {
    border-color: #FECDCA;
    background: var(--danger-bg);
    color: var(--danger);
}
.status-pill.offline .status-dot { background: #F04438; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: var(--surface) !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 1rem;
    font-weight: 650;
    margin-bottom: 1rem;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #344054;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] input {
    background: #FFFFFF !important;
    color: #101828 !important;
    border-color: var(--border-strong) !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background: #EEF2FF !important;
    color: #3730A3 !important;
    border-radius: 5px !important;
}
section[data-testid="stSidebar"] span[data-baseweb="tag"] span {
    color: #3730A3 !important;
}

/* ---------- Inputs ---------- */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #101828 !important;
    border-color: var(--border-strong) !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.10) !important;
}

/* ---------- Metric cards ---------- */
.metric-card {
    min-height: 108px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
}
.metric-label {
    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-value {
    color: var(--text);
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.2;
    margin-top: 8px;
}
.metric-hint {
    color: #98A2B3;
    font-size: 0.75rem;
    margin-top: 5px;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 2px;
    background: transparent;
    border-radius: 0;
    color: #667085;
    font-weight: 550;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #344054;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
    height: 2px;
}

/* ---------- Section headers ---------- */
.section-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 650;
    margin: 8px 0 4px 0;
}
.section-subtitle {
    color: var(--muted);
    font-size: 0.82rem;
    margin-bottom: 14px;
}

/* ---------- Ticket rows ---------- */
.ticket {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid #98A2B3;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 9px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.02);
}
.ticket-critical { border-left-color: #DC2626; }
.ticket-high { border-left-color: #EA580C; }
.ticket-medium { border-left-color: #2563EB; }
.ticket-low { border-left-color: #16A34A; }
.ticket-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.ticket-id {
    color: #667085;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.75rem;
}
.ticket-text {
    color: #1D2939;
    font-size: 0.88rem;
    line-height: 1.5;
    font-weight: 500;
}
.ticket-meta {
    color: #667085;
    font-size: 0.76rem;
    margin-top: 9px;
}
.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 5px;
    padding: 3px 7px;
    font-size: 0.68rem;
    font-weight: 650;
    line-height: 1.2;
}
.badge-critical { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
.badge-high { background: #FFF7ED; color: #C2410C; border: 1px solid #FED7AA; }
.badge-medium { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.badge-low { background: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }
.dup-flag {
    color: #6941C6;
    background: #F9F5FF;
    border: 1px solid #E9D7FE;
    border-radius: 5px;
    padding: 3px 7px;
    font-size: 0.68rem;
    font-weight: 600;
}

/* ---------- Tables ---------- */
.table-wrap {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: auto;
}
.pretty-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}
.pretty-table thead th {
    position: sticky;
    top: 0;
    background: #F9FAFB;
    color: #475467;
    text-align: left;
    font-size: 0.72rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.025em;
    padding: 11px 13px;
    border-bottom: 1px solid var(--border);
    z-index: 1;
}
.pretty-table tbody td {
    background: #FFFFFF;
    color: #344054;
    padding: 11px 13px;
    border-bottom: 1px solid #EAECF0;
    vertical-align: top;
}
.pretty-table tbody tr:last-child td {
    border-bottom: none;
}
.pretty-table tbody tr:hover td {
    background: #FCFCFD;
}

/* ---------- Information panels ---------- */
.note-box {
    background: #FFFCF5;
    color: #7A2E0E;
    border: 1px solid #FEC84B;
    border-radius: 8px;
    padding: 13px 15px;
    line-height: 1.55;
    font-size: 0.84rem;
}

.result-panel {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-top: 16px;
}
.result-title {
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 650;
    margin-bottom: 12px;
}
.result-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}
.result-item {
    background: #F9FAFB;
    border: 1px solid #EAECF0;
    border-radius: 7px;
    padding: 10px 12px;
}
.result-label {
    color: #667085;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.result-value {
    color: #101828;
    font-size: 0.92rem;
    font-weight: 650;
    margin-top: 4px;
}

/* ---------- Buttons ---------- */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent) !important;
    border-radius: 7px !important;
    min-height: 38px;
    box-shadow: none !important;
    font-weight: 600 !important;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

/* Streamlit dataframe shell */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .app-header { flex-direction: column; }
    .result-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# API helpers
# ============================================================


def api_get(path, params=None):
    """GET from the FastAPI backend. Returns (data, error_message)."""
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            "Can't reach the API. Is the backend running? "
            "Start it with `uvicorn app.main:app --reload --port 8000`."
        )
    except requests.exceptions.Timeout:
        return None, "API request timed out."
    except requests.exceptions.HTTPError as exc:
        return None, f"API returned an error: {exc}"


def api_post(path, json_body, timeout=180):
    """POST to the FastAPI backend."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=timeout)
        if resp.status_code >= 400:
            return None, f"API error ({resp.status_code}): {resp.text}"
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Can't reach the API. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "Classification timed out. The model may still be loading; try again shortly."


@st.cache_data(ttl=30)
def load_reports():
    data, err = api_get("/reports", params={"limit": 1000})
    if err:
        return pd.DataFrame(), err
    return pd.DataFrame(data), None


@st.cache_data(ttl=30)
def load_duplicates():
    data, err = api_get("/duplicates", params={"limit": 1000})
    if err:
        return pd.DataFrame(), err
    return pd.DataFrame(data), None


@st.cache_data(ttl=30)
def load_evaluation():
    data, err = api_get("/evaluation")
    if err:
        return pd.DataFrame(), err
    return pd.DataFrame(data), None


@st.cache_data(ttl=30)
def load_stats():
    return api_get("/stats")


# ============================================================
# Formatting helpers
# ============================================================


def safe_text(value):
    return html.escape(str(value), quote=True)


def format_percent(value, digits=0):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}%}"


def render_metric_card(label, value, hint=""):
    hint_html = f'<div class="metric-hint">{safe_text(hint)}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(value)}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity):
    sev = severity if severity in SEVERITY_ORDER else "Medium"
    return f'<span class="badge badge-{sev.lower()}">{safe_text(sev)}</span>'


def render_ticket(row, has_confidence):
    severity = str(row.get("predicted_severity", "Medium"))
    severity_class = severity.lower() if severity in SEVERITY_ORDER else "medium"
    report_id = row.get("id", "—")
    is_duplicate = bool(row.get("is_duplicate", False))
    dup_html = '<span class="dup-flag">Potential duplicate</span>' if is_duplicate else ""

    full_text = str(row.get("full_text", ""))
    preview = full_text[:185] + ("…" if len(full_text) > 185 else "")

    meta_parts = [f"Priority {row.get('priority_score', '—')}"]
    if has_confidence and pd.notna(row.get("model_confidence")):
        meta_parts.append(f"Confidence {format_percent(row.get('model_confidence'), 0)}")
    if row.get("repo"):
        meta_parts.append(str(row.get("repo")))

    st.markdown(
        f"""
        <div class="ticket ticket-{severity_class}">
            <div class="ticket-top">
                <div style="display:flex; align-items:center; gap:8px;">
                    {severity_badge(severity)}
                    <span class="ticket-id">#{safe_text(report_id)}</span>
                </div>
                {dup_html}
            </div>
            <div class="ticket-text">{safe_text(preview)}</div>
            <div class="ticket-meta">{safe_text('  ·  '.join(meta_parts))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_table(table_df, height=520):
    """Render a compact table with safely escaped content."""
    display_df = table_df.fillna("—").copy()
    html_table = display_df.to_html(
        index=False,
        escape=True,
        classes="pretty-table",
        border=0,
    )
    st.markdown(
        f'<div class="table-wrap" style="max-height:{height}px; overflow-y:auto;">{html_table}</div>',
        unsafe_allow_html=True,
    )


def render_evaluation_table(all_results):
    rows = []
    for model_name, scores in all_results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": f"{scores['accuracy']:.1%}",
                "Macro Precision": f"{scores['macro_precision']:.2f}",
                "Macro Recall": f"{scores['macro_recall']:.2f}",
                "Macro F1": f"{scores['macro_f1']:.2f}",
            }
        )
    render_html_table(pd.DataFrame(rows), height=300)


# ============================================================
# Load live data
# ============================================================

stats, stats_err = load_stats()

if stats_err:
    status_html = (
        '<span class="status-pill offline"><span class="status-dot"></span>API offline</span>'
    )
else:
    status_html = (
        '<span class="status-pill"><span class="status-dot"></span>API operational</span>'
    )

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="app-title">Triage Console</div>
            <div class="app-subtitle">Bug report triage and prioritization workspace</div>
        </div>
        <div>{status_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if stats_err:
    st.error(stats_err)
    st.stop()


df, df_err = load_reports()
if df_err:
    st.error(df_err)
    st.stop()

if df.empty:
    st.info("The API is connected, but no reports were returned yet.")
    st.stop()

# Normalise numeric fields so filtering/charting does not break on string values.
df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce")
if "model_confidence" in df.columns:
    df["model_confidence"] = pd.to_numeric(df["model_confidence"], errors="coerce")

has_confidence = "model_confidence" in df.columns

duplicates_df, dup_err = load_duplicates()
eval_df, eval_err = load_evaluation()


# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.markdown("### Filters")

selected_severities = st.sidebar.multiselect(
    "Severity",
    SEVERITY_ORDER,
    default=SEVERITY_ORDER,
)

valid_priorities = df["priority_score"].dropna()
if valid_priorities.empty:
    min_priority, max_priority = 0.0, 4.0
else:
    min_priority = float(valid_priorities.min())
    max_priority = float(valid_priorities.max())

priority_range = st.sidebar.slider(
    "Priority score",
    min_priority,
    max_priority,
    (min_priority, max_priority),
)

show_duplicates_only = st.sidebar.checkbox("Duplicates only", value=False)
search_term = st.sidebar.text_input("Search reports", "", placeholder="Bug ID or keyword")

repo_options = sorted(df["repo"].dropna().astype(str).unique().tolist())
selected_repos = st.sidebar.multiselect(
    "Repository",
    repo_options,
    default=repo_options,
)

# Apply filters to the client-side dataframe.
filtered_df = df[df["predicted_severity"].isin(selected_severities)].copy()
filtered_df = filtered_df[filtered_df["repo"].isin(selected_repos)]
filtered_df = filtered_df[
    filtered_df["priority_score"].between(priority_range[0], priority_range[1], inclusive="both")
]

if show_duplicates_only:
    filtered_df = filtered_df[filtered_df["is_duplicate"] == True]  # noqa: E712

if search_term.strip():
    term = search_term.strip()
    text_mask = filtered_df["full_text"].astype(str).str.contains(term, case=False, na=False, regex=False)
    id_mask = filtered_df["id"].astype(str).str.contains(term, case=False, na=False, regex=False)
    filtered_df = filtered_df[text_mask | id_mask]

st.sidebar.caption(f"{len(filtered_df):,} reports match the current filters")


# ============================================================
# Summary metrics
# ============================================================

duplicate_count = int(stats.get("duplicates_found", 0))
total_reports = int(stats.get("total_reports", len(df)))
duplicate_rate = (duplicate_count / total_reports * 100) if total_reports else 0.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total reports", f"{total_reports:,}", "Reports stored in the triage database")
with m2:
    critical_count = int(stats.get("critical_count", 0))
    critical_share = (critical_count / total_reports * 100) if total_reports else 0.0
    render_metric_card("Critical", f"{critical_count:,}", f"{critical_share:.1f}% of all reports")
with m3:
    render_metric_card("Duplicates", f"{duplicate_count:,}", f"{duplicate_rate:.1f}% duplicate rate")
with m4:
    render_metric_card("Filtered reports", f"{len(filtered_df):,}", "Based on the active sidebar filters")

st.write("")


# ============================================================
# Main navigation
# ============================================================

tab_overview, tab_reports, tab_duplicates, tab_evaluation, tab_submit = st.tabs(
    ["Overview", "Reports", "Duplicates", "Evaluation", "New report"]
)


# ---------------- Overview ----------------
with tab_overview:
    left, right = st.columns([0.95, 1.35], gap="large")

    with left:
        st.markdown('<div class="section-title">Severity distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Number of reports assigned to each severity level.</div>',
            unsafe_allow_html=True,
        )

        severity_counts = (
            df["predicted_severity"]
            .value_counts()
            .reindex(SEVERITY_ORDER)
            .fillna(0)
            .astype(int)
        )

        fig = go.Figure(
            go.Bar(
                x=severity_counts.values,
                y=SEVERITY_ORDER,
                orientation="h",
                marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                text=[f"{v:,}" for v in severity_counts.values],
                textposition="outside",
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )
        fig.update_layout(
            height=330,
            paper_bgcolor="#F7F8FA",
            plot_bgcolor="#F7F8FA",
            margin=dict(l=8, r=35, t=10, b=20),
            showlegend=False,
            font=dict(color="#475467", family="Arial"),
            xaxis=dict(
                title="",
                showgrid=True,
                gridcolor="#EAECF0",
                zeroline=False,
                tickfont=dict(color="#667085"),
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                autorange="reversed",
                tickfont=dict(color="#344054"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">System snapshot</div>', unsafe_allow_html=True)
        overview_rows = pd.DataFrame(
            [
                {"Measure": "Repositories", "Value": f"{df['repo'].nunique():,}"},
                {"Measure": "Highest priority", "Value": f"{df['priority_score'].max():.1f}"},
                {
                    "Measure": "Average confidence",
                    "Value": format_percent(df["model_confidence"].mean(), 0) if has_confidence else "—",
                },
                {"Measure": "Duplicate rate", "Value": f"{duplicate_rate:.1f}%"},
            ]
        )
        render_html_table(overview_rows, height=250)

    with right:
        st.markdown('<div class="section-title">Priority queue</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Highest-priority reports requiring attention first.</div>',
            unsafe_allow_html=True,
        )

        top_reports = df.sort_values(
            ["priority_score", "model_confidence"] if has_confidence else ["priority_score"],
            ascending=False,
            na_position="last",
        ).head(8)

        for _, row in top_reports.iterrows():
            render_ticket(row, has_confidence)


# ---------------- Reports ----------------
with tab_reports:
    st.markdown('<div class="section-title">Reports</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">{len(filtered_df):,} reports match the current filters.</div>',
        unsafe_allow_html=True,
    )

    report_columns = ["id", "repo", "predicted_severity", "priority_score", "is_duplicate", "full_text"]
    report_names = ["ID", "Repository", "Severity", "Priority", "Duplicate", "Summary"]

    if has_confidence:
        report_columns.insert(4, "model_confidence")
        report_names.insert(4, "Confidence")

    report_table = filtered_df[report_columns].copy()
    report_table.columns = report_names
    report_table["ID"] = report_table["ID"].map(lambda x: f"#{x}")
    report_table["Priority"] = report_table["Priority"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
    )
    report_table["Duplicate"] = report_table["Duplicate"].map(lambda x: "Yes" if bool(x) else "No")
    if "Confidence" in report_table.columns:
        report_table["Confidence"] = report_table["Confidence"].map(lambda x: format_percent(x, 0))
    report_table["Summary"] = report_table["Summary"].astype(str).map(
        lambda x: x[:170] + ("…" if len(x) > 170 else "")
    )

    report_table = report_table.sort_values("Priority", ascending=False)
    render_html_table(report_table, height=560)


# ---------------- Duplicates ----------------
with tab_duplicates:
    st.markdown('<div class="section-title">Potential duplicates</div>', unsafe_allow_html=True)

    if dup_err:
        st.error(dup_err)
    elif duplicates_df.empty:
        st.info("No duplicate pairs were returned by the API.")
    else:
        st.markdown(
            f'<div class="section-subtitle">{len(duplicates_df):,} report pairs were detected through semantic similarity.</div>',
            unsafe_allow_html=True,
        )

        sorted_dups = duplicates_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

        # Compact summary first; raw backend columns stay available below.
        dup_summary = sorted_dups[["report_1_id", "report_2_id", "similarity_score"]].copy()
        dup_summary.columns = ["Reference report", "Potential duplicate", "Similarity"]
        dup_summary["Reference report"] = dup_summary["Reference report"].map(lambda x: f"#{x}")
        dup_summary["Potential duplicate"] = dup_summary["Potential duplicate"].map(lambda x: f"#{x}")
        dup_summary["Similarity"] = dup_summary["Similarity"].map(lambda x: format_percent(x, 1))
        render_html_table(dup_summary, height=310)

        st.write("")
        st.markdown('<div class="section-title">Reference comparison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Open a pair to compare both reports in full.</div>',
            unsafe_allow_html=True,
        )

        for _, row in sorted_dups.iterrows():
            label = (
                f"#{row['report_1_id']}  ↔  #{row['report_2_id']}"
                f"   ·   {row['similarity_score']:.0%} similarity"
            )
            with st.expander(label):
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.markdown(f"**Reference report #{row['report_1_id']}**")
                    st.write(str(row["report_1_text"]))
                with c2:
                    st.markdown(f"**Potential duplicate #{row['report_2_id']}**")
                    st.write(str(row["report_2_text"]))

        with st.expander("Raw duplicate data"):
            st.dataframe(sorted_dups, use_container_width=True, hide_index=True)


# ---------------- Evaluation ----------------
with tab_evaluation:
    st.markdown('<div class="section-title">Model evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Severity classification results on the labeled Eclipse evaluation set.</div>',
        unsafe_allow_html=True,
    )

    if eval_err or eval_df.empty:
        if eval_err:
            st.caption(f"Live evaluation unavailable ({eval_err}). Showing the last known values.")
        all_results = {
            "LLM (Zero-Shot)": {
                "accuracy": 0.5634,
                "macro_f1": 0.22,
                "macro_precision": 0.24,
                "macro_recall": 0.23,
            }
        }
        all_results.update(ML_BASELINE_RESULTS)
    else:
        all_results = {
            row["model_name"]: {
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "macro_precision": float(row["macro_precision"]),
                "macro_recall": float(row["macro_recall"]),
            }
            for _, row in eval_df.iterrows()
        }

    eval_left, eval_right = st.columns([1.15, 0.85], gap="large")

    with eval_left:
        render_evaluation_table(all_results)

    with eval_right:
        model_names = list(all_results.keys())
        f1_values = [all_results[name]["macro_f1"] for name in model_names]
        fig = go.Figure(
            go.Bar(
                x=f1_values,
                y=model_names,
                orientation="h",
                marker_color="#6366F1",
                text=[f"{value:.2f}" for value in f1_values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text="Macro F1 comparison", font=dict(size=14, color="#101828")),
            height=300,
            margin=dict(l=0, r=35, t=45, b=10),
            paper_bgcolor="#F7F8FA",
            plot_bgcolor="#F7F8FA",
            font=dict(color="#475467", family="Arial"),
            xaxis=dict(range=[0, max(0.3, max(f1_values) + 0.04)], gridcolor="#EAECF0", zeroline=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
        <div class="note-box">
            <strong>How to read these results.</strong> Raw accuracy is higher for the traditional ML baselines,
            but the evaluation set is highly imbalanced and those models heavily favour the majority
            <em>Medium</em> class. Macro F1 gives equal importance to every severity class, making it a more
            informative comparison for this proof-of-concept evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    with st.expander("Known limitations"):
        st.markdown(
            """
            - Defect **category** classification (UI / Backend / Performance / Security) is not yet implemented; only severity is currently classified. This is planned for FYP2.
            - The evaluation set contains **71 labeled examples** and is severely imbalanced (**61 Medium vs 1 Low**), so the results should be interpreted as proof-of-concept evidence rather than a final benchmark.
            """
        )


# ---------------- New report ----------------
with tab_submit:
    form_col, info_col = st.columns([1.35, 0.65], gap="large")

    with form_col:
        st.markdown('<div class="section-title">Submit a new report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">The report is classified, checked for semantic duplicates, prioritised and stored through the live API.</div>',
            unsafe_allow_html=True,
        )

        with st.form("submit_report_form", clear_on_submit=False):
            repo_input = st.text_input("Repository", value="microsoft/vscode")
            text_input = st.text_area(
                "Bug report",
                height=190,
                placeholder=(
                    "Describe the issue, expected behaviour, actual behaviour, "
                    "and any useful environment details."
                ),
            )
            submitted = st.form_submit_button("Submit for triage", type="primary")

    with info_col:
        st.markdown('<div class="section-title">Processing pipeline</div>', unsafe_allow_html=True)
        pipeline_df = pd.DataFrame(
            [
                {"Step": "1", "Action": "Severity classification"},
                {"Step": "2", "Action": "Semantic duplicate check"},
                {"Step": "3", "Action": "Priority scoring"},
                {"Step": "4", "Action": "Database storage"},
            ]
        )
        render_html_table(pipeline_df, height=260)

    if submitted:
        if not text_input.strip():
            st.warning("Enter a bug report before submitting.")
        else:
            with st.spinner("Triaging report… the first model call may take longer while the classifier loads."):
                result, err = api_post(
                    "/reports",
                    {"repo": repo_input, "full_text": text_input},
                )

            if err:
                st.error(err)
            else:
                severity = result.get("predicted_severity", "—")
                confidence = result.get("model_confidence")
                priority = result.get("priority_score", "—")
                is_duplicate = bool(result.get("is_duplicate", False))
                duplicate_value = "Yes" if is_duplicate else "No"

                duplicate_detail = ""
                if is_duplicate:
                    matched_id = result.get("matched_duplicate_id", "—")
                    similarity = result.get("similarity_score")
                    duplicate_detail = (
                        f'<div class="ticket-meta" style="margin-top:14px;">Closest match: '
                        f'#{safe_text(matched_id)} · similarity {safe_text(format_percent(similarity, 0))}</div>'
                    )

                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="result-title">Triage result · report #{safe_text(result.get('id', '—'))}</div>
                        <div class="result-grid">
                            <div class="result-item">
                                <div class="result-label">Severity</div>
                                <div class="result-value">{safe_text(severity)}</div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Model confidence</div>
                                <div class="result-value">{safe_text(format_percent(confidence, 0))}</div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Priority</div>
                                <div class="result-value">{safe_text(priority)}</div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Potential duplicate</div>
                                <div class="result-value">{duplicate_value}</div>
                            </div>
                        </div>
                        {duplicate_detail}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success("Report submitted and stored successfully.")

                # Invalidate cached backend data so a rerun immediately reflects the new report.
                load_reports.clear()
                load_duplicates.clear()
                load_stats.clear()