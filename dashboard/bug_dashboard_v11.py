import html
import math
from collections import defaultdict, deque

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Triage Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

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
    "Critical": "#C2414B",
    "High": "#D97706",
    "Medium": "#2563EB",
    "Low": "#0D9488",
}

# Prototype decision-support setting. Tune this value after FYP II threshold testing.
# It only controls the manual-review flag shown in the UI; it does not alter the classifier.
REVIEW_CONFIDENCE_THRESHOLD = 0.50

# Palette inspired by the supplied Modern Slate Streamlit reference.
PRIMARY = "#0F172A"
SECONDARY = "#2563EB"
TERTIARY = "#0D9488"
NEUTRAL = "#64748B"
BG = "#F5F8FC"
PANEL = "#EAF2FF"
SURFACE = "#FFFFFF"
BORDER = "#CBD5E1"


# ============================================================
# Theme
# ============================================================

st.markdown(
    f"""
<style>
:root {{
    --primary: {PRIMARY};
    --secondary: {SECONDARY};
    --tertiary: {TERTIARY};
    --neutral: {NEUTRAL};
    --bg: {BG};
    --panel: {PANEL};
    --surface: {SURFACE};
    --border: {BORDER};
    --muted-border: #E2E8F0;
    --muted: #94A3B8;
    --danger: #C2414B;
    --warning: #D97706;
}}

html, body, [class*="css"] {{
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: var(--bg) !important;
    color: var(--primary) !important;
}}

[data-testid="stMainBlockContainer"] {{
    max-width: 1540px;
    padding-top: 1.65rem;
    padding-bottom: 4rem;
    padding-left: 2.15rem;
    padding-right: 2.15rem;
}}

[data-testid="stToolbar"], #MainMenu, footer {{
    visibility: hidden !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--primary) !important;
    letter-spacing: -0.025em;
}}

p, label, .stCaption {{ color: var(--neutral); }}

/* App header */
.app-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 14px;
}}

.brand-wrap {{
    display: flex;
    align-items: center;
    gap: 13px;
}}

.brand-mark {{
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: var(--primary);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.10);
}}

.app-title {{
    margin: 0;
    font-size: 1.72rem;
    line-height: 1.1;
    font-weight: 760;
    color: var(--primary);
}}

.app-subtitle {{
    margin-top: 4px;
    font-size: 0.87rem;
    color: var(--neutral);
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #99F6E4;
    background: #F0FDFA;
    color: #0F766E;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 0.77rem;
    font-weight: 650;
    white-space: nowrap;
}}
.status-pill.offline {{
    border-color: #FECACA;
    background: #FEF2F2;
    color: #B91C1C;
}}
.status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--tertiary);
    display: inline-block;
}}
.status-pill.offline .status-dot {{ background: #DC2626; }}

/* Architecture strip */
.arch-strip {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
    margin: 0 0 18px 0;
}}
.arch-item {{
    background: #FFFFFF;
    padding: 10px 13px;
}}
.arch-label {{
    color: var(--neutral);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.63rem;
    font-weight: 700;
}}
.arch-value {{
    color: var(--primary);
    margin-top: 2px;
    font-size: 0.80rem;
    font-weight: 650;
}}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background: #FFFFFF !important;
}}
section[data-testid="stSidebar"] {{
    border-right: 1px solid var(--muted-border) !important;
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
    color: var(--primary) !important;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: #334155;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] input {{
    background: #FFFFFF !important;
    color: var(--primary) !important;
    border-color: var(--border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
    background: #DBEAFE !important;
    color: #1E3A8A !important;
    border-radius: 4px !important;
}}
section[data-testid="stSidebar"] span[data-baseweb="tag"] span {{
    color: #1E3A8A !important;
}}
.sidebar-note {{
    background: var(--panel);
    border: 1px solid #D7E6FB;
    border-radius: 8px;
    padding: 10px 11px;
    font-size: 0.75rem;
    color: #475569;
    line-height: 1.45;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid var(--muted-border);
}}
.stTabs [data-baseweb="tab"] {{
    height: 42px;
    background: transparent;
    border-radius: 0;
    padding: 0 10px;
    color: var(--neutral);
    font-weight: 600;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: var(--primary) !important;
    border-bottom: 2px solid var(--secondary) !important;
}}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div {{
    background: #FFFFFF !important;
    color: var(--primary) !important;
    border-color: var(--border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {{
    border-color: var(--secondary) !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.08) !important;
}}

/* Buttons */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {{
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--primary) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    font-weight: 650 !important;
}}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {{
    background: #1E293B !important;
    color: #FFFFFF !important;
    border-color: #1E293B !important;
}}

/* Metric cards */
.metric-card {{
    background: #FFFFFF;
    border: 1px solid var(--muted-border);
    border-radius: 8px;
    padding: 15px 16px;
    min-height: 102px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}}
.metric-topline {{
    width: 28px;
    height: 3px;
    border-radius: 999px;
    margin-bottom: 11px;
}}
.metric-label {{
    font-size: 0.69rem;
    color: var(--neutral);
    text-transform: uppercase;
    letter-spacing: 0.055em;
    font-weight: 700;
}}
.metric-value {{
    font-size: 1.75rem;
    color: var(--primary);
    line-height: 1.15;
    font-weight: 780;
    margin-top: 3px;
}}
.metric-note {{
    margin-top: 5px;
    font-size: 0.72rem;
    color: #94A3B8;
}}

/* General panels */
.panel {{
    background: #FFFFFF;
    border: 1px solid var(--muted-border);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.025);
}}
.panel-soft {{
    background: var(--panel);
    border: 1px solid #D7E6FB;
    border-radius: 8px;
    padding: 14px 15px;
}}
.section-kicker {{
    color: var(--secondary);
    font-size: 0.66rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 3px;
}}
.section-title {{
    color: var(--primary);
    font-size: 1rem;
    font-weight: 720;
    margin-bottom: 3px;
}}
.section-desc {{
    color: var(--neutral);
    font-size: 0.80rem;
    line-height: 1.45;
}}

/* Priority queue */
.ticket-row {{
    background: #FFFFFF;
    border: 1px solid var(--muted-border);
    border-left-width: 3px;
    border-radius: 7px;
    padding: 12px 13px;
    margin-bottom: 8px;
}}
.ticket-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}}
.ticket-id {{
    color: var(--neutral);
    font-size: 0.70rem;
    font-weight: 650;
}}
.severity-badge {{
    display: inline-flex;
    align-items: center;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.65rem;
    font-weight: 750;
    letter-spacing: 0.02em;
}}
.ticket-text {{
    color: var(--primary);
    font-size: 0.82rem;
    line-height: 1.42;
    font-weight: 550;
}}
.ticket-meta {{
    margin-top: 7px;
    color: var(--neutral);
    font-size: 0.72rem;
}}
.review-badge {{
    display: inline-flex;
    align-items: center;
    margin-left: 6px;
    border: 1px solid #FCD34D;
    background: #FFFBEB;
    color: #92400E;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.62rem;
    font-weight: 750;
    letter-spacing: 0.02em;
}}

.research-card {{
    background: #FFFFFF;
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 16px;
    color: #475569;
    font-size: 0.80rem;
    line-height: 1.5;
}}
.research-card strong {{ color: var(--primary); }}

/* Live triage results */
.result-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
}}
.result-card {{
    background: var(--panel);
    border: 1px solid #D7E6FB;
    border-radius: 7px;
    padding: 13px;
}}
.result-label {{
    color: var(--neutral);
    text-transform: uppercase;
    letter-spacing: 0.055em;
    font-size: 0.62rem;
    font-weight: 700;
}}
.result-value {{
    color: var(--primary);
    font-weight: 760;
    margin-top: 4px;
    font-size: 1.03rem;
}}

/* Pipeline */
.pipeline {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 11px;
}}
.pipeline-step {{
    background: #FFFFFF;
    border: 1px solid var(--muted-border);
    border-top: 3px solid var(--secondary);
    border-radius: 7px;
    padding: 11px;
}}
.pipeline-step:nth-child(3) {{ border-top-color: var(--tertiary); }}
.pipeline-step:nth-child(4) {{ border-top-color: var(--primary); }}
.pipeline-n {{
    color: var(--neutral);
    font-size: 0.62rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.pipeline-name {{
    color: var(--primary);
    margin-top: 3px;
    font-size: 0.77rem;
    font-weight: 650;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--muted-border);
    border-radius: 8px;
    overflow: hidden;
}}

/* Expander */
details[data-testid="stExpander"] {{
    background: #FFFFFF;
    border: 1px solid var(--muted-border) !important;
    border-radius: 7px !important;
}}

/* Notes */
.note-box {{
    background: var(--panel);
    border: 1px solid #D7E6FB;
    border-left: 3px solid var(--secondary);
    border-radius: 7px;
    padding: 12px 14px;
    color: #334155;
    font-size: 0.79rem;
    line-height: 1.5;
}}
.note-box.teal {{ border-left-color: var(--tertiary); }}

@media (max-width: 900px) {{
    .arch-strip, .result-grid, .pipeline {{ grid-template-columns: 1fr 1fr; }}
    .app-header {{ align-items: flex-start; flex-direction: column; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# API helpers
# ============================================================


def api_get(path, params=None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the API at {API_BASE}. Start the FastAPI backend first: "
            "uvicorn app.main:app --reload --port 8000"
        )
    except requests.exceptions.Timeout:
        return None, "API request timed out."
    except requests.exceptions.HTTPError as exc:
        return None, f"API returned an error: {exc}"
    except requests.exceptions.RequestException as exc:
        return None, f"API request failed: {exc}"


def api_post(path, json_body, timeout=180):
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=timeout)
        if resp.status_code >= 400:
            return None, f"API error ({resp.status_code}): {resp.text}"
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_BASE}."
    except requests.exceptions.Timeout:
        return None, "Classification timed out. The model may still be loading. Try again shortly."
    except requests.exceptions.RequestException as exc:
        return None, f"API request failed: {exc}"


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


@st.cache_data(ttl=30)
def load_confusion_matrix():
    """Optional endpoint for final FYP evaluation. The dashboard remains usable if it is absent."""
    return api_get("/evaluation/confusion-matrix")


@st.cache_data(ttl=30)
def load_duplicate_evaluation():
    """Optional endpoint for duplicate-detection precision/recall/F1."""
    return api_get("/evaluation/duplicates")


# ============================================================
# Formatting helpers
# ============================================================


def safe_text(value):
    return html.escape(str(value))


def bool_text(value):
    if pd.isna(value):
        return "No"
    if isinstance(value, str):
        return "Yes" if value.strip().lower() in {"true", "1", "yes"} else "No"
    return "Yes" if bool(value) else "No"


def metric_card(label, value, note="", accent=SECONDARY):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-topline" style="background:{accent};"></div>
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(value)}</div>
            <div class="metric-note">{safe_text(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge_html(severity):
    color = SEVERITY_COLORS.get(severity, NEUTRAL)
    bg = {
        "Critical": "#FEF2F2",
        "High": "#FFF7ED",
        "Medium": "#EFF6FF",
        "Low": "#F0FDFA",
    }.get(severity, "#F8FAFC")
    return (
        f'<span class="severity-badge" style="background:{bg};color:{color};'
        f'border:1px solid {color}33;">{safe_text(severity)}</span>'
    )


def render_ticket(row, has_confidence=True):
    severity = str(row.get("predicted_severity", "Unknown"))
    color = SEVERITY_COLORS.get(severity, NEUTRAL)
    bug_id = row.get("id", "-")
    preview = str(row.get("full_text", ""))[:175]
    priority = row.get("priority_score", "-")

    confidence = row.get("model_confidence", None)
    needs_review = False
    if has_confidence and pd.notna(confidence):
        confidence = float(confidence)
        conf_text = f"{confidence:.0%} confidence"
        needs_review = confidence < REVIEW_CONFIDENCE_THRESHOLD
    else:
        conf_text = "confidence unavailable"

    review_html = '<span class="review-badge">MANUAL REVIEW</span>' if needs_review else ""
    dup_text = "possible duplicate" if bool(row.get("is_duplicate", False)) else "unique candidate"

    st.markdown(
        f"""
        <div class="ticket-row" style="border-left-color:{color};">
            <div class="ticket-head">
                <span class="ticket-id">Report #{safe_text(bug_id)} {review_html}</span>
                {severity_badge_html(severity)}
            </div>
            <div class="ticket-text">{safe_text(preview)}{'...' if len(str(row.get('full_text', ''))) > 175 else ''}</div>
            <div class="ticket-meta">Priority {safe_text(priority)} &nbsp; / &nbsp; {safe_text(conf_text)} &nbsp; / &nbsp; {safe_text(dup_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def review_status(confidence):
    """Return a human-readable decision-support flag without changing model output."""
    try:
        if pd.isna(confidence):
            return "Review"
        return "Review" if float(confidence) < REVIEW_CONFIDENCE_THRESHOLD else "Standard"
    except (TypeError, ValueError):
        return "Review"


def parse_confusion_payload(payload):
    """Accept a few common confusion-matrix payload shapes from a backend endpoint."""
    if payload is None:
        return None, None, None

    if isinstance(payload, dict):
        model_name = payload.get("model_name") or payload.get("model") or "LLM classifier"
        labels = payload.get("labels") or payload.get("classes") or payload.get("severity_labels")
        matrix = payload.get("matrix") or payload.get("confusion_matrix")
        if labels is not None and matrix is not None:
            return list(labels), matrix, model_name

        # Row format: {rows: [{actual: 'Critical', predicted: 'High', count: 2}, ...]}
        rows = payload.get("rows")
        if isinstance(rows, list):
            payload = rows

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        required = {"actual", "predicted", "count"}
        if required.issubset(payload[0].keys()):
            labels = [s for s in SEVERITY_ORDER if any(r.get("actual") == s or r.get("predicted") == s for r in payload)]
            extras = sorted({str(r.get("actual")) for r in payload} | {str(r.get("predicted")) for r in payload})
            labels += [x for x in extras if x not in labels and x != "None"]
            lookup = {(str(r.get("actual")), str(r.get("predicted"))): int(r.get("count", 0)) for r in payload}
            matrix = [[lookup.get((a, p), 0) for p in labels] for a in labels]
            return labels, matrix, "LLM classifier"

    return None, None, None


def parse_duplicate_eval(payload):
    """Read duplicate precision/recall/F1 without inventing values when the backend has none."""
    if payload is None:
        return None
    if isinstance(payload, list):
        if not payload:
            return None
        payload = payload[0]
    if not isinstance(payload, dict):
        return None

    def get_float(*keys):
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    pass
        return None

    precision = get_float("precision", "duplicate_precision")
    recall = get_float("recall", "duplicate_recall")
    f1 = get_float("f1", "f1_score", "duplicate_f1")
    threshold = get_float("threshold", "similarity_threshold")
    if precision is None and recall is None and f1 is None:
        return None
    return {"precision": precision, "recall": recall, "f1": f1, "threshold": threshold}


def build_similarity_graph(dup_df, min_similarity=0.0, max_pairs=40):
    """Build a lightweight connected-component topology without extra dependencies."""
    if dup_df.empty:
        return None

    work = dup_df.copy()
    if "similarity_score" not in work.columns:
        return None

    work = work[pd.to_numeric(work["similarity_score"], errors="coerce") >= min_similarity]
    work = work.sort_values("similarity_score", ascending=False).head(max_pairs)
    if work.empty:
        return None

    adjacency = defaultdict(set)
    edge_scores = {}
    for _, row in work.iterrows():
        a = str(row["report_1_id"])
        b = str(row["report_2_id"])
        score = float(row["similarity_score"])
        adjacency[a].add(b)
        adjacency[b].add(a)
        edge_scores[tuple(sorted((a, b)))] = score

    # Connected components.
    visited = set()
    components = []
    for node in adjacency:
        if node in visited:
            continue
        queue = deque([node])
        visited.add(node)
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        components.append(comp)

    components.sort(key=len, reverse=True)

    positions = {}
    palette = [SECONDARY, TERTIARY, "#8B5CF6", "#D97706", "#475569"]
    for c_idx, comp in enumerate(components):
        center_x = c_idx * 3.4
        center_y = 0.0 if c_idx % 2 == 0 else 1.4
        radius = max(0.48, min(1.0, 0.20 * len(comp) + 0.25))
        for i, node in enumerate(comp):
            angle = 2 * math.pi * i / max(len(comp), 1)
            positions[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
                palette[c_idx % len(palette)],
                c_idx + 1,
            )

    edge_x, edge_y = [], []
    for pair, score in edge_scores.items():
        a, b = pair
        if a not in positions or b not in positions:
            continue
        x0, y0, _, _ = positions[a]
        x1, y1, _, _ = positions[b]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.3, color="#94A3B8"),
        hoverinfo="none",
        showlegend=False,
    )

    node_x, node_y, node_color, node_text = [], [], [], []
    for node, (x, y, color, cluster_num) in positions.items():
        node_x.append(x)
        node_y.append(y)
        node_color.append(color)
        node_text.append(f"Report #{node}<br>Cluster {cluster_num}<br>{len(adjacency[node])} linked report(s)")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=12, color=node_color, line=dict(width=2, color="#FFFFFF")),
        text=node_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False, showticklabels=False),
        hovermode="closest",
    )
    return fig


# ============================================================
# Load backend data
# ============================================================

stats, stats_err = load_stats()

st.markdown(
    f"""
<div class="app-header">
    <div class="brand-wrap">
        <div class="brand-mark">TC</div>
        <div>
            <div class="app-title">Triage Console</div>
            <div class="app-subtitle">LLM-assisted bug report triaging and prioritization workspace</div>
        </div>
    </div>
    <div class="status-pill{' offline' if stats_err else ''}">
        <span class="status-dot"></span>
        {'API offline' if stats_err else 'System operational'}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if stats_err:
    st.error(stats_err)
    st.stop()

st.markdown(
    """
<div class="arch-strip">
    <div class="arch-item"><div class="arch-label">Classification</div><div class="arch-value">BART zero-shot severity model</div></div>
    <div class="arch-item"><div class="arch-label">Similarity</div><div class="arch-value">Sentence embeddings + FAISS</div></div>
    <div class="arch-item"><div class="arch-label">Backend</div><div class="arch-value">FastAPI service layer</div></div>
    <div class="arch-item"><div class="arch-label">Storage</div><div class="arch-value">SQLite report store</div></div>
</div>
""",
    unsafe_allow_html=True,
)


df, df_err = load_reports()
if df_err:
    st.error(df_err)
    st.stop()

duplicates_df, dup_err = load_duplicates()
eval_df, eval_err = load_evaluation()
confusion_payload, confusion_err = load_confusion_matrix()
duplicate_eval_payload, duplicate_eval_err = load_duplicate_evaluation()
has_confidence = "model_confidence" in df.columns

# ============================================================
# Sidebar / global filters
# ============================================================

st.sidebar.markdown("### Workspace filters")

severity_options = [s for s in SEVERITY_ORDER if s in set(df["predicted_severity"].dropna())] or SEVERITY_ORDER
selected_severities = st.sidebar.multiselect(
    "Severity",
    severity_options,
    default=severity_options,
)

min_priority = float(pd.to_numeric(df["priority_score"], errors="coerce").min())
max_priority = float(pd.to_numeric(df["priority_score"], errors="coerce").max())
priority_range = st.sidebar.slider(
    "Priority score",
    min_value=min_priority,
    max_value=max_priority,
    value=(min_priority, max_priority),
)

show_duplicates_only = st.sidebar.checkbox("Duplicates only", value=False)
search_term = st.sidebar.text_input("Search reports", placeholder="Report ID or keyword")
repo_options = sorted(df["repo"].dropna().astype(str).unique().tolist())
selected_repos = st.sidebar.multiselect("Repository", repo_options, default=repo_options)

filtered_df = df[df["predicted_severity"].isin(selected_severities)].copy()
filtered_df = filtered_df[filtered_df["repo"].isin(selected_repos)]
filtered_df = filtered_df[
    (pd.to_numeric(filtered_df["priority_score"], errors="coerce") >= priority_range[0])
    & (pd.to_numeric(filtered_df["priority_score"], errors="coerce") <= priority_range[1])
]

if show_duplicates_only:
    filtered_df = filtered_df[filtered_df["is_duplicate"] == True]  # noqa: E712

if search_term:
    text_mask = filtered_df["full_text"].astype(str).str.contains(search_term, case=False, na=False)
    id_mask = filtered_df["id"].astype(str).str.contains(search_term, case=False, na=False) if "id" in filtered_df.columns else False
    filtered_df = filtered_df[text_mask | id_mask]

if st.sidebar.button("Refresh workspace data", use_container_width=True):
    load_reports.clear()
    load_duplicates.clear()
    load_evaluation.clear()
    load_stats.clear()
    load_confusion_matrix.clear()
    load_duplicate_evaluation.clear()
    st.rerun()

st.sidebar.markdown(
    f"""
<div class="sidebar-note">
    <strong>{len(filtered_df)}</strong> reports match the current filters.<br>
    Filters affect the dashboard and dataset explorer; model benchmark results remain unchanged.<br><br>
    Manual-review display threshold: <strong>{REVIEW_CONFIDENCE_THRESHOLD:.0%}</strong>. Tune this value after final confidence-threshold testing.
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Navigation
# ============================================================

overview_tab, triage_tab, explorer_tab, duplicate_tab, eval_tab = st.tabs(
    ["Overview", "Live Triage", "Dataset Explorer", "Duplicate Analysis", "Model Evaluation"]
)

# ============================================================
# Overview
# ============================================================

with overview_tab:
    total_reports = int(stats.get("total_reports", len(df)))
    critical_count = int(stats.get("critical_count", (df["predicted_severity"] == "Critical").sum()))
    duplicates_found = int(stats.get("duplicates_found", df["is_duplicate"].sum()))
    duplicate_rate = (duplicates_found / total_reports * 100) if total_reports else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total reports", f"{total_reports:,}", "Stored and available for triage", PRIMARY)
    with c2:
        metric_card("Critical", f"{critical_count:,}", "Highest severity category", "#C2414B")
    with c3:
        metric_card("Duplicate candidates", f"{duplicates_found:,}", "Flagged by semantic similarity", TERTIARY)
    with c4:
        metric_card("Duplicate rate", f"{duplicate_rate:.1f}%", "Share of all reports", SECONDARY)

    st.write("")
    left, right = st.columns([1.0, 1.25], gap="large")

    with left:
        st.markdown("### Severity distribution")
        st.caption("Current report mix by predicted severity.")
        severity_counts = (
            filtered_df["predicted_severity"]
            .value_counts()
            .reindex(SEVERITY_ORDER)
            .fillna(0)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=severity_counts.values,
                y=SEVERITY_ORDER,
                orientation="h",
                marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                text=[int(v) for v in severity_counts.values],
                textposition="outside",
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )
        fig.update_layout(
            height=355,
            margin=dict(l=10, r=35, t=5, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis=dict(gridcolor="#E2E8F0", zeroline=False, title="Reports"),
            yaxis=dict(autorange="reversed", title=""),
            font=dict(color=PRIMARY),
            bargap=0.42,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
<div class="note-box teal">
    Severity is one input to priority ranking. Model confidence and duplicate status are presented as supporting information for human review.
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Priority queue")
        st.caption("Highest-priority reports after applying the current workspace filters.")
        top_reports = filtered_df.sort_values("priority_score", ascending=False).head(6)
        if top_reports.empty:
            st.info("No reports match the current filters.")
        else:
            for _, row in top_reports.iterrows():
                render_ticket(row, has_confidence)

    with st.expander("Prototype scope and decision-support role"):
        st.markdown(
            """
            **Included in this prototype**

            - Severity classification for bug reports.
            - Semantic duplicate-candidate detection using sentence embeddings and FAISS.
            - Priority ranking to help reviewers decide what should be reviewed first.
            - Model-confidence display and low-confidence manual-review flags.
            - Search, filtering, dataset inspection, and model-evaluation views.

            **Not automated by this dashboard**

            The system does not automatically assign developers, repair source code, locate defective code,
            predict resolution time, or automatically merge duplicate reports. Predictions and duplicate
            candidates are decision-support outputs and remain subject to human review.
            """
        )


# ============================================================
# Live triage
# ============================================================

with triage_tab:
    st.markdown("### Live bug report triage")
    st.caption("Submit a report to the current backend for severity classification, semantic duplicate checking and priority scoring.")

    input_col, process_col = st.columns([1.25, 0.75], gap="large")

    with input_col:
        with st.form("submit_report_form", clear_on_submit=False):
            repo_input = st.text_input("Repository", value="microsoft/vscode")
            text_input = st.text_area(
                "Bug report text",
                height=220,
                placeholder="Describe the issue, impact, expected behaviour and any relevant context.",
            )
            submitted = st.form_submit_button("Analyse and submit", type="primary")

    with process_col:
        st.markdown(
            """
<div class="panel-soft">
    <div class="section-kicker">Processing workflow</div>
    <div class="section-title">One submission, four processing stages</div>
    <div class="section-desc">The page follows the same integrated flow described in the system architecture.</div>
    <div class="pipeline">
        <div class="pipeline-step"><div class="pipeline-n">Stage 1</div><div class="pipeline-name">Prepare report text</div></div>
        <div class="pipeline-step"><div class="pipeline-n">Stage 2</div><div class="pipeline-name">Predict severity</div></div>
        <div class="pipeline-step"><div class="pipeline-n">Stage 3</div><div class="pipeline-name">Search duplicates</div></div>
        <div class="pipeline-step"><div class="pipeline-n">Stage 4</div><div class="pipeline-name">Calculate priority</div></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


    with st.expander("How the triage result is produced"):
        st.markdown(
            f"""
            **Severity classification**  
            The language model analyses the submitted report text and returns one severity class: Critical, High, Medium, or Low.

            **Model confidence**  
            The confidence value is the classifier's confidence for its selected severity. A report is marked **Manual review** in this interface when confidence is below **{REVIEW_CONFIDENCE_THRESHOLD:.0%}**. This flag does not change the prediction.

            **Semantic duplicate detection**  
            The report is converted to a sentence embedding and compared with existing report embeddings through FAISS. The backend similarity threshold determines whether the report is flagged as a potential duplicate.

            **Priority ranking**  
            The priority score shown here is calculated by the backend's current priority-ranking rules. The project methodology defines predicted severity as the main ranking factor, while model confidence and duplicate information support the review decision. The Streamlit interface displays the returned score rather than recalculating or altering it.
            """
        )

    if submitted:
        if not text_input.strip():
            st.warning("Enter a bug report description before submitting.")
        else:
            with st.spinner("Processing the report. The first model call can take longer while the model loads."):
                result, err = api_post("/reports", {"repo": repo_input, "full_text": text_input})

            if err:
                st.error(err)
            else:
                st.markdown("#### Triage result")
                severity = result.get("predicted_severity", "Unknown")
                priority = result.get("priority_score", "-")
                confidence = result.get("model_confidence", None)
                confidence_text = f"{float(confidence):.0%}" if confidence is not None else "Unavailable"
                duplicate_text = "Potential duplicate" if result.get("is_duplicate") else "No candidate found"

                st.markdown(
                    f"""
<div class="result-grid">
    <div class="result-card"><div class="result-label">Severity</div><div class="result-value">{safe_text(severity)}</div></div>
    <div class="result-card"><div class="result-label">Priority</div><div class="result-value">{safe_text(priority)}</div></div>
    <div class="result-card"><div class="result-label">Confidence</div><div class="result-value">{safe_text(confidence_text)}</div></div>
    <div class="result-card"><div class="result-label">Duplicate status</div><div class="result-value">{safe_text(duplicate_text)}</div></div>
</div>
""",
                    unsafe_allow_html=True,
                )

                if confidence is not None and float(confidence) < REVIEW_CONFIDENCE_THRESHOLD:
                    st.markdown(
                        f"""
<div class="note-box" style="margin-top:12px;border-left-color:#D97706;">
    <strong>Manual review recommended.</strong> The severity confidence ({float(confidence):.0%}) is below the current review threshold ({REVIEW_CONFIDENCE_THRESHOLD:.0%}).
    The flag supports human judgement and does not override the model prediction.
</div>
""",
                        unsafe_allow_html=True,
                    )

                if result.get("is_duplicate"):
                    matched_id = result.get("matched_duplicate_id", "-")
                    similarity = result.get("similarity_score", None)
                    similarity_text = f"{float(similarity):.1%}" if similarity is not None else "Unavailable"
                    st.markdown(
                        f"""
<div class="note-box" style="margin-top:12px;">
    The system linked this submission to report <strong>#{safe_text(matched_id)}</strong> with a similarity score of <strong>{safe_text(similarity_text)}</strong>.
    This is a review recommendation, not an automatic confirmation that both reports describe the same defect.
</div>
""",
                        unsafe_allow_html=True,
                    )

                st.success(f"Report #{result.get('id', '-')} was processed and stored in the database.")
                load_reports.clear()
                load_duplicates.clear()
                load_stats.clear()
                load_confusion_matrix.clear()
                load_duplicate_evaluation.clear()


# ============================================================
# Dataset explorer
# ============================================================

with explorer_tab:
    top_left, top_right = st.columns([1, 0.25])
    with top_left:
        st.markdown("### Bug report corpus and dataset explorer")
        st.caption("Inspect the processed report collection used by the triage workspace. Global sidebar filters are applied here.")
    with top_right:
        export_cols = [c for c in ["id", "repo", "predicted_severity", "model_confidence", "priority_score", "is_duplicate", "full_text"] if c in filtered_df.columns]
        export_df = filtered_df[export_cols].copy()
        if "model_confidence" in export_df.columns:
            export_df["review_status"] = export_df["model_confidence"].map(review_status)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export current view",
            data=csv_bytes,
            file_name="triage_reports_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    summary_cols = st.columns(4)
    with summary_cols[0]:
        metric_card("Visible reports", f"{len(filtered_df):,}", "Current filtered view", PRIMARY)
    with summary_cols[1]:
        repos_visible = filtered_df["repo"].nunique() if "repo" in filtered_df.columns else 0
        metric_card("Repositories", f"{repos_visible:,}", "Distinct sources in view", SECONDARY)
    with summary_cols[2]:
        avg_conf = pd.to_numeric(filtered_df.get("model_confidence"), errors="coerce").mean() if has_confidence else float("nan")
        metric_card("Average confidence", f"{avg_conf:.0%}" if pd.notna(avg_conf) else "N/A", "Severity model confidence", TERTIARY)
    with summary_cols[3]:
        avg_priority = pd.to_numeric(filtered_df["priority_score"], errors="coerce").mean() if len(filtered_df) else float("nan")
        metric_card("Average priority", f"{avg_priority:.2f}" if pd.notna(avg_priority) else "N/A", "Current filtered view", "#475569")

    st.write("")

    cols = [c for c in ["id", "repo", "predicted_severity", "model_confidence", "priority_score", "is_duplicate", "full_text"] if c in filtered_df.columns]
    display_df = filtered_df[cols].copy()
    if "model_confidence" in display_df.columns:
        display_df["review_status"] = display_df["model_confidence"].map(review_status)
    rename_map = {
        "id": "Report ID",
        "repo": "Repository",
        "predicted_severity": "Severity",
        "model_confidence": "Confidence",
        "priority_score": "Priority",
        "is_duplicate": "Duplicate",
        "review_status": "Review status",
        "full_text": "Bug report",
    }
    display_df = display_df.rename(columns=rename_map)
    if "Duplicate" in display_df.columns:
        display_df["Duplicate"] = display_df["Duplicate"].map(bool_text)
    if "Confidence" in display_df.columns:
        display_df["Confidence"] = pd.to_numeric(display_df["Confidence"], errors="coerce")

    st.dataframe(
        display_df.sort_values("Priority", ascending=False) if "Priority" in display_df.columns else display_df,
        use_container_width=True,
        height=520,
        hide_index=True,
        column_config={
            "Report ID": st.column_config.TextColumn("Report ID", width="small"),
            "Repository": st.column_config.TextColumn("Repository", width="medium"),
            "Severity": st.column_config.TextColumn("Severity", width="small"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent"),
            "Priority": st.column_config.NumberColumn("Priority", format="%.1f", width="small"),
            "Duplicate": st.column_config.TextColumn("Duplicate", width="small"),
            "Review status": st.column_config.TextColumn("Review status", width="small"),
            "Bug report": st.column_config.TextColumn("Bug report", width="large"),
        },
    )

    if not filtered_df.empty and "id" in filtered_df.columns:
        st.markdown("#### Report inspector")
        st.caption("Open a report from the filtered dataset to view its full triage output without truncation.")
        inspect_ids = filtered_df["id"].astype(str).tolist()
        selected_report_id = st.selectbox("Select report", inspect_ids, key="report_inspector")
        selected_row = filtered_df[filtered_df["id"].astype(str) == str(selected_report_id)].iloc[0]

        i1, i2, i3, i4 = st.columns(4)
        with i1:
            metric_card("Severity", selected_row.get("predicted_severity", "N/A"), "Predicted severity", SEVERITY_COLORS.get(str(selected_row.get("predicted_severity")), SECONDARY))
        with i2:
            conf = selected_row.get("model_confidence", None)
            metric_card("Confidence", f"{float(conf):.0%}" if pd.notna(conf) else "N/A", review_status(conf) if has_confidence else "Unavailable", TERTIARY)
        with i3:
            metric_card("Priority", selected_row.get("priority_score", "N/A"), "Current ranking score", PRIMARY)
        with i4:
            metric_card("Duplicate", bool_text(selected_row.get("is_duplicate", False)), "Semantic duplicate status", SECONDARY)

        st.markdown("**Full bug report**")
        st.markdown(
            f'<div class="panel" style="line-height:1.6;color:#334155;">{safe_text(selected_row.get("full_text", ""))}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Duplicate analysis
# ============================================================

with duplicate_tab:
    st.markdown("### Semantic duplicate analysis")
    st.caption("Review candidate duplicate relationships returned by the semantic similarity module. Final confirmation remains a human decision.")

    if dup_err:
        st.error(dup_err)
    elif duplicates_df.empty:
        st.info("No duplicate pairs are currently available from the backend.")
    else:
        dup_work = duplicates_df.copy()
        dup_work["similarity_score"] = pd.to_numeric(dup_work["similarity_score"], errors="coerce")
        min_score = float(dup_work["similarity_score"].min())
        max_score = float(dup_work["similarity_score"].max())

        d1, d2, d3 = st.columns(3)
        with d1:
            metric_card("Candidate pairs", f"{len(dup_work):,}", "Returned by the duplicate endpoint", PRIMARY)
        with d2:
            metric_card("Highest similarity", f"{max_score:.1%}", "Strongest semantic match", TERTIARY)
        with d3:
            metric_card("Average similarity", f"{dup_work['similarity_score'].mean():.1%}", "Across current candidate pairs", SECONDARY)

        st.write("")
        control_col, note_col = st.columns([0.7, 1.3], gap="large")
        with control_col:
            threshold = st.slider(
                "Visualization threshold",
                min_value=0.0,
                max_value=1.0,
                value=max(0.80, min_score),
                step=0.01,
                help="This only filters the visualization on this page. It does not change the backend duplicate threshold.",
            )
        with note_col:
            st.markdown(
                """
<div class="note-box">
    The similarity map below is a supporting visualization of the FAISS-based duplicate candidates. It does not introduce a new prediction model or change the stored duplicate decisions.
</div>
""",
                unsafe_allow_html=True,
            )

        graph_fig = build_similarity_graph(dup_work, min_similarity=threshold, max_pairs=40)
        if graph_fig is not None:
            st.markdown("#### Similarity cluster view")
            st.caption("Connected nodes represent report pairs above the selected display threshold. Cluster colours are visual groupings only.")
            st.plotly_chart(graph_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Candidate pairs")
        pair_display = dup_work.copy()
        pair_display["similarity_score"] = pair_display["similarity_score"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        pair_display = pair_display.rename(
            columns={
                "report_1_id": "Report A",
                "report_1_text": "Report A summary",
                "report_2_id": "Report B",
                "report_2_text": "Report B summary",
                "similarity_score": "Similarity",
            }
        )
        st.dataframe(pair_display, use_container_width=True, height=300, hide_index=True)

        st.markdown("#### Pair review")
        sorted_dups = dup_work.sort_values("similarity_score", ascending=False).head(12).reset_index(drop=True)
        for _, row in sorted_dups.iterrows():
            label = f"Report #{row['report_1_id']} vs #{row['report_2_id']}  /  {row['similarity_score']:.1%} similarity"
            with st.expander(label):
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.markdown(f"**Report #{row['report_1_id']}**")
                    st.write(str(row["report_1_text"]))
                with c2:
                    st.markdown(f"**Report #{row['report_2_id']}**")
                    st.write(str(row["report_2_text"]))
                st.caption("Recommendation: review this pair before confirming that both reports describe the same underlying defect.")

        st.markdown("#### Duplicate-detection evaluation")
        dup_eval = parse_duplicate_eval(duplicate_eval_payload)
        if dup_eval:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card("Precision", f"{dup_eval['precision']:.1%}" if dup_eval['precision'] is not None else "N/A", "Correctness of duplicate flags", PRIMARY)
            with m2:
                metric_card("Recall", f"{dup_eval['recall']:.1%}" if dup_eval['recall'] is not None else "N/A", "Coverage of true duplicates", SECONDARY)
            with m3:
                metric_card("F1 score", f"{dup_eval['f1']:.2f}" if dup_eval['f1'] is not None else "N/A", "Precision-recall balance", TERTIARY)
            with m4:
                metric_card("Threshold", f"{dup_eval['threshold']:.2f}" if dup_eval['threshold'] is not None else "N/A", "Evaluated similarity cut-off", "#475569")
        else:
            st.markdown(
                """
<div class="research-card">
    <strong>Awaiting labelled duplicate-evaluation output.</strong><br>
    This section is intentionally not populated with estimated values. Once the backend exposes ground-truth duplicate results at <code>/evaluation/duplicates</code>, the dashboard will show precision, recall, F1 and the evaluated similarity threshold here.
</div>
""",
                unsafe_allow_html=True,
            )

        with st.expander("How duplicate-detection metrics are calculated"):
            st.markdown(
                """
                **True Positive (TP):** a real duplicate pair correctly flagged as duplicate.  
                **False Positive (FP):** a non-duplicate pair incorrectly flagged as duplicate.  
                **False Negative (FN):** a real duplicate pair that the system missed.

                **Precision** = TP / (TP + FP)  
                Measures how many flagged duplicate pairs are actually duplicates.

                **Recall** = TP / (TP + FN)  
                Measures how many real duplicate pairs the system successfully finds.

                **F1-score** = 2 x (Precision x Recall) / (Precision + Recall)  
                Summarises the balance between precision and recall.

                **Similarity threshold**  
                The backend compares semantic similarity against a selected cut-off. Threshold testing should use labelled duplicate relationships so the final threshold balances missed duplicates against false matches.
                """
            )


# ============================================================
# Model evaluation
# ============================================================

with eval_tab:
    st.markdown("### Model benchmark and evaluation")
    st.caption("Compare the LLM severity classifier with the traditional machine-learning baselines using the current evaluation results.")

    if eval_err or eval_df.empty:
        all_results = {
            "LLM (Zero-Shot)": {
                "accuracy": 0.5634,
                "macro_f1": 0.22,
                "macro_precision": 0.24,
                "macro_recall": 0.23,
            }
        }
        all_results.update(ML_BASELINE_RESULTS)
        if eval_err:
            st.caption("Live evaluation results were unavailable, so the dashboard is displaying the last stored benchmark values.")
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

    metrics_df = pd.DataFrame(all_results).T.reset_index().rename(columns={"index": "Model"})
    metrics_df = metrics_df[["Model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]]
    metrics_df.columns = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
    best_f1 = metrics_df.loc[metrics_df["Macro F1"].idxmax()]
    llm_rows = metrics_df[metrics_df["Model"].str.contains("LLM", case=False, na=False)]
    llm_f1 = float(llm_rows.iloc[0]["Macro F1"]) if not llm_rows.empty else float("nan")

    e1, e2, e3 = st.columns(3)
    with e1:
        metric_card("Best accuracy", f"{best_acc['Accuracy']:.1%}", str(best_acc["Model"]), PRIMARY)
    with e2:
        metric_card("Best Macro F1", f"{best_f1['Macro F1']:.2f}", str(best_f1["Model"]), SECONDARY)
    with e3:
        metric_card("LLM Macro F1", f"{llm_f1:.2f}" if pd.notna(llm_f1) else "N/A", "Zero-shot severity classifier", TERTIARY)

    st.write("")
    chart_col, table_col = st.columns([1.0, 1.15], gap="large")

    with chart_col:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Accuracy",
                x=metrics_df["Model"],
                y=metrics_df["Accuracy"],
                marker_color=PRIMARY,
            )
        )
        fig.add_trace(
            go.Bar(
                name="Macro F1",
                x=metrics_df["Model"],
                y=metrics_df["Macro F1"],
                marker_color=SECONDARY,
            )
        )
        fig.update_layout(
            barmode="group",
            height=390,
            margin=dict(l=10, r=10, t=10, b=45),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#E2E8F0"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=1.08, x=0),
            font=dict(color=PRIMARY),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with table_col:
        formatted = metrics_df.copy()
        formatted["Accuracy"] = formatted["Accuracy"].map(lambda x: f"{x:.1%}")
        for col in ["Macro Precision", "Macro Recall", "Macro F1"]:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2f}")
        st.dataframe(formatted, use_container_width=True, hide_index=True, height=255)

        st.markdown(
            """
<div class="note-box">
    Accuracy should be interpreted together with macro-averaged metrics when the severity classes are imbalanced. Macro F1 gives each severity class equal influence and is therefore useful for comparing performance across Critical, High, Medium and Low reports.
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("How classification metrics are calculated"):
        st.markdown(
            """
            **Accuracy** = Correct predictions / Total predictions  
            Measures the overall proportion of severity predictions that are correct.

            For each severity class:

            **Precision** = TP / (TP + FP)  
            Of the reports predicted as that severity, how many were correct?

            **Recall** = TP / (TP + FN)  
            Of the reports that truly belong to that severity, how many did the model find?

            **F1-score** = 2 x (Precision x Recall) / (Precision + Recall)

            **Macro Precision / Recall / F1** = arithmetic mean of the corresponding class-level scores across Critical, High, Medium and Low. Each severity class receives equal weight, which is useful when the dataset is imbalanced.

            **Confusion matrix**  
            Rows represent the actual severity labels and columns represent the predicted severity labels. Diagonal cells are correct predictions; off-diagonal cells show which severity classes are being confused.
            """
        )

    st.markdown("#### Confusion matrix")
    labels, matrix, matrix_model = parse_confusion_payload(confusion_payload)
    if labels is not None and matrix is not None:
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale=[[0, "#EFF6FF"], [0.35, "#93C5FD"], [1, "#0F172A"]],
                showscale=True,
                text=matrix,
                texttemplate="%{text}",
                hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
            )
        )
        cm_fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=20, b=45),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis_title="Predicted severity",
            yaxis_title="Actual severity",
            font=dict(color=PRIMARY),
        )
        st.caption(f"Confusion matrix for {matrix_model}. Rows are actual classes and columns are predicted classes.")
        st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            """
<div class="research-card">
    <strong>Awaiting confusion-matrix output from the evaluation backend.</strong><br>
    The dashboard does not fabricate class-level counts. When <code>/evaluation/confusion-matrix</code> returns the actual labels and matrix, this panel will render them automatically as a heatmap.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("#### Evaluation completeness")
    st.markdown(
        """
<div class="note-box teal">
    The dashboard now has dedicated locations for all evaluation evidence required by the project: classification accuracy, precision, recall and Macro F1; a confusion matrix; and separate duplicate-detection precision, recall and F1. Missing research results are shown as unavailable rather than replaced with invented values.
</div>
""",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div style="margin-top:28px;padding-top:14px;border-top:1px solid #E2E8F0;color:#94A3B8;font-size:0.72rem;">
    Triage Console is a research decision-support prototype for severity classification, semantic duplicate detection and priority ranking. Model outputs support, rather than replace, reviewer judgement.
</div>
""",
    unsafe_allow_html=True,
)