import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Triage Console", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# This version reads from the FastAPI + SQLite backend (localhost:8000)
# instead of the CSV files directly. Run the backend first:
#   cd bug-triaging-backend
#   uvicorn app.main:app --reload --port 8000
# If the API isn't running, every page shows a clear error instead of
# silently falling back to stale data.
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"

# Static results from your earlier traditional ML baseline run (Step 9).
# Kept as a fallback in case /evaluation isn't reachable, so the Evaluation
# tab still renders something useful even if the backend is briefly down.
ML_BASELINE_RESULTS = {
    "Naive Bayes":    {"accuracy": 0.8592, "macro_f1": 0.23, "macro_precision": 0.21, "macro_recall": 0.25},
    "SVM":            {"accuracy": 0.8169, "macro_f1": 0.22, "macro_precision": 0.21, "macro_recall": 0.24},
    "Random Forest":  {"accuracy": 0.8592, "macro_f1": 0.23, "macro_precision": 0.21, "macro_recall": 0.25},
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

.api-pill { display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; padding:4px 10px; border-radius:20px; }
.api-pill-ok { background: rgba(74,222,128,0.12); color:#4ADE80; }
.api-pill-down { background: rgba(248,113,113,0.12); color:#F87171; }
</style>
""", unsafe_allow_html=True)


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
            "Can't reach the API at " + API_BASE + ". Is the backend running? "
            "In a terminal: `cd bug-triaging-backend` then `uvicorn app.main:app --reload --port 8000`"
        )
    except requests.exceptions.Timeout:
        return None, "API request timed out."
    except requests.exceptions.HTTPError as e:
        return None, f"API returned an error: {e}"


def api_post(path, json_body, timeout=180):
    """POST to the FastAPI backend. Long timeout — first classification call loads bart-large-mnli."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=timeout)
        if resp.status_code >= 400:
            return None, f"API error ({resp.status_code}): {resp.text}"
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the API at {API_BASE}. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "Classification timed out — the model may still be loading. Try again in a moment."


@st.cache_data(ttl=30)
def load_reports():
    """All reports, used for the Overview tab metrics/charts (unfiltered)."""
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
    data, err = api_get("/stats")
    return data, err


SEVERITY_BADGE = {
    "Critical": ("badge-critical", "ticket-critical", "🔴"),
    "High": ("badge-high", "ticket-high", "🟠"),
    "Medium": ("badge-medium", "ticket-medium", "🟡"),
    "Low": ("badge-low", "ticket-low", "⚪"),
}
# (Low kept the white dot here to match your earlier palette; harmless either way.)
SEVERITY_BADGE["Low"] = ("badge-low", "ticket-low", "🟢")


def render_ticket(row, has_confidence):
    badge_class, ticket_class, dot = SEVERITY_BADGE.get(row["predicted_severity"], ("badge-medium", "ticket-medium", "⚪"))
    dup_html = '<span class="dup-flag">⛓ DUPLICATE</span>' if row["is_duplicate"] else ""
    text_preview = str(row["full_text"])[:160].replace("<", "").replace(">", "")
    conf_html = f'<span class="confidence-score">confidence → {row["model_confidence"]:.0%}</span>' if has_confidence and pd.notna(row.get("model_confidence")) else ""

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


# ============================================================
# Header + API status pill
# ============================================================

stats, stats_err = load_stats()

st.markdown("""
<div class="console-header"><h1>🛰️ Triage Console</h1></div>
<div class="console-sub">LLM-ASSISTED BUG REPORT TRIAGING &amp; PRIORITIZATION SYSTEM — LIVE API</div>
""", unsafe_allow_html=True)

if stats_err:
    st.markdown(f'<span class="api-pill api-pill-down">● API offline</span>', unsafe_allow_html=True)
    st.error(stats_err)
    st.stop()
else:
    st.markdown(f'<span class="api-pill api-pill-ok">● API connected — {API_BASE}</span>', unsafe_allow_html=True)

st.write("")

df, df_err = load_reports()
if df_err:
    st.error(df_err)
    st.stop()

# Backend always returns model_confidence (may be null for empty text), so this
# stays True as long as the column exists — mirrors the original has_confidence flag.
has_confidence = "model_confidence" in df.columns
duplicates_df, dup_err = load_duplicates()
eval_df, eval_err = load_evaluation()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("### ⚙ Filters")
severity_options = ["Critical", "High", "Medium", "Low"]
selected_severities = st.sidebar.multiselect("Severity", severity_options, default=severity_options)
min_priority, max_priority = float(df["priority_score"].min()), float(df["priority_score"].max())
priority_range = st.sidebar.slider("Priority score range", min_priority, max_priority, (min_priority, max_priority))
show_duplicates_only = st.sidebar.checkbox("Duplicates only", value=False)
search_term = st.sidebar.text_input("Search by keyword or bug ID", "")
repo_options = sorted(df["repo"].dropna().unique().tolist())
selected_repos = st.sidebar.multiselect("Repository", repo_options, default=repo_options)

# Filters are applied client-side against the already-fetched DataFrame —
# matches the original behavior exactly, just sourced from the API instead of CSV.
filtered_df = df[df["predicted_severity"].isin(selected_severities)]
filtered_df = filtered_df[filtered_df["repo"].isin(selected_repos)]
filtered_df = filtered_df[
    (filtered_df["priority_score"] >= priority_range[0]) & (filtered_df["priority_score"] <= priority_range[1])
]
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
    (m1, "Total Reports", stats["total_reports"]),
    (m2, "Critical", stats["critical_count"]),
    (m3, "Duplicates Found", stats["duplicates_found"]),
    (m4, "Currently Showing", len(filtered_df)),
]
for col, label, value in metrics:
    with col:
        st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📋 All Reports", "🔁 Duplicates", "🧪 Evaluation", "➕ Submit Report"])

with tab1:
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("##### Severity Breakdown")
        severity_counts = df["predicted_severity"].value_counts().reindex(severity_options).fillna(0)
        st.bar_chart(severity_counts, color="#5EEAD4")
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
    if dup_err:
        st.error(dup_err)
    elif not duplicates_df.empty:
        st.markdown(f"##### {len(duplicates_df)} duplicate pairs detected via semantic similarity")
        st.caption("Each pair shows the matched report and how similar the system judged them to be (1.0 = identical meaning). Click any pair below to see the full reference comparison.")
        st.dataframe(duplicates_df.sort_values("similarity_score", ascending=False), use_container_width=True, height=350)

        st.write("")
        st.markdown("###### 🔗 Reference comparison")
        sorted_dups = duplicates_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
        for idx, row in sorted_dups.iterrows():
            label = f"#{row['report_1_id']} ↔ #{row['report_2_id']}  ·  similarity {row['similarity_score']:.0%}"
            with st.expander(label):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Report #{row['report_1_id']}** (original)")
                    st.markdown(f'<div class="ticket-text">{row["report_1_text"]}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Report #{row['report_2_id']}** (flagged as duplicate)")
                    st.markdown(f'<div class="ticket-text">{row["report_2_text"]}</div>', unsafe_allow_html=True)
    else:
        st.info("No duplicate pairs returned by the API yet.")

with tab4:
    st.markdown("##### Model Performance Comparison")
    st.caption("Severity classification evaluated on the labeled Eclipse dataset (Zenodo), served live from GET /evaluation.")

    if eval_err or eval_df.empty:
        if eval_err:
            st.caption(f"⚠️ Could not load live evaluation results ({eval_err}) — showing last known values instead.")
        all_results = {
            "LLM (Zero-Shot)": {"accuracy": 0.5634, "macro_f1": 0.22, "macro_precision": 0.24, "macro_recall": 0.23}
        }
        all_results.update(ML_BASELINE_RESULTS)
    else:
        all_results = {
            row["model_name"]: {
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "macro_precision": row["macro_precision"],
                "macro_recall": row["macro_recall"],
            }
            for _, row in eval_df.iterrows()
        }

    cols = st.columns(len(all_results))
    for col, (name, scores) in zip(cols, all_results.items()):
        with col:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-model-name">{name}</div>
                <div style="margin-top:10px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748B;">ACCURACY</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700;">{scores['accuracy']:.1%}</div>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:{scores['accuracy']*100}%; background:#5EEAD4;"></div></div>

                <div style="margin-top:12px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748B;">MACRO PRECISION</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;">{scores['macro_precision']:.2f}</div>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:{scores['macro_precision']*100}%; background:#FBBF24;"></div></div>

                <div style="margin-top:12px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748B;">MACRO RECALL</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;">{scores['macro_recall']:.2f}</div>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:{scores['macro_recall']*100}%; background:#FB923C;"></div></div>

                <div style="margin-top:12px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748B;">MACRO F1-SCORE</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;">{scores['macro_f1']:.2f}</div>
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

with tab5:
    st.markdown("##### Submit a new bug report for live classification")
    st.caption(
        "Sends the report to `POST /reports` on the live backend — runs severity classification "
        "(facebook/bart-large-mnli), checks for duplicates against the existing corpus (FAISS + MiniLM), "
        "and computes a priority score, exactly as your batch scripts do."
    )

    with st.form("submit_report_form", clear_on_submit=False):
        repo_input = st.text_input("Repository", value="microsoft/vscode")
        text_input = st.text_area(
            "Bug report text",
            height=160,
            placeholder="e.g. Editor crashes immediately on startup, complete data loss of unsaved work",
        )
        submitted = st.form_submit_button("Classify & Submit", type="primary")

    if submitted:
        if not text_input.strip():
            st.warning("Please enter some bug report text before submitting.")
        else:
            with st.spinner("Classifying… first call can take a minute or more while the model loads."):
                result, err = api_post("/reports", {"repo": repo_input, "full_text": text_input})

            if err:
                st.error(err)
            else:
                badge_class, ticket_class, dot = SEVERITY_BADGE.get(
                    result["predicted_severity"], ("badge-medium", "ticket-medium", "⚪")
                )
                dup_note = ""
                if result["is_duplicate"]:
                    dup_note = (
                        f'<div class="dup-flag" style="margin-top:8px;">⛓ Flagged as a likely duplicate of '
                        f'report #{result["matched_duplicate_id"]} '
                        f'(similarity {result["similarity_score"]:.0%})</div>'
                    )

                st.markdown(f"""
                <div class="ticket {ticket_class}" style="margin-top:14px;">
                    <div class="ticket-top">
                        <span class="badge {badge_class}">{dot} {result['predicted_severity']}</span>
                        <span class="confidence-score">confidence → {result['model_confidence']:.0%}</span>
                    </div>
                    <div class="ticket-text">New report #{result['id']} — {repo_input}</div>
                    <div class="meta-row">
                        <span class="priority-score">priority → {result['priority_score']}</span>
                    </div>
                    {dup_note}
                </div>
                """, unsafe_allow_html=True)

                st.success("Submitted and stored in the database. Refresh the other tabs (or wait ~30s for cache to expire) to see it included.")
                # Clear cached data so Overview/All Reports/Duplicates reflect the new report
                load_reports.clear()
                load_duplicates.clear()
                load_stats.clear()
