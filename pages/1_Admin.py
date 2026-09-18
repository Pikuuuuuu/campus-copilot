"""Admin console: product analytics, content gaps, feedback review, document management."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core import config, db
from core.index_builder import rebuild_index
from core.ingest import load_chunks

st.set_page_config(page_title="Admin | Campus Copilot", page_icon="📊", layout="wide")

if not st.session_state.get("admin_ok"):
    st.title("📊 Admin console")
    pw = st.text_input("Admin password", type="password")
    if st.button("Enter"):
        if pw == config.ADMIN_PASSWORD:
            st.session_state.admin_ok = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()

st.title("📊 Admin console")
tab_metrics, tab_gaps, tab_feedback, tab_docs, tab_eval = st.tabs(
    ["Usage", "Content gaps", "Feedback", "Documents", "Evaluation"]
)


def show_sql(name: str) -> None:
    with st.expander("SQL behind this view"):
        st.code(db.ANALYTICS_QUERIES[name].strip(), language="sql")


# ---------------------------------------------------------------- Usage
with tab_metrics:
    head = db.run_query("headline")[0]
    c = st.columns(5)
    c[0].metric("Questions", head["total_questions"] or 0)
    c[1].metric("Sessions", head["sessions"] or 0)
    c[2].metric("Answer rate", f"{head['answer_rate_pct'] or 0}%",
                help="Answered ÷ (answered + not found). Excludes out-of-scope and blocked.")
    c[3].metric("Helpful", f"{head['helpful_pct'] or 0}%", help=f"From {head['feedback_count']} ratings")
    c[4].metric("Avg latency", f"{head['avg_latency_ms'] or 0} ms")
    show_sql("headline")

    left, right = st.columns(2)
    with left:
        st.subheader("Questions per day")
        daily = pd.DataFrame(db.run_query("daily"))
        if not daily.empty:
            st.line_chart(daily.set_index("day")[["questions", "answered"]])
        else:
            st.caption("No data yet.")
        show_sql("daily")
    with right:
        st.subheader("By topic")
        cats = pd.DataFrame(db.run_query("by_category"))
        if not cats.empty:
            st.bar_chart(cats.set_index("category")["questions"])
            st.dataframe(cats, hide_index=True, width="stretch")
        else:
            st.caption("No data yet.")
        show_sql("by_category")

# ---------------------------------------------------------------- Gaps
with tab_gaps:
    st.markdown(
        "Questions students asked that the documents **could not answer**, ranked by demand. "
        "Each row is either a missing document to add or a question to route to an office."
    )
    gaps = pd.DataFrame(db.run_query("content_gaps"))
    if gaps.empty:
        st.success("No unanswered questions yet.")
    else:
        st.dataframe(gaps, hide_index=True, width="stretch")
        st.download_button("Download CSV", gaps.to_csv(index=False), "content_gaps.csv", "text/csv")
    show_sql("content_gaps")

# ---------------------------------------------------------------- Feedback
with tab_feedback:
    st.markdown("Answers students marked 👎. Review these first: each one is a trust problem.")
    neg = pd.DataFrame(db.run_query("negative_feedback"))
    if neg.empty:
        st.success("No negative feedback yet.")
    else:
        st.dataframe(neg, hide_index=True, width="stretch")
    show_sql("negative_feedback")

# ---------------------------------------------------------------- Documents
with tab_docs:
    chunks, meta = load_chunks()
    st.markdown(
        f"**{len({c.source for c in chunks})} documents · {len(chunks)} chunks** · "
        f"built {meta.get('built_at', 'never')}"
    )
    if chunks:
        summary = (
            pd.DataFrame([{"document": c.title, "file": c.source, "topic": c.category} for c in chunks])
            .value_counts()
            .reset_index(name="chunks")
        )
        st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Add a document")
    up_cat = st.selectbox("Topic", config.CATEGORIES)
    up_file = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])
    up_url = st.text_input("Official URL (shown in citations)", placeholder="https://nitrkl.ac.in/docs/…")
    up_title = st.text_input("Display title", placeholder="UG Academic Regulations 2019")
    if st.button("Save document", disabled=up_file is None):
        target = config.RAW_DIR / up_cat / up_file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(up_file.getvalue())
        manifest_path = config.RAW_DIR / "sources.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest[up_file.name] = {"title": up_title or target.stem, "url": up_url}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        st.success(f"Saved to {target.relative_to(config.ROOT)}. Re-index to make it searchable.")

    if st.button("🔄 Re-index all documents", type="primary"):
        with st.spinner("Chunking and embedding…"):
            report = rebuild_index()
        st.cache_resource.clear()
        st.success(report)
    st.caption("On Streamlit Community Cloud, uploaded files reset when the app restarts. "
               "For permanent changes, add the file to the repo and run scripts/ingest.py locally.")

# ---------------------------------------------------------------- Eval
with tab_eval:
    latest = config.ROOT / "eval" / "results" / "latest.json"
    if not latest.exists():
        st.info("No evaluation run yet. Run `python eval/run_eval.py`.")
    else:
        report = json.loads(latest.read_text())
        m = report["metrics"]
        c = st.columns(5)
        c[0].metric("Retrieval hit@k", f"{m['retrieval_hit_rate']}%")
        c[1].metric("Answer correctness", f"{m['answer_correctness']}%")
        c[2].metric("Refusal accuracy", f"{m['refusal_accuracy']}%")
        c[3].metric("p50 latency", f"{m['latency_p50_ms']} ms")
        c[4].metric("p95 latency", f"{m['latency_p95_ms']} ms")
        st.caption(f"Run at {report['run_at']} · {report['n']} questions · {report['retrieval_mode']}")
        st.dataframe(pd.DataFrame(report["rows"]), hide_index=True, width="stretch")
