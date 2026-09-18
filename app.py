"""Campus Copilot: student-facing chat.  Run:  streamlit run app.py"""
from __future__ import annotations

import uuid
from collections import Counter

import streamlit as st

from core import config, db
from core.ingest import load_chunks
from core.rag import CampusCopilot

st.set_page_config(page_title="Campus Copilot | NIT Rourkela", page_icon="🧭", layout="centered")

STARTERS = [
    "What is the minimum attendance required to sit for end-semester exams?",
    "When does semester registration start for Autumn 2026-27?",
    "How is CGPA calculated?",
    "What are the rules for change of branch?",
]
STATUS_BADGE = {
    "answered": "✅ Answered from official documents",
    "not_found": "🔎 Not in the documents I have",
    "out_of_scope": "🧭 Outside what I can help with",
}


@st.cache_resource(show_spinner="Loading institute documents…")
def get_copilot(index_stamp: str) -> CampusCopilot:  # index_stamp busts the cache after re-indexing
    chunks, _meta = load_chunks()
    return CampusCopilot(chunks)


def index_stamp() -> str:
    return str(config.CHUNKS_PATH.stat().st_mtime) if config.CHUNKS_PATH.exists() else "empty"


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])  # {role, content, result?}
ss.setdefault("pending", None)
ss.setdefault("feedback_given", set())

copilot = get_copilot(index_stamp())
chunks, meta = load_chunks()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Campus Copilot")
    st.caption("Answers from official NIT Rourkela documents, with the page it came from.")

    topic = st.selectbox(
        "Limit to a topic",
        ["All topics"] + [c.title() for c in config.CATEGORIES],
        help="Narrowing the topic makes answers more precise when a term means different things in different rules.",
    )
    category = None if topic == "All topics" else topic.lower()

    if st.button("🗒️ New conversation", width="stretch"):
        ss.messages, ss.pending = [], None
        st.rerun()

    st.divider()
    docs = Counter((c.title, c.category) for c in chunks)
    st.markdown(f"**Documents indexed:** {len({c.source for c in chunks})}")
    for (title, cat), _n in sorted(docs.items(), key=lambda x: x[0][1]):
        st.caption(f"• {title} · _{cat}_")
    if meta.get("built_at"):
        st.caption(f"Last updated: {meta['built_at'][:10]}")
    st.caption(f"Retrieval: {copilot.retriever.mode} · Model: {config.ANSWER_MODEL}")
    st.divider()
    st.caption(
        "⚠️ Always confirm deadlines and money-related rules on nitrkl.ac.in or with the office concerned. "
        "This is a student-built tool, not an official institute service."
    )

# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_result(result, msg_index: int) -> None:
    badge = STATUS_BADGE.get(result.status)
    if badge:
        st.caption(badge)
    st.markdown(result.answer)

    if result.sources:
        with st.expander(f"📄 Sources ({len(result.sources)})"):
            for n, chunk in enumerate(result.sources, start=1):
                label = f"**{chunk.title}**, page {chunk.page}"
                st.markdown(f"{label} · [open official PDF]({chunk.url})" if chunk.url else label)
                st.caption(chunk.text[:350] + ("…" if len(chunk.text) > 350 else ""))

    if result.escalation:
        e = result.escalation
        st.info(f"**Who can help:** {e['office']}  \n{e['how']}")

    if result.follow_ups:
        cols = st.columns(len(result.follow_ups))
        for col, q in zip(cols, result.follow_ups):
            if col.button(q, key=f"fu-{msg_index}-{q}", width="stretch"):
                ss.pending = q
                st.rerun()

    if result.interaction_id and result.status in {"answered", "not_found"}:
        fid = result.interaction_id
        if fid in ss.feedback_given:
            st.caption("Thanks for the feedback!")
            return
        c1, c2, _ = st.columns([1, 1, 6])
        if c1.button("👍", key=f"up-{fid}", help="This helped"):
            db.log_feedback(fid, 1)
            ss.feedback_given.add(fid)
            st.rerun()
        if c2.button("👎", key=f"down-{fid}", help="This was wrong or unhelpful"):
            ss[f"comment-open-{fid}"] = True
        if ss.get(f"comment-open-{fid}"):
            comment = st.text_input("What was wrong? (optional)", key=f"comment-{fid}")
            if st.button("Send feedback", key=f"send-{fid}"):
                db.log_feedback(fid, -1, comment)
                ss.feedback_given.add(fid)
                st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Ask anything about NITR rules")

if not chunks:
    st.warning("No documents indexed yet. Add PDFs to `data/raw/<topic>/` and run `python scripts/ingest.py`.")
    st.stop()
if not config.llm_api_key():
    st.error(
        f"No API key found for provider `{config.LLM_PROVIDER}`. "
        "Add it to `.env` locally, or to Secrets on Streamlit Cloud."
    )
    st.stop()

if not ss.messages:
    st.markdown("Attendance, exams, grading, fees, hostel, placements. Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(STARTERS):
        if cols[i % 2].button(q, key=f"starter-{i}", width="stretch"):
            ss.pending = q
            st.rerun()

for i, msg in enumerate(ss.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("result"):
            render_result(msg["result"], i)
        else:
            st.markdown(msg["content"])

typed = st.chat_input("e.g. Can I take a summer course for a backlog paper?")
question = typed or ss.pending
ss.pending = None

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the official documents…"):
            result = copilot.ask(question, history=history, category=category, session_id=ss.session_id)
        ss.messages.append({"role": "assistant", "content": result.answer, "result": result})
    st.rerun()
