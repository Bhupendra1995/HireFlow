import os
import time

import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv

from src.ingestion.pdf_parser import parse_all_resumes, load_all_jds
from src.embeddings.embedder import index_resume_chunks, is_index_populated, delete_index_namespace
from src.retrieval.retriever import retrieve_candidates, group_chunks_by_candidate
from src.evaluation.evaluator import evaluate_all_candidates, run_adaptive_ragas_loop
from src.models.schemas import (
    CandidateEvaluation,
    ConfidenceLevel,
    PipelineConfig,
    RAGASScores,
    SelfRAGResult,
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUMES_DIR = os.path.join(BASE_DIR, "..", "data", "resumes")
JDS_DIR = os.path.join(BASE_DIR, "..", "data", "job_descriptions")

st.set_page_config(page_title="HireFlow", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_ragas_scores" not in st.session_state:
    st.session_state.last_ragas_scores = None
if "last_config_used" not in st.session_state:
    st.session_state.last_config_used = PipelineConfig()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    if score >= 0.60:
        return "🟡"
    return "🔴"


def _confidence_badge(result: SelfRAGResult) -> str:
    if result.evaluation.low_confidence_flag:
        return "⚠️ LOW CONFIDENCE"
    level = result.evaluation.confidence_level
    return {"HIGH": "✅ HIGH CONFIDENCE", "MEDIUM": "🔶 MEDIUM CONFIDENCE", "LOW": "⚠️ LOW CONFIDENCE"}.get(level, "")


def _metric_bar(label: str, value: float | None, threshold: float) -> None:
    if value is None:
        st.write(f"**{label}:** N/A")
        return
    icon = "✅" if value >= threshold else "⚠️"
    st.progress(value, text=f"{icon} {label}: {value:.2f}  (threshold: {threshold})")


# ---------------------------------------------------------------------------
# Page 1 — Resume Management
# ---------------------------------------------------------------------------

def page_resume_management():
    st.title("📂 Resume Management")
    st.caption("Upload and index your resume corpus. This runs once — vectors are persisted in Pinecone.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Resume Corpus")
        pdf_files = [f for f in os.listdir(RESUMES_DIR) if f.endswith(".pdf")] if os.path.isdir(RESUMES_DIR) else []
        st.metric("Resumes in data/resumes/", len(pdf_files))

        uploaded = st.file_uploader(
            "Upload additional PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded:
            os.makedirs(RESUMES_DIR, exist_ok=True)
            for uf in uploaded:
                with open(os.path.join(RESUMES_DIR, uf.name), "wb") as f:
                    f.write(uf.read())
            st.success(f"Uploaded {len(uploaded)} resume(s).")

    with col2:
        st.subheader("Index Status")
        already_indexed = is_index_populated()
        if already_indexed:
            st.success("Pinecone index is populated.")
        else:
            st.warning("Index is empty — click to index.")

        if st.button("Index All Resumes", type="primary"):
            with st.spinner("Parsing and embedding resumes..."):
                chunks = parse_all_resumes(RESUMES_DIR)
                st.session_state.all_chunks = chunks
                count = index_resume_chunks(chunks)
                st.session_state.index_ready = True
            st.success(f"Indexed {count} vectors from {len(set(c.metadata.candidate_id for c in chunks))} resumes.")

        if already_indexed and st.button("Re-index (wipe & rebuild)", type="secondary"):
            with st.spinner("Deleting existing index..."):
                delete_index_namespace()
                chunks = parse_all_resumes(RESUMES_DIR)
                st.session_state.all_chunks = chunks
                count = index_resume_chunks(chunks)
                st.session_state.index_ready = True
            st.success(f"Re-indexed {count} vectors.")

    # Pre-load chunks into session state if index exists but chunks not loaded
    if already_indexed and not st.session_state.all_chunks:
        with st.spinner("Loading resume corpus into memory..."):
            st.session_state.all_chunks = parse_all_resumes(RESUMES_DIR)
            st.session_state.index_ready = True


# ---------------------------------------------------------------------------
# Page 2 — Candidate Search
# ---------------------------------------------------------------------------

def page_candidate_search():
    st.title("🔍 Candidate Search")

    if not st.session_state.index_ready and not is_index_populated():
        st.warning("Resume index is empty. Go to **Resume Management** and index your resumes first.")
        return

    # --- JD Input ---
    st.subheader("Job Description")
    jd_source = st.radio("JD Source", ["Paste JD Text", "Load from file"], horizontal=True)

    jd_text = ""
    jd_title = ""

    if jd_source == "Paste JD Text":
        jd_title = st.text_input("JD Title (optional)", placeholder="e.g. Senior ML Engineer")
        jd_text = st.text_area("Paste job description here", height=200)
    else:
        available_jds = load_all_jds(JDS_DIR) if os.path.isdir(JDS_DIR) else {}
        if not available_jds:
            st.error("No JD files found in data/job_descriptions/")
            return
        selected = st.selectbox("Select a JD", list(available_jds.keys()))
        jd_text = available_jds[selected]
        jd_title = selected.replace("jd_", "").replace("_", " ").title()
        st.text_area("JD Preview", jd_text, height=150, disabled=True)

    # --- Advanced Settings ---
    with st.expander("⚙️ Search Settings (Advanced)"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top-K Candidates to Retrieve", 4, 20, 8)
            use_hybrid = st.toggle("Hybrid Search (BM25 + Dense)", value=True)
            use_reranker = st.toggle("Cross-Encoder Reranker", value=True)
        with col2:
            use_compression = st.toggle("Context Compression", value=True)
            use_mmr = st.toggle("MMR Deduplication", value=True)
            temperature = st.slider("LLM Temperature", 0.1, 0.9, 0.3, step=0.1)

    config = PipelineConfig(
        top_k=top_k,
        use_hybrid_search=use_hybrid,
        use_reranker=use_reranker,
        use_compression=use_compression,
        use_mmr_dedup=use_mmr,
        temperature=temperature,
    )

    # --- Search ---
    if st.button("🚀 Find Candidates", type="primary", disabled=not jd_text.strip()):
        if not st.session_state.all_chunks:
            with st.spinner("Loading resume corpus..."):
                st.session_state.all_chunks = parse_all_resumes(RESUMES_DIR)
                st.session_state.index_ready = True

        with st.spinner("Retrieving candidates..."):
            retrieved_chunks = retrieve_candidates(
                jd_text, st.session_state.all_chunks, config
            )
            grouped = group_chunks_by_candidate(retrieved_chunks)
            st.info(f"Retrieved {len(grouped)} unique candidates. Running evaluations...")

        with st.spinner("Evaluating candidates with Self-RAG..."):
            results = evaluate_all_candidates(grouped, jd_text, config)

        st.session_state.search_history.append({
            "jd_title": jd_title or "Untitled JD",
            "jd_text": jd_text,
            "results": results,
            "grouped": grouped,
            "config": config,
        })
        st.session_state.last_config_used = config
        st.session_state.chat_history = []

    # --- Display Results ---
    if st.session_state.search_history:
        latest = st.session_state.search_history[-1]
        results: list[SelfRAGResult] = latest["results"]
        grouped = latest["grouped"]
        current_config = latest["config"]

        # Session switcher
        if len(st.session_state.search_history) > 1:
            st.divider()
            st.subheader("Recent JD Sessions")
            for idx, session in enumerate(reversed(st.session_state.search_history)):
                st.write(f"{'→ ' if idx == 0 else ''}**{session['jd_title']}**")

        st.divider()
        st.subheader(f"Results for: {latest['jd_title']}")
        st.caption(f"{len(results)} candidates evaluated")

        for rank, result in enumerate(results, 1):
            ev: CandidateEvaluation = result.evaluation
            score_pct = int(ev.score * 100)
            color = _score_color(ev.score)
            badge = _confidence_badge(result)

            with st.expander(
                f"#{rank}  {ev.candidate_name}  —  {color} {score_pct}%  {badge}",
                expanded=(rank <= 3),
            ):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**✅ Strengths**")
                    for s in ev.strengths:
                        st.markdown(f"- {s}")
                with col2:
                    st.markdown("**⚠️ Gaps**")
                    for g in ev.gaps:
                        st.markdown(f"- {g}")
                st.markdown("**📋 Summary**")
                st.write(ev.summary)
                st.caption(
                    f"Self-RAG: attempts={result.attempts_used} | "
                    f"relevant={result.critique.is_relevant} | "
                    f"supported={result.critique.is_supported} | "
                    f"usefulness={result.critique.usefulness_score}/5"
                )

        # --- Follow-up Q&A ---
        st.divider()
        st.subheader("💬 Ask About These Candidates")
        user_q = st.chat_input("e.g. Why was the top candidate ranked higher than #2?")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            # Build context from top-3 results for the follow-up answer
            context_summary = "\n".join(
                f"#{i+1} {r.evaluation.candidate_name} (score={r.evaluation.score:.0%}): {r.evaluation.summary}"
                for i, r in enumerate(results[:3])
            )
            answer = (
                f"Based on the current JD '{latest['jd_title']}' and the evaluated candidates:\n\n"
                f"{context_summary}\n\n"
                f"*(Full conversational Q&A powered by LangChain ConversationMemory — connect evaluator.py)*"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # --- Run RAGAS ---
        st.divider()
        if st.button("📊 Run RAGAS Evaluation on These Results"):
            with st.spinner("Running RAGAS adaptive evaluation loop..."):
                ragas_scores, best_config = run_adaptive_ragas_loop(
                    results, grouped, latest["jd_text"], current_config
                )
                st.session_state.last_ragas_scores = ragas_scores
                st.session_state.last_config_used = best_config
            st.success("RAGAS evaluation complete. See the Evaluation Dashboard.")


# ---------------------------------------------------------------------------
# Page 3 — Evaluation Dashboard
# ---------------------------------------------------------------------------

def page_evaluation_dashboard():
    st.title("📊 Evaluation Dashboard")
    st.caption("RAGAS pipeline health metrics. Run from the Candidate Search page.")

    scores: RAGASScores | None = st.session_state.last_ragas_scores
    config: PipelineConfig = st.session_state.last_config_used

    if scores is None:
        st.info("No RAGAS evaluation has been run yet. Search for candidates and click **Run RAGAS Evaluation**.")
        return

    st.subheader(f"RAGAS Scores — Iteration {scores.iteration}")

    col1, col2 = st.columns(2)
    with col1:
        _metric_bar("Context Precision", scores.context_precision, config.threshold_context_precision)
        _metric_bar("Context Recall", scores.context_recall, config.threshold_context_recall)
    with col2:
        _metric_bar("Faithfulness", scores.faithfulness, config.threshold_faithfulness)
        _metric_bar("Answer Relevancy", scores.answer_relevancy, config.threshold_answer_relevancy)

    st.divider()

    failing = scores.failing_metrics(config)
    if not failing:
        st.success("All metrics passed thresholds.")
    else:
        st.warning(f"Failing metrics: {', '.join(failing)}")
        remediation_map = {
            "context_precision": "Auto-applied: Reduced Top-K, Reranker ON, Compression ON",
            "context_recall": "Auto-applied: Increased Top-K, Hybrid Search ON, Query Expansion ON",
            "faithfulness": "Auto-applied: Temperature reduced, Compression ON",
            "answer_relevancy": "Auto-applied: MMR Deduplication ON",
        }
        for metric in failing:
            st.info(f"**{metric}** → {remediation_map.get(metric, 'Config adjusted')}")

    st.divider()
    st.subheader("Active Pipeline Configuration")
    st.json(config.model_dump())


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("HireFlow")
    st.sidebar.caption("Intelligent Candidate Search")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["Resume Management", "Candidate Search", "Evaluation Dashboard"],
        index=1,
    )

    if page == "Resume Management":
        page_resume_management()
    elif page == "Candidate Search":
        page_candidate_search()
    else:
        page_evaluation_dashboard()

    st.sidebar.divider()
    st.sidebar.caption("Models: gemini-2.5-flash · gemini-embedding-001 · Pinecone Starter")


if __name__ == "__main__":
    main()
