import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.ingestion.pdf_parser import parse_all_resumes, load_all_jds
from src.embeddings.embedder import index_resume_chunks, is_index_populated, delete_index_namespace
from src.retrieval.retriever import retrieve_candidates, group_chunks_by_candidate
from src.evaluation.evaluator import evaluate_all_candidates, run_adaptive_ragas_loop
from src.models.schemas import PipelineConfig, SelfRAGResult, CandidateEvaluation

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUMES_DIR = os.path.join(BASE_DIR, "data", "resumes")
JDS_DIR = os.path.join(BASE_DIR, "data", "job_descriptions")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _score_label(score: float) -> str:
    if score >= 0.75:
        return "STRONG FIT"
    if score >= 0.60:
        return "MODERATE FIT"
    return "WEAK FIT"


def _print_divider(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _print_result(rank: int, result: SelfRAGResult) -> None:
    ev: CandidateEvaluation = result.evaluation
    score_pct = int(ev.score * 100)
    label = _score_label(ev.score)
    flag = " [LOW CONFIDENCE]" if ev.low_confidence_flag else ""

    _print_divider()
    print(f"#{rank}  {ev.candidate_name:<30}  Score: {score_pct}%  |  {label}{flag}")
    _print_divider()

    print("\nSTRENGTHS:")
    for s in ev.strengths:
        print(f"  + {s}")

    print("\nGAPS:")
    for g in ev.gaps:
        print(f"  - {g}")

    print("\nSUMMARY:")
    print(f"  {ev.summary}")

    print(
        f"\n  [Self-RAG] attempts={result.attempts_used} | "
        f"relevant={result.critique.is_relevant} | "
        f"supported={result.critique.is_supported} | "
        f"usefulness={result.critique.usefulness_score}/5"
    )
    print()


def _print_ragas(scores, config: PipelineConfig) -> None:
    _print_divider("=")
    print("RAGAS EVALUATION SCORES")
    _print_divider("=")

    metrics = [
        ("Context Precision", scores.context_precision, config.threshold_context_precision),
        ("Context Recall",    scores.context_recall,    config.threshold_context_recall),
        ("Faithfulness",      scores.faithfulness,      config.threshold_faithfulness),
        ("Answer Relevancy",  scores.answer_relevancy,  config.threshold_answer_relevancy),
    ]

    for name, value, threshold in metrics:
        if value is None:
            print(f"  {name:<22}: N/A")
            continue
        status = "PASS" if value >= threshold else "FAIL"
        bar_filled = int(value * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  {name:<22}: [{bar}] {value:.2f}  (>={threshold})  {status}")

    failing = scores.failing_metrics(config)
    print()
    if not failing:
        print("  All metrics passed.")
    else:
        print(f"  Failing: {', '.join(failing)}")
        print("  Config was auto-adjusted for next run.")
    _print_divider("=")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_index(args) -> None:
    """Parse all resumes and upsert vectors into Pinecone."""
    if is_index_populated() and not args.force:
        print("Index already populated. Use --force to re-index.")
        return

    if args.force:
        print("Wiping existing index...")
        delete_index_namespace()

    print(f"Parsing resumes from: {RESUMES_DIR}")
    chunks = parse_all_resumes(RESUMES_DIR)
    unique_candidates = len(set(c.metadata.candidate_id for c in chunks))
    print(f"Parsed {unique_candidates} resumes into {len(chunks)} chunks.")

    print("Embedding and upserting to Pinecone (this may take a few minutes)...")
    count = index_resume_chunks(chunks)
    print(f"Done. {count} vectors indexed.")


def cmd_search(args) -> None:
    """Run the full pipeline: retrieve → evaluate → display → optional RAGAS."""
    if not is_index_populated():
        print("ERROR: Pinecone index is empty. Run:  python main.py index")
        return

    # Resolve JD text
    if args.jd_file:
        jd_path = args.jd_file if os.path.isabs(args.jd_file) else os.path.join(JDS_DIR, args.jd_file)
        with open(jd_path, encoding="utf-8") as f:
            jd_text = f.read().strip()
        jd_title = os.path.basename(jd_path).replace(".txt", "").replace("_", " ").title()
    elif args.jd_text:
        jd_text = args.jd_text
        jd_title = "Custom JD"
    else:
        # Interactive prompt
        print("Paste the job description below. Enter a blank line when done:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        jd_text = "\n".join(lines)
        jd_title = input("JD Title (optional, press Enter to skip): ").strip() or "Custom JD"

    config = PipelineConfig(
        top_k=args.top_k,
        use_hybrid_search=not args.no_hybrid,
        use_reranker=not args.no_reranker,
        use_compression=not args.no_compression,
        use_mmr_dedup=not args.no_mmr,
        temperature=args.temperature,
    )

    # --- Retrieve ---
    print(f"\nSearching for candidates matching: {jd_title}")
    print("Loading resume corpus...")
    all_chunks = parse_all_resumes(RESUMES_DIR)

    print("Running retrieval pipeline...")
    retrieved = retrieve_candidates(jd_text, all_chunks, config)
    grouped = group_chunks_by_candidate(retrieved)
    print(f"Retrieved {len(grouped)} unique candidates.\n")

    # --- Evaluate ---
    print("Evaluating candidates with Self-RAG (this may take a few minutes)...")
    results = evaluate_all_candidates(grouped, jd_text, config)

    # --- Display ---
    _print_divider("=")
    print(f"  HIREFLOW RESULTS  —  {jd_title}  ({len(results)} candidates)")
    _print_divider("=")
    for rank, result in enumerate(results[: args.top_n], 1):
        _print_result(rank, result)

    # --- Optional RAGAS ---
    if args.ragas:
        print("\nRunning RAGAS adaptive evaluation loop...")
        scores, best_config = run_adaptive_ragas_loop(results, grouped, jd_text, config)
        _print_ragas(scores, best_config)


def cmd_list_jds(args) -> None:
    """List all available JD files."""
    jds = load_all_jds(JDS_DIR)
    print(f"\nAvailable JDs in {JDS_DIR}:\n")
    for i, name in enumerate(jds, 1):
        print(f"  {i}. {name}.txt")
    print()


def cmd_status(args) -> None:
    """Show current index and data status."""
    pdf_count = len(list(Path(RESUMES_DIR).glob("*.pdf")))
    jd_count = len(list(Path(JDS_DIR).glob("*.txt")))
    indexed = is_index_populated()

    _print_divider("=")
    print("  HIREFLOW STATUS")
    _print_divider("=")
    print(f"  Resumes in corpus : {pdf_count}")
    print(f"  JDs available     : {jd_count}")
    print(f"  Pinecone indexed  : {'YES' if indexed else 'NO — run: python main.py index'}")
    _print_divider("=")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hireflow",
        description="HireFlow — Intelligent Candidate Search (CLI)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_index = sub.add_parser("index", help="Parse resumes and index into Pinecone")
    p_index.add_argument("--force", action="store_true", help="Wipe existing index and re-index")
    p_index.set_defaults(func=cmd_index)

    # --- search ---
    p_search = sub.add_parser("search", help="Search candidates for a job description")
    jd_group = p_search.add_mutually_exclusive_group()
    jd_group.add_argument("--jd-file", metavar="FILENAME", help="JD filename from data/job_descriptions/ or absolute path")
    jd_group.add_argument("--jd-text", metavar="TEXT", help="JD text passed directly as a string")
    p_search.add_argument("--top-k", type=int, default=8, help="Chunks to retrieve (default: 8)")
    p_search.add_argument("--top-n", type=int, default=10, help="Candidates to display (default: 10)")
    p_search.add_argument("--temperature", type=float, default=0.3, help="LLM temperature (default: 0.3)")
    p_search.add_argument("--no-hybrid", action="store_true", help="Disable hybrid BM25 search")
    p_search.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    p_search.add_argument("--no-compression", action="store_true", help="Disable context compression")
    p_search.add_argument("--no-mmr", action="store_true", help="Disable MMR deduplication")
    p_search.add_argument("--ragas", action="store_true", help="Run RAGAS evaluation after search")
    p_search.set_defaults(func=cmd_search)

    # --- list-jds ---
    p_list = sub.add_parser("list-jds", help="List available JD files")
    p_list.set_defaults(func=cmd_list_jds)

    # --- status ---
    p_status = sub.add_parser("status", help="Show corpus and index status")
    p_status.set_defaults(func=cmd_status)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
