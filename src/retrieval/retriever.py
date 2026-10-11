import os
import time
from collections import defaultdict

from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.embeddings.embedder import embed_query, _get_pinecone_index
from src.models.schemas import PipelineConfig, ResumeChunk, ResumeMetadata

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hireflow-resumes")

# Local cross-encoder reranker — no API cost
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None

# Free-tier: 15 RPM for Gemini LLM, pad embedding calls too
_QUERY_DELAY_SECONDS = 0.5


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


def _dense_search(
    query_vector: list[float],
    top_k: int,
    namespace: str = "resumes",
) -> list[dict]:
    """Query Pinecone with a dense embedding vector."""
    index = _get_pinecone_index()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return response.matches


def _bm25_search(
    query_text: str,
    all_chunks: list[ResumeChunk],
    top_k: int,
) -> list[tuple[str, float]]:
    """BM25 sparse search over in-memory chunks. Returns [(chunk_id, score)]."""
    tokenized_corpus = [chunk.text.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(all_chunks[i].chunk_id, float(scores[i])) for i in top_indices]


def _reciprocal_rank_fusion(
    dense_matches: list[dict],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge dense and sparse rankings using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = defaultdict(float)

    for rank, match in enumerate(dense_matches):
        rrf_scores[match.id] += 1.0 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(sparse_results):
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def _mmr_deduplicate(
    chunks: list[ResumeChunk],
    top_n: int,
    lambda_param: float = 0.7,
) -> list[ResumeChunk]:
    """
    Maximal Marginal Relevance deduplication.
    Removes near-duplicate chunks from the same candidate.
    """
    seen_candidates: set[str] = set()
    deduplicated: list[ResumeChunk] = []

    for chunk in chunks:
        cid = chunk.metadata.candidate_id
        # Allow max 2 chunks per candidate to preserve context diversity
        candidate_chunks = sum(1 for c in deduplicated if c.metadata.candidate_id == cid)
        if candidate_chunks < 2:
            deduplicated.append(chunk)
            if len(deduplicated) >= top_n:
                break

    return deduplicated


def _rerank(
    query_text: str,
    chunks: list[ResumeChunk],
    top_n: int,
) -> list[ResumeChunk]:
    """Cross-encoder reranking — scores each chunk against the full JD query."""
    reranker = _get_reranker()
    pairs = [[query_text, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]


def _expand_query(query_text: str) -> list[str]:
    """
    Simple rule-based query expansion — generates role-specific sub-queries.
    Avoids an extra LLM call to stay within free-tier rate limits.
    """
    lines = [line.strip() for line in query_text.splitlines() if line.strip()]
    sub_queries = [query_text]
    for line in lines[:5]:
        if any(kw in line.lower() for kw in ["required", "must", "experience", "skills", "qualifications"]):
            sub_queries.append(line)
    return sub_queries[:4]


def retrieve_candidates(
    jd_text: str,
    all_chunks: list[ResumeChunk],
    config: PipelineConfig,
    namespace: str = "resumes",
) -> list[ResumeChunk]:
    """
    Full retrieval pipeline:
      1. Dense search (Pinecone)
      2. Sparse search (BM25) — if hybrid enabled
      3. Reciprocal Rank Fusion
      4. Reranker — if enabled
      5. MMR deduplication — if enabled
    Returns the final ordered list of relevant chunks.
    """
    queries = _expand_query(jd_text) if config.use_query_expansion else [jd_text]
    fetch_k = config.top_k * 2  # over-fetch before reranking/dedup

    # --- Step 1: Dense search across all query variants ---
    all_dense_matches: list[dict] = []
    for query in queries:
        query_vector = embed_query(query)
        matches = _dense_search(query_vector, top_k=fetch_k, namespace=namespace)
        all_dense_matches.extend(matches)
        time.sleep(_QUERY_DELAY_SECONDS)

    # Deduplicate dense matches by id, keep highest score
    dense_seen: dict[str, dict] = {}
    for m in all_dense_matches:
        if m.id not in dense_seen or m.score > dense_seen[m.id].score:
            dense_seen[m.id] = m
    dense_matches = sorted(dense_seen.values(), key=lambda x: x.score, reverse=True)

    # --- Step 2: BM25 sparse search ---
    if config.use_hybrid_search:
        sparse_results = _bm25_search(jd_text, all_chunks, top_k=fetch_k)
        fused = _reciprocal_rank_fusion(dense_matches, sparse_results)
    else:
        fused = [(m.id, m.score) for m in dense_matches]

    # Map fused chunk IDs back to ResumeChunk objects
    chunk_map: dict[str, ResumeChunk] = {c.chunk_id: c for c in all_chunks}
    candidate_chunks: list[ResumeChunk] = [
        chunk_map[cid] for cid, _ in fused if cid in chunk_map
    ][:fetch_k]

    # --- Step 3: Cross-encoder reranking ---
    if config.use_reranker and candidate_chunks:
        candidate_chunks = _rerank(jd_text, candidate_chunks, top_n=fetch_k)

    # --- Step 4: MMR deduplication ---
    if config.use_mmr_dedup and candidate_chunks:
        candidate_chunks = _mmr_deduplicate(candidate_chunks, top_n=config.top_k)

    return candidate_chunks[:config.top_k]


def group_chunks_by_candidate(
    chunks: list[ResumeChunk],
) -> dict[str, list[ResumeChunk]]:
    """Group retrieved chunks by candidate_id for per-candidate evaluation."""
    grouped: dict[str, list[ResumeChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.metadata.candidate_id].append(chunk)
    return dict(grouped)
