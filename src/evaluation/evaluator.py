import os
import sys
import types
import time
from dotenv import load_dotenv

# ragas 0.4.x imports from langchain_community.chat_models.vertexai which was
# removed in langchain-community 0.4.x — inject a shim before ragas loads
try:
    import langchain_community.chat_models.vertexai  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from langchain_google_vertexai import ChatVertexAI
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from src.models.schemas import (
    CandidateEvaluation,
    ConfidenceLevel,
    PipelineConfig,
    RAGASScores,
    ResumeChunk,
    SelfRAGCritique,
    SelfRAGResult,
)
from src.retrieval.retriever import group_chunks_by_candidate

load_dotenv()

LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
# Parallel evaluations capped to respect 15 RPM free-tier limit
_MAX_PARALLEL = 5
_LLM_CALL_DELAY = 4.5  # seconds between parallel batches


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert technical recruiter. Evaluate the candidate ONLY based on the "
        "resume context provided. Do NOT infer or assume skills not explicitly mentioned. "
        "If a requirement is unclear, state 'not evidenced'.\n\n"
        "Respond with valid JSON matching this schema:\n{format_instructions}"
    )),
    ("human", (
        "Job Description:\n{jd_text}\n\n"
        "Candidate Resume Context:\n{resume_context}\n\n"
        "Candidate ID: {candidate_id}\n"
        "Candidate Name: {candidate_name}\n\n"
        "Evaluate this candidate against the job description."
    )),
])

_SELF_RAG_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a quality control agent for a recruitment AI. "
        "Answer with valid JSON only. Schema:\n{format_instructions}"
    )),
    ("human", (
        "Job Description:\n{jd_text}\n\n"
        "Resume Context Used:\n{resume_context}\n\n"
        "Generated Evaluation Summary:\n{evaluation_summary}\n\n"
        "Answer these questions:\n"
        "1. is_relevant: Is the resume context relevant to this job description? (true/false)\n"
        "2. is_supported: Is the evaluation summary fully grounded in the resume context "
        "with no fabricated details? (true/false)\n"
        "3. usefulness_score: How useful is this evaluation for a recruiter? (1=poor, 5=excellent)\n"
        "4. attempt: Set to {attempt}"
    )),
])


def _get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        google_api_key=api_key,
    )


def _build_context(chunks: list[ResumeChunk]) -> str:
    return "\n\n---\n\n".join(chunk.text for chunk in chunks)


def _evaluate_candidate_once(
    candidate_id: str,
    candidate_name: str,
    chunks: list[ResumeChunk],
    jd_text: str,
    config: PipelineConfig,
    attempt: int,
) -> SelfRAGResult:
    """Single attempt: evaluate + Self-RAG critique."""
    llm = _get_llm(temperature=config.temperature)
    eval_parser = PydanticOutputParser(pydantic_object=CandidateEvaluation)
    critique_parser = PydanticOutputParser(pydantic_object=SelfRAGCritique)

    resume_context = _build_context(chunks)

    # --- Step A: Generate evaluation ---
    eval_chain = _EVALUATION_PROMPT | llm | eval_parser
    evaluation: CandidateEvaluation = eval_chain.invoke({
        "jd_text": jd_text,
        "resume_context": resume_context,
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "format_instructions": eval_parser.get_format_instructions(),
    })

    # --- Step B: Self-RAG critique ---
    critique_chain = _SELF_RAG_CRITIQUE_PROMPT | llm | critique_parser
    critique: SelfRAGCritique = critique_chain.invoke({
        "jd_text": jd_text,
        "resume_context": resume_context,
        "evaluation_summary": evaluation.summary,
        "attempt": attempt,
        "format_instructions": critique_parser.get_format_instructions(),
    })
    critique.attempt = attempt

    return SelfRAGResult(evaluation=evaluation, critique=critique, attempts_used=attempt)


def evaluate_candidate_with_self_rag(
    candidate_id: str,
    candidate_name: str,
    chunks: list[ResumeChunk],
    jd_text: str,
    config: PipelineConfig,
) -> SelfRAGResult:
    """
    Self-RAG loop: evaluate → critique → retry if not supported.
    Exits early on success or after max_self_rag_retries attempts.
    """
    best_result: SelfRAGResult | None = None

    for attempt in range(1, config.max_self_rag_retries + 1):
        result = _evaluate_candidate_once(
            candidate_id, candidate_name, chunks, jd_text, config, attempt
        )

        # Accept if critique passes both relevance and support checks
        if result.critique.is_relevant and result.critique.is_supported:
            return result

        # Track best result in case all attempts fail
        if best_result is None or result.critique.usefulness_score > best_result.critique.usefulness_score:
            best_result = result

        # Harden prompt on faithfulness failure — lower temperature for next attempt
        if not result.critique.is_supported:
            config = config.model_copy(update={"temperature": max(0.1, config.temperature - 0.1)})

        if attempt < config.max_self_rag_retries:
            time.sleep(_LLM_CALL_DELAY)

    # All retries exhausted — return best attempt with low confidence flag
    best_result.evaluation.low_confidence_flag = True
    best_result.evaluation.confidence_level = ConfidenceLevel.LOW
    return best_result


def evaluate_all_candidates(
    grouped_chunks: dict[str, list[ResumeChunk]],
    jd_text: str,
    config: PipelineConfig,
) -> list[SelfRAGResult]:
    """
    Evaluate all retrieved candidates in batches to respect free-tier rate limits.
    Results are sorted by evaluation score descending.
    """
    results: list[SelfRAGResult] = []
    items = list(grouped_chunks.items())

    for i in range(0, len(items), _MAX_PARALLEL):
        batch = items[i: i + _MAX_PARALLEL]
        for candidate_id, chunks in batch:
            candidate_name = chunks[0].metadata.candidate_name if chunks else "Unknown"
            result = evaluate_candidate_with_self_rag(
                candidate_id, candidate_name, chunks, jd_text, config
            )
            results.append(result)
            time.sleep(_LLM_CALL_DELAY)

    return sorted(results, key=lambda r: r.evaluation.score, reverse=True)


# ---------------------------------------------------------------------------
# RAGAS evaluation + adaptive config
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    results: list[SelfRAGResult],
    grouped_chunks: dict[str, list[ResumeChunk]],
    jd_text: str,
    config: PipelineConfig,
    iteration: int = 1,
) -> RAGASScores:
    """Run the 4 RAGAS metrics on a batch of candidate evaluation results."""
    api_key = os.getenv("GEMINI_API_KEY")
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, google_api_key=api_key)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=f"models/{os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')}", google_api_key=api_key)
    )

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for result in results:
        cid = result.evaluation.candidate_id
        chunks = grouped_chunks.get(cid, [])
        data["question"].append(jd_text)
        data["answer"].append(result.evaluation.summary)
        data["contexts"].append([c.text for c in chunks])
        data["ground_truth"].append(result.evaluation.summary)

    dataset = Dataset.from_dict(data)
    ragas_result = evaluate(
        dataset,
        metrics=[
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            ),
        ],
        raise_exceptions=False,
    )

    df = ragas_result.to_pandas()

    def _mean(col: str) -> float | None:
        return float(df[col].mean()) if col in df.columns else None

    return RAGASScores(
        context_precision=_mean("context_precision"),
        context_recall=_mean("context_recall"),
        faithfulness=_mean("faithfulness"),
        answer_relevancy=_mean("answer_relevancy"),
        iteration=iteration,
    )


def adapt_config(scores: RAGASScores, config: PipelineConfig) -> PipelineConfig:
    """
    Return an updated PipelineConfig based on which RAGAS metrics are failing.
    Each metric maps to specific remediation strategies.
    """
    updates: dict = {}
    failing = scores.failing_metrics(config)

    if "context_precision" in failing:
        # Noise too high → reduce top_k and force reranking
        updates["top_k"] = max(4, config.top_k - 2)
        updates["use_reranker"] = True
        updates["use_compression"] = True

    if "context_recall" in failing:
        # Missing relevant content → cast wider net + enable hybrid search
        updates["top_k"] = min(20, config.top_k + 2)
        updates["use_hybrid_search"] = True
        updates["use_query_expansion"] = True

    if "faithfulness" in failing:
        # Hallucination detected → lower temperature + harden context
        updates["temperature"] = max(0.1, config.temperature - 0.1)
        updates["use_compression"] = True

    if "answer_relevancy" in failing:
        # Off-topic evaluation → enforce deduplication for cleaner context
        updates["use_mmr_dedup"] = True

    return config.model_copy(update=updates)


def run_adaptive_ragas_loop(
    results: list[SelfRAGResult],
    grouped_chunks: dict[str, list[ResumeChunk]],
    jd_text: str,
    config: PipelineConfig,
) -> tuple[RAGASScores, PipelineConfig]:
    """
    Run RAGAS → adapt config → repeat until all metrics pass or max iterations reached.
    Returns the final scores and the best-performing config.
    """
    current_config = config

    for iteration in range(1, config.max_ragas_iterations + 1):
        scores = run_ragas_evaluation(results, grouped_chunks, jd_text, current_config, iteration)

        if scores.all_pass(current_config):
            return scores, current_config

        # Check if config actually changed — avoid pointless iterations
        new_config = adapt_config(scores, current_config)
        if new_config == current_config:
            break

        current_config = new_config

    return scores, current_config
