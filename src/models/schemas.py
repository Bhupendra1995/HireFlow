from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Domain(str, Enum):
    ML_ENGINEER = "ML Engineer"
    DEVOPS_ENGINEER = "DevOps Engineer"
    FRONTEND_DEVELOPER = "Frontend Developer"
    BACKEND_DEVELOPER = "Backend Developer"
    PRODUCT_MANAGER = "Product Manager"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Resume ingestion
# ---------------------------------------------------------------------------

class ResumeMetadata(BaseModel):
    """Metadata stored alongside each vector in Pinecone."""
    candidate_id: str = Field(description="Unique identifier derived from filename")
    candidate_name: str = Field(default="Unknown")
    file_name: str
    domain: str = Field(default=Domain.UNKNOWN)
    total_years: int = Field(default=0, description="Estimated years of experience")
    chunk_index: int = Field(description="Position of this chunk within the resume")
    total_chunks: int = Field(description="Total number of chunks for this resume")


class ResumeChunk(BaseModel):
    """A single text chunk from a parsed resume."""
    chunk_id: str = Field(description="Unique ID: candidate_id + chunk_index")
    text: str
    metadata: ResumeMetadata


# ---------------------------------------------------------------------------
# LLM evaluation output (used by PydanticOutputParser)
# ---------------------------------------------------------------------------

class CandidateEvaluation(BaseModel):
    """Structured LLM output for a single candidate against a JD."""
    candidate_id: str
    candidate_name: str
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    strengths: list[str] = Field(
        description="Skills and experiences that match the JD requirements"
    )
    gaps: list[str] = Field(
        description="JD requirements not evidenced in the resume"
    )
    summary: str = Field(
        description="One-paragraph human-readable hiring recommendation"
    )
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    # Set to True by Self-RAG when context does not fully support the evaluation
    low_confidence_flag: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Self-RAG critique tokens
# ---------------------------------------------------------------------------

class SelfRAGCritique(BaseModel):
    """Per-chunk and per-generation critique produced by Self-RAG."""
    is_relevant: bool = Field(description="Is this chunk relevant to the query?")
    is_supported: bool = Field(
        description="Is the generated answer grounded in the retrieved context?"
    )
    usefulness_score: int = Field(
        ge=1,
        le=5,
        description="How useful is the answer on a 1-5 scale?",
    )
    attempt: int = Field(default=1, description="Which retry attempt produced this result")


class SelfRAGResult(BaseModel):
    """Final output of the Self-RAG loop for a candidate."""
    evaluation: CandidateEvaluation
    critique: SelfRAGCritique
    attempts_used: int = Field(default=1)


# ---------------------------------------------------------------------------
# Pipeline configuration (adaptive — changes based on RAGAS feedback)
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    top_k: int = Field(default=4, ge=1, le=20)
    use_hybrid_search: bool = True
    use_reranker: bool = True
    use_compression: bool = True
    use_mmr_dedup: bool = True
    use_query_expansion: bool = False
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_self_rag_retries: int = Field(default=2, ge=1, le=5)
    max_ragas_iterations: int = Field(default=2, ge=1, le=10)

    # RAGAS thresholds
    threshold_context_precision: float = Field(default=0.70)
    threshold_context_recall: float = Field(default=0.75)
    threshold_faithfulness: float = Field(default=0.80)
    threshold_answer_relevancy: float = Field(default=0.75)


# ---------------------------------------------------------------------------
# RAGAS evaluation scores
# ---------------------------------------------------------------------------

class RAGASScores(BaseModel):
    """Holds the 4 RAGAS metric scores for a pipeline evaluation run."""
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    iteration: int = Field(default=1)

    def all_pass(self, config: PipelineConfig) -> bool:
        return (
            (self.context_precision or 0) >= config.threshold_context_precision
            and (self.context_recall or 0) >= config.threshold_context_recall
            and (self.faithfulness or 0) >= config.threshold_faithfulness
            and (self.answer_relevancy or 0) >= config.threshold_answer_relevancy
        )

    def failing_metrics(self, config: PipelineConfig) -> list[str]:
        failing = []
        if (self.context_precision or 0) < config.threshold_context_precision:
            failing.append("context_precision")
        if (self.context_recall or 0) < config.threshold_context_recall:
            failing.append("context_recall")
        if (self.faithfulness or 0) < config.threshold_faithfulness:
            failing.append("faithfulness")
        if (self.answer_relevancy or 0) < config.threshold_answer_relevancy:
            failing.append("answer_relevancy")
        return failing


# ---------------------------------------------------------------------------
# Search request / response
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Input to the HireFlow search pipeline."""
    jd_text: str = Field(description="Full job description text")
    jd_title: str = Field(default="", description="Optional JD title for display")
    top_n_results: int = Field(default=10, ge=1, le=50)
    config: PipelineConfig = Field(default_factory=PipelineConfig)


class SearchResponse(BaseModel):
    """Final ranked output returned to the Streamlit UI."""
    jd_title: str
    results: list[SelfRAGResult]
    pipeline_config_used: PipelineConfig
    ragas_scores: Optional[RAGASScores] = None
    total_candidates_retrieved: int
