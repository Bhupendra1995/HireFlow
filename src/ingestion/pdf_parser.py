import os
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from src.models.schemas import Domain, ResumeChunk, ResumeMetadata

# Chunk size tuned for resume sections; overlap preserves cross-boundary context
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    Domain.ML_ENGINEER: [
        "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
        "llm", "rag", "langchain", "llamaindex", "hugging face", "mlflow",
        "data scientist", "ai engineer", "neural",
    ],
    Domain.DEVOPS_ENGINEER: [
        "devops", "kubernetes", "docker", "terraform", "ci/cd", "aws", "gcp",
        "azure", "jenkins", "ansible", "site reliability", "platform engineer",
        "helm", "argocd", "prometheus",
    ],
    Domain.FRONTEND_DEVELOPER: [
        "react", "typescript", "frontend", "next.js", "vue", "css", "html",
        "redux", "tailwind", "webpack", "ui engineer", "storybook", "figma",
    ],
    Domain.BACKEND_DEVELOPER: [
        "fastapi", "django", "flask", "postgresql", "microservices", "redis",
        "celery", "sqlalchemy", "backend", "rest api", "graphql", "kafka",
        "rabbitmq", "python developer",
    ],
    Domain.PRODUCT_MANAGER: [
        "product manager", "roadmap", "okr", "agile", "scrum", "jira",
        "stakeholder", "a/b testing", "user research", "go-to-market",
        "product discovery", "sprint",
    ],
}


def _infer_domain(text: str) -> str:
    """Infer resume domain by counting keyword hits per domain."""
    text_lower = text.lower()
    scores = {
        domain: sum(1 for kw in keywords if kw in text_lower)
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else Domain.UNKNOWN


def _extract_years(text: str) -> int:
    """Extract the highest years-of-experience mention found in the text."""
    pattern = r'(\d+)\s*\+?\s*years?'
    matches = [int(m) for m in re.findall(pattern, text, re.IGNORECASE)]
    return max(matches) if matches else 0


def _extract_name_from_text(text: str) -> str:
    """Return the first non-empty line as a best-effort candidate name."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not re.search(r'[@|:]', stripped):
            return stripped[:80]
    return "Unknown"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def parse_resume_to_chunks(pdf_path: str) -> list[ResumeChunk]:
    """Parse a single resume PDF into a list of overlapping text chunks."""
    path = Path(pdf_path)
    full_text = extract_text_from_pdf(pdf_path)

    candidate_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path.name))
    candidate_name = _extract_name_from_text(full_text)
    domain = _infer_domain(full_text)
    total_years = _extract_years(full_text)

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    raw_chunks = splitter.split_text(full_text)

    chunks: list[ResumeChunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        metadata = ResumeMetadata(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            file_name=path.name,
            domain=domain,
            total_years=total_years,
            chunk_index=i,
            total_chunks=len(raw_chunks),
        )
        chunks.append(ResumeChunk(
            chunk_id=f"{candidate_id}_{i}",
            text=chunk_text.strip(),
            metadata=metadata,
        ))
    return chunks


def parse_all_resumes(resumes_dir: str) -> list[ResumeChunk]:
    """Parse every PDF in the given directory and return all chunks."""
    pdf_files = list(Path(resumes_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {resumes_dir}")

    all_chunks: list[ResumeChunk] = []
    for pdf_path in pdf_files:
        chunks = parse_resume_to_chunks(str(pdf_path))
        all_chunks.extend(chunks)

    return all_chunks


def load_jd_text(jd_path: str) -> str:
    """Read a job description .txt file and return its content."""
    with open(jd_path, encoding="utf-8") as f:
        return f.read().strip()


def load_all_jds(jds_dir: str) -> dict[str, str]:
    """Return a mapping of {jd_filename_stem: jd_text} for all .txt files."""
    jd_files = list(Path(jds_dir).glob("*.txt"))
    if not jd_files:
        raise FileNotFoundError(f"No .txt JD files found in {jds_dir}")
    return {p.stem: load_jd_text(str(p)) for p in jd_files}
