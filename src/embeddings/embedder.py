import os
import ssl
import time
import certifi
import urllib3
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.models.schemas import ResumeChunk

load_dotenv()

# Disable SSL verification when running behind corporate SSL inspection proxies
if os.getenv("PINECONE_SSL_VERIFY", "true").lower() == "false":
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SIM117
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hireflow-resumes")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
# gemini-embedding-001 outputs 3072-dimensional vectors
EMBEDDING_DIMENSION = 3072
# Stay within free tier: 100 RPM for embedding API
_EMBED_BATCH_SIZE = 10
_EMBED_DELAY_SECONDS = 1.0


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment.")
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{EMBEDDING_MODEL}",
        google_api_key=api_key,
    )


def _get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY is not set in the environment.")

    # ssl_verify=False only for dev environments with corporate SSL inspection
    ssl_verify = os.getenv("PINECONE_SSL_VERIFY", "true").lower() != "false"
    pc = Pinecone(
        api_key=api_key,
        ssl_ca_certs=certifi.where(),
        ssl_verify=ssl_verify,
    )
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)

    return pc.Index(PINECONE_INDEX_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a single string using Gemini embedding-001."""
    model = _get_embedding_model()
    return model.embed_query(text)


def embed_query(query_text: str) -> list[float]:
    """Embed a JD query string — alias kept separate for clarity."""
    return embed_text(query_text)


def index_resume_chunks(chunks: list[ResumeChunk], namespace: str = "resumes") -> int:
    """
    Embed and upsert all resume chunks into Pinecone.
    Returns the total number of vectors upserted.
    """
    model = _get_embedding_model()
    index = _get_pinecone_index()

    total_upserted = 0
    batch: list[tuple] = []

    for i, chunk in enumerate(chunks):
        vector = model.embed_query(chunk.text)
        metadata_dict = chunk.metadata.model_dump()
        batch.append((chunk.chunk_id, vector, metadata_dict))

        # Upsert in batches and throttle to respect free-tier rate limits
        if len(batch) >= _EMBED_BATCH_SIZE:
            index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            batch = []
            time.sleep(_EMBED_DELAY_SECONDS)

    if batch:
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    return total_upserted


def is_index_populated(namespace: str = "resumes") -> bool:
    """Return True if Pinecone index already has vectors (skip re-indexing)."""
    try:
        index = _get_pinecone_index()
        stats = index.describe_index_stats()
        ns_stats = stats.namespaces.get(namespace)
        return ns_stats is not None and ns_stats.vector_count > 0
    except Exception:
        return False


def delete_index_namespace(namespace: str = "resumes") -> None:
    """Wipe all vectors in a namespace — used for re-indexing."""
    index = _get_pinecone_index()
    index.delete(delete_all=True, namespace=namespace)
