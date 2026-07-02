"""
phase3/00_config.py — Central config for Phase 3 cascade pipeline.
Import this in every other phase3 script.
"""
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "data"
PHASE3_DIR    = BASE_DIR

# ── Input files ───────────────────────────────────────────────────────────────
CANDIDATES_FILE      = DATA_DIR / "candidates_full.jsonl"
JD_FILE              = DATA_DIR / "job_description.docx"
FAISS_INDEX_FILE     = ARTIFACTS_DIR / "faiss_index.bin"
BM25_INDEX_FILE      = ARTIFACTS_DIR / "bm25_index.pkl"
FEATURES_FILE        = ARTIFACTS_DIR / "features.parquet"
CANDIDATE_IDS_FILE   = ARTIFACTS_DIR / "candidate_ids.json"
JD_EMBEDDING_FILE    = ARTIFACTS_DIR / "jd_query_embedding.npy"

# ── Phase 3 intermediate outputs ──────────────────────────────────────────────
RRF_POOL_FILE  = DATA_DIR / "rrf_pool.json"       # Top-N post-RRF candidate pool
CE_SCORES_FILE = DATA_DIR / "ce_scores.json"      # Cross-encoder scores per candidate

# ── Final output ──────────────────────────────────────────────────────────────
SUBMISSION_FILE = BASE_DIR / "submission.csv"

# ── Retrieval hyperparameters ─────────────────────────────────────────────────
TOP_L1_DENSE  = 1000   # Candidates recalled from FAISS
TOP_L1_SPARSE = 1000   # Candidates recalled from BM25
TOP_RRF       = 500    # Post-fusion pool fed to cross-encoder
TOP_FINAL     = 100    # Required by submission spec
RRF_K         = 60     # Standard RRF constant

# ── Cross-Encoder ─────────────────────────────────────────────────────────────
CE_MODEL      = str(BASE_DIR / "models" / "gte-reranker-modernbert-base")
CE_MAX_TOKENS = 2048    # Hard limit of the model

# ── Final blend weights (must sum to 1.0) ─────────────────────────────────────
ALPHA = 0.55    # Cross-encoder score weight
BETA  = 0.45    # Rubric composite score weight

# ── JD query text (used for BM25 and cross-encoder) ──────────────────────────
# This is also the first section extracted from job_description.docx.
# We trim to the most discriminating part to stay within 512 tokens.
JD_QUERY_TEXT = (
    "search_query: Senior AI Engineer with expertise in Python, production ML systems, "
    "dense retrieval, sentence transformers, vector databases (FAISS, Pinecone, Qdrant, Weaviate), "
    "learning to rank, NDCG, MRR, RAG pipelines, LLM fine-tuning (LoRA, QLoRA), "
    "embedding models (BGE, E5, nomic-embed), reranking, hybrid search BM25+dense, "
    "deployed ranking or recommendation systems at product companies."
)
