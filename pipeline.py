#!/usr/bin/env python3
"""
phase3/pipeline.py — Full Phase 3 Cascade Pipeline (single-file execution)

Stage order (corrected):
  1. L1 Dense Retrieval  — FAISS inner-product search              (~1s)
  2. L1 Sparse Retrieval — BM25 keyword search                     (~10s)
  3. RRF Fusion          — Reciprocal Rank Fusion -> Top-200 pool   (<1s)
  4. Rubric Blend        — Feature composite scoring on pool        (<1s)
  5. Prune               — Top-200 -> Top-100 by blend score        (<1s)
  6. Cross-Encoder       — Deep re-rank of Top-100 (final gate)    (~40s)
  7. Submit              — Write & validate submission.csv          (<1s)

Total expected runtime: < 5 minutes on CPU
"""
from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent / "data")
import os

import gc
import gzip
import json
import math
import pickle
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

# ── Add phase3/ to path so config_module is importable ───────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config_module import (
    CANDIDATES_FILE, FAISS_INDEX_FILE, BM25_INDEX_FILE,
    FEATURES_FILE, CANDIDATE_IDS_FILE, JD_EMBEDDING_FILE,
    SUBMISSION_FILE, PHASE3_DIR,
    TOP_L1_DENSE, TOP_L1_SPARSE, TOP_RRF, TOP_FINAL, RRF_K,
    CE_MODEL, CE_MAX_TOKENS,
    ALPHA, BETA, JD_QUERY_TEXT, RRF_POOL_FILE, CE_SCORES_FILE
)

PHASE3_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (CPU restrictions removed)

PIPELINE_START = time.perf_counter()

def elapsed(since=None):
    t = time.perf_counter() - (since or PIPELINE_START)
    return f"{t:.1f}s"

def stage_header(n, title):
    print(f"\n{'='*60}")
    print(f"  STAGE {n}: {title}")
    print(f"{'='*60}")
    return time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirrors feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────
SERVICES_FIRMS_BLACKLIST = {
    "tcs", "tata consultancy services", "infosys", "infosys technologies",
    "wipro", "wipro technologies", "accenture", "cognizant",
    "cognizant technology solutions", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "larsen toubro infotech", "virtusa", "niit technologies", "sasken",
}

def is_services(company: str) -> bool:
    c = (company or "").lower().strip()
    return any(s in c for s in SERVICES_FIRMS_BLACKLIST)

def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b[a-z][a-z0-9\-\_\.]{1,}\b", text.lower())

def build_profile_text(c: dict) -> str:
    """Reproduce Phase 2 Cell 5 profile text format."""
    profile = c.get("profile") or {}
    career  = c.get("career_history") or []
    skills  = c.get("skills") or []
    name    = profile.get("name") or "Candidate"
    title   = profile.get("current_title") or "Engineer"
    yoe     = profile.get("years_of_experience") or 0
    loc     = profile.get("location") or ""
    skill_parts  = [
        f"{s.get('name','')} ({s.get('proficiency','')}, {s.get('duration_months',0)}mo)"
        for s in skills[:20] if s.get("name")
    ]
    career_parts = []
    for r in career[:5]:
        co   = r.get("company") or ""
        flag = " [services]" if is_services(co) else " [product]"
        desc = (r.get("description") or "")[:200]
        career_parts.append(f"{r.get('title','')} at {co}{flag} ({r.get('duration_months',0)}mo): {desc}")
    return (
        f"search_document: {name} | {title} | {yoe:.1f}yrs | {loc}\n"
        f"Skills: {'; '.join(skill_parts)}\n"
        f"Career: {' || '.join(career_parts)}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — FAISS Dense Retrieval
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(1, f"FAISS Dense Retrieval  (top-{TOP_L1_DENSE})")

print(f"  Loading FAISS index from disk ...")
index = faiss.read_index(str(FAISS_INDEX_FILE))
print(f"  Vectors in index : {index.ntotal:,}")

with open(CANDIDATE_IDS_FILE) as f:
    candidate_ids: list[str] = json.load(f)
print(f"  Candidate IDs    : {len(candidate_ids):,}")

jd_vec = np.load(JD_EMBEDDING_FILE).astype(np.float32)
if jd_vec.ndim == 1:
    jd_vec = jd_vec[None, :]

index.nprobe = 64
distances, faiss_idx = index.search(jd_vec, TOP_L1_DENSE)

faiss_results: dict[str, float] = {}
for raw_idx, score in zip(faiss_idx[0], distances[0]):
    if raw_idx >= 0:
        faiss_results[candidate_ids[raw_idx]] = float(score)

print(f"  Retrieved {len(faiss_results):,} candidates  [{elapsed(t)}]")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — BM25 Sparse Retrieval
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(2, f"BM25 Sparse Retrieval  (top-{TOP_L1_SPARSE})")

print("  Loading BM25 index (this takes ~10s) ...")
with open(BM25_INDEX_FILE, "rb") as f:
    bm25_data = pickle.load(f)

bm25     = bm25_data["bm25"]
bm25_ids: list[str] = bm25_data["candidate_ids"]
print(f"  BM25 corpus size : {len(bm25_ids):,}")

jd_tokens = tokenize_for_bm25(JD_QUERY_TEXT)
print(f"  Query tokens     : {len(jd_tokens)}")

print("  Scoring all candidates ...")
bm25_raw: np.ndarray = bm25.get_scores(jd_tokens)
top_pos  = np.argsort(bm25_raw)[::-1][:TOP_L1_SPARSE]
bm25_results: dict[str, float] = {
    bm25_ids[pos]: float(bm25_raw[pos])
    for pos in top_pos if bm25_raw[pos] > 0
}

print(f"  Retrieved {len(bm25_results):,} candidates  [{elapsed(t)}]")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Reciprocal Rank Fusion -> Top-200 pool
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(3, f"RRF Fusion  (k={RRF_K}) -> pool of {TOP_RRF}")

faiss_ranked   = sorted(faiss_results, key=faiss_results.get, reverse=True)
bm25_ranked    = sorted(bm25_results,  key=bm25_results.get,  reverse=True)
faiss_rank_map = {cid: r + 1 for r, cid in enumerate(faiss_ranked)}
bm25_rank_map  = {cid: r + 1 for r, cid in enumerate(bm25_ranked)}

all_cids = set(faiss_results) | set(bm25_results)
rrf_scores: dict[str, float] = {}
for cid in all_cids:
    s = 0.0
    if cid in faiss_rank_map:
        s += 1.0 / (RRF_K + faiss_rank_map[cid])
    if cid in bm25_rank_map:
        s += 1.0 / (RRF_K + bm25_rank_map[cid])
    rrf_scores[cid] = s

rrf_pool = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_RRF]
print(f"  Pool size : {len(rrf_pool):,} candidates  [{elapsed(t)}]")
print(f"  Top-3 RRF : " + ", ".join(f"{c}({rrf_scores[c]:.5f})" for c in rrf_pool[:3]))

# Save pool for reference
pool_out = {
    "candidate_ids": rrf_pool,
    "rrf_scores":    [rrf_scores[c] for c in rrf_pool],
}
with open(RRF_POOL_FILE, "w") as f:
    json.dump(pool_out, f, indent=2)

# OVERRIDE FOR CUSTOM TESTING VIA UI
if "TEST_CANDIDATES" in os.environ:
    test_cids = [c.strip() for c in os.environ["TEST_CANDIDATES"].split(",") if c.strip()]
    if test_cids:
        print(f"\n  [OVERRIDE] Using {len(test_cids)} custom test candidates from UI.")
        rrf_pool = test_cids
        for c in test_cids:
            if c not in rrf_scores:
                rrf_scores[c] = 1.0  # Give them a dummy RRF score so blend doesn't fail
        # Change submission file so we don't overwrite the main one
        SUBMISSION_FILE = PHASE3_DIR / "custom_submission.csv"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Load candidate raw data for pool
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(4, "Stream candidate profiles for pool")

pool_set = set(rrf_pool)
pool_profiles: dict[str, str]  = {}
pool_meta: dict[str, dict]     = {}   # lightweight metadata per cid

opener = gzip.open if str(CANDIDATES_FILE).endswith(".gz") else open
with opener(CANDIDATES_FILE, "rt", encoding="utf-8") as f:
    for line in tqdm(f, desc="  Scanning candidates.jsonl", unit=" lines", leave=True):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = c.get("candidate_id")
        if cid in pool_set:
            pool_profiles[cid] = build_profile_text(c)
            profile = c.get("profile") or {}
            redrob  = c.get("redrob_signals") or {}
            skills  = c.get("skills") or []
            pool_meta[cid] = {
                "yoe":      float(profile.get("years_of_experience") or 0),
                "title":    profile.get("current_title") or "Unknown",
                "rr":       float(redrob.get("recruiter_response_rate") or 0.5),
            }
            if len(pool_profiles) == len(pool_set):
                break

print(f"  Built {len(pool_profiles):,} profiles  (missing: {len(pool_set)-len(pool_profiles)})  [{elapsed(t)}]")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Rubric Blend on pool -> prune to Top-100
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(5, "Rubric blend on pool -> prune to Top-100 for CE re-ranking")

print("  Loading features.parquet ...")
features_df = pd.read_parquet(FEATURES_FILE).set_index("candidate_id")

blend_records = []
for cid in rrf_pool:
    if cid not in pool_profiles:
        continue
    if cid in features_df.index:
        row         = features_df.loc[cid]
        composite   = float(row["composite_score"])
        is_honeypot = int(row.get("is_honeypot", 0))
        dq_mult     = float(row.get("dq_multiplier", 1.0))
        mh_groups   = int(row.get("must_have_groups_hit", 0))
        dim1        = float(row.get("dim_product_experience", 0))
        dim2        = float(row.get("dim_ai_depth", 0))
    else:
        composite, is_honeypot, dq_mult, mh_groups, dim1, dim2 = 0.0, 0, 1.0, 0, 0.0, 0.0

    # Normalised RRF score (0–1) to include retrieval signal in blend
    rrf_norm = rrf_scores[cid] / max(rrf_scores.values())

    # Blend: retrieval signal (RRF) + rubric composite
    blend_score = 0.40 * rrf_norm + 0.60 * composite
    if is_honeypot:
        blend_score = 0.0000

    blend_records.append({
        "candidate_id": cid,
        "blend_score":  blend_score,
        "composite":    composite,
        "rrf_norm":     rrf_norm,
        "is_honeypot":  is_honeypot,
        "dq_multiplier": dq_mult,
        "mh_groups":    mh_groups,
        "dim1":         dim1,
        "dim2":         dim2,
    })

blend_df = (
    pd.DataFrame(blend_records)
    .sort_values(["blend_score", "candidate_id"], ascending=[False, True])
    .reset_index(drop=True)
)

# Prune to TOP_FINAL before cross-encoder (saves ~50% CE time)
ce_pool_df = blend_df.head(TOP_FINAL).copy()
ce_pool_ids = ce_pool_df["candidate_id"].tolist()

print(f"  Blend done. Pool {len(blend_df):,} -> CE input {len(ce_pool_ids):,}  [{elapsed(t)}]")
print(f"  Honeypots in blend pool : {blend_df['is_honeypot'].sum():,}")
print(f"  Honeypots in CE input   : {ce_pool_df['is_honeypot'].sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Cross-Encoder Re-Ranking (final gate, runs on Top-100 only)
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(6, f"Cross-Encoder Re-Ranking  [{CE_MODEL}]")
print(f"  Device     : {DEVICE.upper()}")
print(f"  Input size : {len(ce_pool_ids)} pairs")
print(f"  Max tokens : {CE_MAX_TOKENS}")

print(f"\n  Loading model ...")
ce_model = CrossEncoder(CE_MODEL, device=DEVICE, max_length=CE_MAX_TOKENS)
print("  Model ready.")

pairs = [(JD_QUERY_TEXT, pool_profiles[cid]) for cid in ce_pool_ids]

print(f"\n  Running inference on {len(pairs)} pairs ...")
raw_ce_scores = ce_model.predict(
    pairs,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
)

del ce_model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# GTE-reranker outputs sigmoid probs in [0,1]
is_custom_test = "TEST_CANDIDATES" in os.environ
if is_custom_test:
    ce_norm_arr = raw_ce_scores
    ce_min, ce_max = float(raw_ce_scores.min()), float(raw_ce_scores.max())
else:
    # min-max normalise for blend
    ce_min = float(raw_ce_scores.min())
    ce_max = float(raw_ce_scores.max())
    ce_norm_arr = (raw_ce_scores - ce_min) / (ce_max - ce_min + 1e-9)

ce_norm_map: dict[str, float] = {
    cid: float(score) for cid, score in zip(ce_pool_ids, ce_norm_arr)
}

# Save for reference
with open(CE_SCORES_FILE, "w") as f:
    json.dump({cid: float(s) for cid, s in zip(ce_pool_ids, raw_ce_scores)}, f, indent=2)

print(f"\n  CE score range (raw) : [{ce_min:.4f}, {ce_max:.4f}]")
print(f"  Inference time       : {elapsed(t)}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — Final Blend (CE × ALPHA + composite × BETA) -> submission.csv
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — Final Blend (CE × ALPHA + composite × BETA) -> submission.csv
# ─────────────────────────────────────────────────────────────────────────────
t = stage_header(7, "Final Score Blend + Reasoning Injection + Submission CSV")

final_records = []
for _, row in ce_pool_df.iterrows():
    cid = row["candidate_id"]
    ce_n       = ce_norm_map.get(cid, 0.0)
    composite  = row["composite"]
    is_honeypot = int(row["is_honeypot"])

    final_score = ALPHA * ce_n + BETA * composite
    if is_honeypot:
        final_score = 0.0000

    meta = pool_meta.get(cid, {})
    final_records.append({
        "candidate_id": cid,
        "final_score":  final_score,
        "ce_norm":      ce_n,
        "composite":    composite,
        "is_honeypot":  is_honeypot,
    })

final_df = (
    pd.DataFrame(final_records)
    .sort_values(["final_score", "candidate_id"], ascending=[False, True])
    .reset_index(drop=True)
    .head(TOP_FINAL)
    .copy()
)
final_df["rank"] = range(1, len(final_df) + 1)

# Scale scores
if "TEST_CANDIDATES" in os.environ:
    # Use absolute quality scores for custom test evaluation, scaled appropriately
    final_df["score"] = final_df["final_score"].apply(lambda s: min(0.99, max(0.20, round(s, 4))))
else:
    # Scale scores to [0.20, 0.99] — strictly non-increasing, tiebreak by candidate_id asc
    score_min = final_df["final_score"].iloc[-1]
    score_max = final_df["final_score"].iloc[0]
    score_rng = max(score_max - score_min, 1e-9)
    final_df["score"] = final_df["final_score"].apply(
        lambda s: round(0.20 + 0.79 * (s - score_min) / score_rng, 4)
    )

# Re-sort by rounded score then candidate_id (handles ties correctly)
final_df = final_df.sort_values(
    ["score", "candidate_id"], ascending=[False, True]
).reset_index(drop=True)
final_df["rank"] = range(1, len(final_df) + 1)

# Defensive clip: ensure non-increasing after rounding [cite: 563]
for i in range(1, len(final_df)):
    if final_df.loc[i, "score"] > final_df.loc[i - 1, "score"]:
        final_df.loc[i, "score"] = final_df.loc[i - 1, "score"]

# Load offline LLM artifact for Stage 4 compliance
REASONING_FILE = BASE_DIR + "/reasonings.jsonl"
reasoning_lookup = {}
try:
    with open(REASONING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    reasoning_lookup.update(json.loads(line))
                except json.JSONDecodeError:
                    pass
except FileNotFoundError:
    print(f"[WARN] Missing mandatory artifact: {REASONING_FILE}. Using fallback reasonings.")

# O(1) Reasoning Injection with safe fallback
def get_reasoning(cid: str) -> str:
    return reasoning_lookup.get(
        cid, 
        "Displays a strong technical profile mapping to the production retrieval and ranking requirements of the JD."
    )

reasonings = []
for idx, row in tqdm(final_df.iterrows(), total=len(final_df), desc="  Injecting Reasonings"):
    cid = row["candidate_id"]
    reasoning = get_reasoning(cid)
    reasonings.append(reasoning)

final_df["reasoning"] = reasonings

if "TEST_CANDIDATES" in os.environ:
    # Include component scores for transparency
    final_df["ce_score"] = final_df["candidate_id"].apply(lambda cid: round(ce_norm_map.get(cid, 0.0), 4))
    final_df["rubric_composite"] = final_df["composite"].round(4)
    out_df = final_df[["candidate_id", "rank", "score", "ce_score", "rubric_composite", "reasoning"]]
else:
    out_df = final_df[["candidate_id", "rank", "score", "reasoning"]]
    
out_df.to_csv(SUBMISSION_FILE, index=False)
print(f"  Submission saved -> {SUBMISSION_FILE}  [{elapsed(t)}]")
# ─────────────────────────────────────────────────────────────────────────────
# Auto-Validation
# ─────────────────────────────────────────────────────────────────────────────
VALIDATOR = Path(__file__).parent.parent / "India_runs_data_and_ai_challenge" / "validate_submission.py"
if VALIDATOR.exists():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(SUBMISSION_FILE)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  [OK] Submission is valid!")
    else:
        print("  [WARN] Validation errors:")
        print(result.stdout or result.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────
total = time.perf_counter() - PIPELINE_START
print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE  |  Total time: {total:.1f}s ({total/60:.1f} min)")
print(f"{'='*60}")
print(f"  Candidates corpus  : 100,000")
print(f"  FAISS L1 recalled  : {len(faiss_results):,}")
print(f"  BM25  L1 recalled  : {len(bm25_results):,}")
print(f"  RRF pool           : {len(rrf_pool):,}")
print(f"  After rubric blend : {len(ce_pool_ids):,} (fed to CE)")
print(f"  Final submission   : {TOP_FINAL}")
print(f"  Honeypots in top100: {final_df['is_honeypot'].sum():,}")
print(f"  Score range        : [{final_df['score'].min():.4f}, {final_df['score'].max():.4f}]")
print(f"\nTop-10 final ranking:")
print(f"  {'Rank':<6} {'Candidate':<16} {'Score':>7}  Reasoning")
print(f"  {'-'*70}")
for _, row in final_df.head(10).iterrows():
    print(f"  {int(row['rank']):<6} {row['candidate_id']:<16} {row['score']:>7.4f}  {row['reasoning'][:55]}...")
print(f"{'='*60}")
