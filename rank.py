#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking
Job: Senior AI Engineer — Founding Team at Redrob AI

Produces submission.csv (top 100 candidates ranked for the role).
Run: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration: AI/ML core skills the JD requires or prefers
# ---------------------------------------------------------------------------

# "Absolutely required" skills — high weight
REQUIRED_SKILLS = {
    # Embeddings / retrieval
    "sentence transformers", "sentence-transformers", "openai embeddings",
    "bge", "e5", "embeddings", "embedding", "text embeddings",
    # Vector DBs / hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "chromadb", "pgvector", "annoy",
    # Ranking / evaluation
    "ranking", "information retrieval", "bm25", "learning to rank",
    "learning-to-rank", "ndcg", "mrr", "map@k", "reranking", "re-ranking",
    "hybrid search", "hybrid retrieval",
    # Core ML / NLP
    "nlp", "natural language processing", "transformers", "bert",
    "llm", "large language models", "rag", "retrieval augmented generation",
    "semantic search", "neural search",
    # Python (required)
    "python",
    # Evaluation frameworks
    "a/b testing", "ab testing", "offline evaluation", "online evaluation",
}

# "Nice to have" skills — medium weight
PREFERRED_SKILLS = {
    "fine-tuning llms", "fine tuning", "lora", "qlora", "peft",
    "xgboost", "lightgbm", "gradient boosting",
    "pytorch", "tensorflow", "jax",
    "mlops", "ml ops", "kubeflow", "mlflow", "weights & biases",
    "docker", "kubernetes", "distributed systems",
    "spark", "kafka", "airflow",
    "github", "open source", "apache beam",
    "recommendation systems", "recommender systems",
    "feature engineering", "feature store",
    "data science", "machine learning", "deep learning",
    "neural networks", "computer vision", "speech recognition",
    "image classification", "object detection",
    "statistical modeling", "statistics",
    "bentoml", "triton", "onnx",
    "aws", "gcp", "azure", "cloud",
    "sql", "postgresql", "redis",
    "golang", "go", "rust", "scala",
}

# Titles that signal a strong AI/ML profile
STRONG_AI_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ml engineer", "senior ai engineer",
    "applied scientist", "applied ml scientist",
    "research scientist", "research engineer",
    "nlp engineer", "nlp scientist",
    "data scientist", "senior data scientist",
    "junior ml engineer", "junior ai engineer",
    "computer vision engineer", "deep learning engineer",
    "llm engineer", "rag engineer",
}

MEDIUM_AI_TITLES = {
    "software engineer", "backend engineer", "full stack engineer",
    "data engineer", "analytics engineer", "platform engineer",
    "senior software engineer", "lead engineer", "staff engineer",
    "principal engineer", "tech lead",
    "ai researcher", "ml researcher",
}

# Pure services companies — JD says "only services" is a disqualifier
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "niit technologies", "l&t infotech", "mindtree",
    "ltimindtree", "happiest minds",
}

# ---------------------------------------------------------------------------
# Honeypot detection helpers
# ---------------------------------------------------------------------------

def detect_honeypot(c: dict) -> bool:
    """Return True if the candidate profile has impossible signals."""
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    profile = c.get("profile", {})

    # Check 1: "expert" skill with 0 duration_months — impossible
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero_duration >= 3:
        return True

    # Check 2: More than 8 "expert" skills (unrealistic)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 8:
        return True

    # Check 3: Career duration mismatch — claimed experience far exceeds history
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    claimed_years = profile.get("years_of_experience", 0)
    if claimed_years > 3 and total_career_months < claimed_years * 6:
        # Claimed 8 years but only 2 years of career history → honeypot
        if claimed_years > 5 and total_career_months < 24:
            return True

    # Check 4: Company founded after experience start would exceed duration
    # (simple heuristic — company timeline impossible)
    for role in career:
        start_date = role.get("start_date", "")
        duration = role.get("duration_months", 0)
        try:
            start_year = int(start_date[:4])
            if start_year > 2024 and duration > 24:
                return True
        except (ValueError, IndexError):
            pass

    return False


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def normalize(val: float, min_val: float, max_val: float) -> float:
    if max_val == max_val and max_val > min_val:
        return max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))
    return 0.0


def score_skills(c: dict) -> tuple[float, list[str]]:
    """Score based on AI/ML skill match. Returns (score 0-1, matched skill names)."""
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})

    required_matches = []
    preferred_matches = []
    total_weight = 0.0

    proficiency_mult = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}

    for skill in skills:
        name_lower = skill["name"].lower()
        prof = proficiency_mult.get(skill.get("proficiency", "beginner"), 0.4)
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        # Trust multiplier: longer duration + endorsements = more credible
        duration_mult = min(1.0, duration / 24) if duration > 0 else 0.3
        endorse_mult = min(1.0, 0.5 + endorsements / 40)

        # Assessment score boost if taken on Redrob platform
        assess_bonus = 0.0
        for assess_key, assess_score in assessment_scores.items():
            if assess_key.lower() in name_lower or name_lower in assess_key.lower():
                assess_bonus = (assess_score / 100) * 0.2
                break

        trust = prof * duration_mult * endorse_mult + assess_bonus

        # Check required skills
        is_required = any(req in name_lower for req in REQUIRED_SKILLS)
        is_preferred = any(pref in name_lower for pref in PREFERRED_SKILLS)

        if is_required:
            required_matches.append(skill["name"])
            total_weight += 2.0 * trust
        elif is_preferred:
            preferred_matches.append(skill["name"])
            total_weight += 1.0 * trust

    # Normalize: max achievable with 10 required + 5 preferred = 20 + 5 = 25
    raw_score = min(1.0, total_weight / 12.0)

    # Bonus for having actual assessment scores (proof of skill)
    if assessment_scores:
        avg_assess = sum(assessment_scores.values()) / len(assessment_scores)
        raw_score = min(1.0, raw_score + (avg_assess / 100) * 0.1)

    matched = required_matches[:6] + preferred_matches[:3]
    return raw_score, matched


def score_title_and_career(c: dict) -> float:
    """Score based on current title and career history relevance."""
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    current_title = profile.get("current_title", "").lower()
    current_company = profile.get("current_company", "").lower()

    # Title score
    title_score = 0.0
    if any(t in current_title for t in STRONG_AI_TITLES):
        title_score = 1.0
    elif any(t in current_title for t in MEDIUM_AI_TITLES):
        title_score = 0.55
    elif "analyst" in current_title:
        title_score = 0.3
    elif any(w in current_title for w in ["manager", "accountant", "designer",
                                           "writer", "support", "sales", "hr",
                                           "civil", "mechanical", "operations"]):
        title_score = 0.05  # Clear mismatches

    # Career history: bonus for past AI/ML roles at product companies
    career_bonus = 0.0
    all_services = True  # Track if entire career is services-only

    for role in career:
        role_title = role.get("title", "").lower()
        company = role.get("company", "").lower()
        description = role.get("description", "").lower()
        duration = role.get("duration_months", 0)

        # Check if this is a product company role
        is_services = any(svc in company for svc in SERVICES_COMPANIES)
        if not is_services:
            all_services = False

        # AI/ML role in career history
        is_ai_title = any(t in role_title for t in STRONG_AI_TITLES)
        is_medium = any(t in role_title for t in MEDIUM_AI_TITLES)

        # Description mentions core AI concepts
        ai_desc_signals = sum(1 for kw in [
            "embeddings", "vector", "ranking", "retrieval", "nlp", "transformer",
            "pytorch", "tensorflow", "model", "training", "inference", "rag",
            "recommendation", "search", "llm", "fine-tun"
        ] if kw in description)

        if is_ai_title and duration >= 6:
            career_bonus += 0.15
        elif is_medium and ai_desc_signals >= 3 and duration >= 6:
            career_bonus += 0.10
        elif ai_desc_signals >= 4:
            career_bonus += 0.06

    # Penalty: entire career in services companies without product-company experience
    if all_services and len(career) >= 2:
        title_score *= 0.6
        career_bonus *= 0.6

    return min(1.0, title_score + min(0.4, career_bonus))


def score_experience(c: dict) -> float:
    """Score based on years of experience (target: 5-9 years for this JD)."""
    profile = c.get("profile", {})
    years = profile.get("years_of_experience", 0)

    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5 or 9 < years <= 11:
        return 0.8
    elif 3 <= years < 4 or 11 < years <= 13:
        return 0.6
    elif 2 <= years < 3 or 13 < years <= 15:
        return 0.4
    else:
        return 0.2


def score_behavioral_signals(c: dict) -> float:
    """Score behavioral signals — availability and engagement multiplier."""
    signals = c.get("redrob_signals", {})

    score = 0.0
    weights = 0.0

    # 1. Open to work — very important
    if signals.get("open_to_work_flag"):
        score += 0.25
    weights += 0.25

    # 2. Recent activity (last_active_date)
    last_active = signals.get("last_active_date", "2020-01-01")
    try:
        last_active_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
        days_ago = (date(2026, 6, 21) - last_active_dt).days
        if days_ago <= 30:
            activity_score = 1.0
        elif days_ago <= 90:
            activity_score = 0.8
        elif days_ago <= 180:
            activity_score = 0.5
        elif days_ago <= 365:
            activity_score = 0.3
        else:
            activity_score = 0.0
    except (ValueError, TypeError):
        activity_score = 0.0
    score += 0.20 * activity_score
    weights += 0.20

    # 3. Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0)
    score += 0.15 * min(1.0, response_rate / 0.5)
    weights += 0.15

    # 4. Profile completeness
    completeness = signals.get("profile_completeness_score", 0)
    score += 0.10 * (completeness / 100)
    weights += 0.10

    # 5. GitHub activity (relevant for AI engineers)
    github = signals.get("github_activity_score", -1)
    if github >= 0:
        score += 0.10 * (github / 100)
    else:
        score += 0.0  # No GitHub linked — neutral/slight negative
    weights += 0.10

    # 6. Interview completion rate
    icr = signals.get("interview_completion_rate", 0)
    score += 0.08 * icr
    weights += 0.08

    # 7. Saved by recruiters in 30d (market signal)
    saved = signals.get("saved_by_recruiters_30d", 0)
    score += 0.07 * min(1.0, saved / 10)
    weights += 0.07

    # 8. Notice period — JD says sub-30-day preferred
    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2
    score += 0.05 * notice_score
    weights += 0.05

    return score / weights if weights > 0 else 0.0


def score_education(c: dict) -> float:
    """Score education — tier 1/2 institutions boost, relevant fields."""
    education = c.get("education", [])
    if not education:
        return 0.3

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        tier_score = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.35}.get(tier, 0.3)

        field_score = 0.5  # default
        ai_fields = ["computer science", "machine learning", "artificial intelligence",
                     "data science", "statistics", "mathematics", "information technology",
                     "electronics", "electrical", "information systems"]
        if any(f in field for f in ai_fields):
            field_score = 1.0

        degree_score = 1.0 if any(d in degree for d in ["b.tech", "m.tech", "b.e.", "m.e.",
                                                          "ph.d", "m.sc", "b.sc", "m.s",
                                                          "b.s"]) else 0.7

        edu_score = tier_score * 0.5 + field_score * 0.3 + degree_score * 0.2
        best_score = max(best_score, edu_score)

    return best_score


def compute_score(c: dict) -> tuple[float, str]:
    """Compute final composite score for a candidate. Returns (score, reasoning)."""

    # Honeypot check — immediately disqualify
    if detect_honeypot(c):
        return 0.01, "Honeypot: impossible profile signals detected."

    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    cid = c.get("candidate_id", "")

    # Component scores
    skill_score, matched_skills = score_skills(c)
    title_score = score_title_and_career(c)
    exp_score = score_experience(c)
    behavioral = score_behavioral_signals(c)
    edu_score = score_education(c)

    # Weights — skills and title/career are most important for this JD
    composite = (
        skill_score   * 0.35 +
        title_score   * 0.30 +
        exp_score     * 0.10 +
        behavioral    * 0.18 +
        edu_score     * 0.07
    )

    # Hard disqualifiers from JD:
    # 1. Pure services career with no product-company experience
    career = c.get("career_history", [])
    all_services = all(
        any(svc in r.get("company", "").lower() for svc in SERVICES_COMPANIES)
        for r in career
    ) if career else False
    if all_services and len(career) >= 2:
        composite *= 0.55

    # 2. Candidate not open to work AND not recently active
    if not signals.get("open_to_work_flag") and behavioral < 0.3:
        composite *= 0.75

    composite = max(0.0, min(1.0, composite))

    # Build reasoning string
    title = profile.get("current_title", "N/A")
    years = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "N/A")
    location = profile.get("location", "N/A")
    country = profile.get("country", "")
    response_rate = signals.get("recruiter_response_rate", 0)
    last_active = signals.get("last_active_date", "N/A")
    open_flag = signals.get("open_to_work_flag", False)
    notice = signals.get("notice_period_days", "N/A")
    skills_display = ", ".join(matched_skills[:4]) if matched_skills else "no core AI skills matched"

    reasons = []
    if title_score >= 0.8:
        reasons.append(f"strong AI/ML title ({title})")
    elif title_score >= 0.5:
        reasons.append(f"adjacent title ({title})")
    else:
        reasons.append(f"non-technical title ({title})")

    reasons.append(f"{years}yr exp")
    if matched_skills:
        reasons.append(f"skills: {skills_display}")
    reasons.append(f"response rate {response_rate:.0%}")
    if open_flag:
        reasons.append("open to work")
    else:
        reasons.append("not marked open to work")
    if notice != "N/A":
        reasons.append(f"{notice}d notice")
    reasons.append(f"last active {last_active}")

    location_str = f"{location}, {country}" if country else location
    reasoning = f"{'; '.join(reasons[:6])}; at {company}, {location_str}."

    return round(composite, 4), reasoning


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def rank_candidates(candidates_path: str, out_path: str, top_n: int = 100):
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                candidates.append(c)
            except json.JSONDecodeError:
                print(f"  Warning: skipped bad line {i+1}")

    print(f"Loaded {len(candidates):,} candidates. Scoring...")

    scored = []
    for c in candidates:
        score, reasoning = compute_score(c)
        scored.append({
            "candidate_id": c["candidate_id"],
            "score": score,
            "reasoning": reasoning,
        })

    # Sort descending by score, break ties by candidate_id ascending
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    top = scored[:top_n]

    # Assign ranks 1-100
    rows = []
    prev_score = None
    for rank_idx, item in enumerate(top):
        rank = rank_idx + 1
        rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": item["score"],
            "reasoning": item["reasoning"],
        })
        prev_score = item["score"]

    print(f"Top {top_n} candidates selected. Writing to {out_path}...")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Submission written to {out_path}")
    print("\nTop 5 preview:")
    for r in rows[:5]:
        print(f"  Rank {r['rank']}: {r['candidate_id']} (score={r['score']}) — {r['reasoning'][:80]}...")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of candidates to rank")
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.top)
