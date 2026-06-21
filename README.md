# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Track 1: Data & AI Challenge  
Job: Senior AI Engineer — Founding Team at Redrob AI

## Overview

This project ranks 100,000 candidates from `candidates.jsonl` against a Senior AI Engineer job description and outputs the top 100 as a CSV.

## Files

| File | Description |
|------|-------------|
| `rank.py` | Main ranking script — produces the submission CSV |
| `validate_submission.py` | Official validator from the hackathon bundle |
| `submission.csv` | Our final submission |
| `submission_metadata.yaml` | Submission metadata |

## Setup

Python 3.11+ required. No external packages needed (pure stdlib).

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: ~60 seconds for 100,000 candidates on CPU (no GPU, no API calls).

## Approach

### Scoring Components (weighted composite)

| Component | Weight | Description |
|-----------|--------|-------------|
| Skills match | 35% | AI/ML skill relevance with trust multipliers |
| Title & Career | 30% | Current title + career history at product companies |
| Behavioral signals | 18% | Availability, activity, response rate, GitHub |
| Experience years | 10% | Target 5-9 years per JD |
| Education | 7% | Institution tier + relevant field |

### Skills Scoring

Skills are matched against two tiers:
- **Required skills** (2× weight): embeddings, vector DBs (Pinecone/Milvus/FAISS/Qdrant/Weaviate), NLP, RAG, LLMs, Python, ranking/retrieval, NDCG/MRR/evaluation
- **Preferred skills** (1× weight): LLM fine-tuning (LoRA/QLoRA), PyTorch, MLOps, distributed systems, recommendation systems

Trust multiplier = `proficiency × duration_normalised × endorsements_normalised + assessment_bonus`  
This penalises keyword stuffing (high proficiency, 0-duration skills).

### Title & Career Scoring

- **Strong AI titles** (1.0): ML/AI/NLP Engineer, Applied Scientist, Data Scientist, etc.
- **Medium titles** (0.55): Software/Backend/Data Engineer — boosted if career descriptions mention AI/ML work
- **Non-technical titles** (0.05): Marketing Manager, Accountant, Customer Support, etc.
- **Services-only penalty**: Candidates whose entire career is at TCS/Infosys/Wipro/Accenture/etc. (no product-company experience) receive a 0.55× multiplier as per JD's explicit disqualifier

### Behavioral Signals

The JD explicitly warns that a "perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is not actually available."

Signals used:
- `open_to_work_flag` (25%): strong positive signal
- `last_active_date` (20%): recency-decayed — 1.0 if active within 30d, 0.0 if >1 year
- `recruiter_response_rate` (15%): hiring ease signal
- `profile_completeness_score` (10%)
- `github_activity_score` (10%): critical for AI engineers
- `interview_completion_rate` (8%): reliability
- `saved_by_recruiters_30d` (7%): market demand signal
- `notice_period_days` (5%): JD prefers <30 days

### Honeypot Detection

The spec warns of ~80 honeypot profiles. Our detector flags:
1. ≥3 skills with "expert" proficiency but 0 `duration_months`
2. >8 "expert" skills total
3. Claimed experience >5 years but total career history <24 months
4. Roles with start dates after 2024 with >24 months duration (impossible)

Flagged candidates receive a score of 0.01.

## Reproducing the Submission

```bash
# From the repo root:
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate the output:
python validate_submission.py ./submission.csv
# Expected output: Submission is valid.
```

## Validation

```
$ python validate_submission.py submission.csv
Submission is valid.
```
