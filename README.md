# Amazon ML Challenge 2026 - Multilingual Entity Resolution System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-212%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)]()

A high-performance, memory-efficient entity resolution pipeline engineered for the **Amazon ML Challenge 2026**. The system matches heterogeneous, multilingual business records across three distinct data sources (Source 1 $\rightarrow$ Source 2 and Source 1 $\rightarrow$ Source 3) under strict zero-leakage, bounded-RAM, and real-world scale constraints (2.52 GB raw data, >2.2M primary entities).

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Dataset Characteristics & Profiling](#dataset-characteristics--profiling)
- [Repository Structure](#repository-structure)
- [Core Modules](#core-modules)
  - [1. Unicode-Safe Normalization (`src/normalization.py`)](#1-unicode-safe-normalization-srcnormalizationpy)
  - [2. Streaming Data Loader (`src/data_loader.py`)](#2-streaming-data-loader-srcdata_loaderpy)
  - [3. Multi-Strategy Blocking (`src/blocking.py`)](#3-multi-strategy-blocking-srcblockingpy)
  - [4. Feature Engineering (`src/features.py`)](#4-feature-engineering-srcfeaturespy)
  - [5. Evaluation Engine (`src/evaluation.py`)](#5-evaluation-engine-srcevaluationpy)
  - [6. Training Pair Generation & Hard Negative Mining (`src/training_pairs.py`)](#6-training-pair-generation--hard-negative-mining-srctraining_pairspy)
- [Feature Schema (26 ML Features)](#feature-schema-26-ml-features)
- [Zero-Leakage & Data Integrity Guarantees](#zero-leakage--data-integrity-guarantees)
- [Installation & Setup](#installation--setup)
- [Running the Test Suite](#running-the-test-suite)
- [Usage Examples](#usage-examples)
- [Roadmap & Next Steps](#roadmap--next-steps)

---

## System Architecture

```mermaid
flowchart TD
    subgraph Data Sources
        S1[Source 1: Primary Entities]
        S2[Source 2: Secondary Entities]
        S3[Source 3: Secondary Entities]
        GT[Train Ground Truth]
    end

    subgraph Preprocessing & Normalization
        NORM[Unicode NFC Normalization<br/>Devanagari Preservation<br/>Legal Suffix Canonicalization<br/>Address & Country Standards]
    end

    subgraph Bipartite Blocking
        IDX2[S2 Multi-Key Index<br/>Country + Name + Prefix + Token]
        IDX3[S3 Multi-Key Index<br/>Country + Name + Prefix + Token]
        BLOCK[Candidate Generator<br/>Prunes Cross-Country & O(N^2) Space]
    end

    subgraph Hard Negative Mining & Class Balancing
        HNEG[Hard Negative Prioritization<br/>Tier 1: Identical Normalized Name<br/>Tier 2: Token / Prefix Match + Different Address<br/>Tier 3: Same-Country Blocking Candidates]
        SPLIT[Grouped S1 Entity Split<br/>Train: 70% | Val: 15% | Holdout: 15%<br/>Strict Disjoint Partition]
    end

    subgraph Feature Engineering
        FEAT[26 ML Features<br/>Country + Name + Address + Interaction<br/>Missing-Address Resilience]
    end

    subgraph Sharded Output & Validation
        SHARDS[Chunked Shards<br/>Parquet / CSV Parts]
        AUDIT[Leakage & Integrity Auditor<br/>0 S1 Cross-Split Leaks<br/>0 Mislabeled Positives]
        SUMM[Summary Report<br/>output/training_pairs_summary.txt]
    end

    S1 & S2 & S3 --> NORM
    NORM --> IDX2 & IDX3
    S1 & IDX2 & IDX3 --> BLOCK
    BLOCK & GT --> HNEG
    HNEG --> SPLIT
    SPLIT --> FEAT
    FEAT --> SHARDS
    SHARDS --> AUDIT --> SUMM
```

---

## Dataset Characteristics & Profiling

Profiling of the 2.52 GB raw dataset revealed key distributional and schema properties:

- **Entity Schema**: `entity_id`, `business_name`, `business_address`, `country`.
- **Entity Volume**: Source 1 comprises 2,206,821 primary entities; Ground truth contains 7,638,365 match edges (3,693,619 S1-S2 matches and 3,944,746 S1-S3 matches).
- **Match Multiplicity**: 80.48% of S1 entities match both S2 and S3; 5.58% are true singletons (0 matches).
- **Missing Value Profile**:
  - `country`: 0.0% missing across all sources.
  - `business_name`: 0.0% missing across all sources.
  - `business_address`: 0.0% missing in S1; ~3.3% missing in S2 and S3.
- **Multilingual Scope**: Training data covers **US** and **India** (`IN`), with test sets adding **France** (`FR`). Names include accented Latin (`Société Générale`) and Indic scripts (`रिलायंस इंडस्ट्रीज`).
- **Scale Constraint**: A naive Cartesian join ($2.2\text{M} \times 3.7\text{M} \approx 8.1 \times 10^{12}$ comparisons) is computationally intractable. Bipartite blocking bounds candidate evaluation to a tractable subspace.

---

## Repository Structure

```
.
├── config/                     # Pipeline configurations
├── dataset/                    # Dataset symlinks and paths (raw TSVs remain read-only)
├── models/                     # Trained classifier artifacts (future stage)
├── notebooks/                  # EDA and experimentation notebooks
├── output/                     # Generated artifacts
│   ├── dataset_profile.txt     # Streaming dataset profiling report
│   ├── training_pairs/         # Chunked training shards (train_part_000.csv, etc.)
│   └── training_pairs_summary.txt # Pair generation and class balance audit report
├── src/                        # Production modules
│   ├── __init__.py
│   ├── blocking.py             # Multi-strategy bipartite blocking index
│   ├── data_loader.py          # Chunked streaming TSV data loader
│   ├── evaluation.py           # Precision, Recall, F1, confusion matrices
│   ├── features.py             # 26 ML similarity features & composite scores
│   ├── normalization.py        # Unicode NFC, Devanagari-safe text cleaning
│   ├── profile_dataset.py      # Streaming dataset profiling utility
│   ├── training_pairs.py       # Labeled pair generator, S1 splitting, hard negative miner
│   └── utils.py                # Logging, progress timers, validation helpers
├── tests/                      # Unit and integration test suite (212 tests)
│   ├── __init__.py
│   ├── test_blocking.py        # Blocking index unit tests (30 tests)
│   ├── test_evaluation.py      # Evaluation metric unit tests (53 tests)
│   ├── test_features.py        # Feature extraction unit tests (60 tests)
│   ├── test_normalization.py   # Normalization & script preservation unit tests (44 tests)
│   └── test_training_pairs.py  # Pair generation, splitting, leakage audit tests (21 tests)
├── requirements.txt            # Project dependencies
└── README.md
```

---

## Core Modules

### 1. Unicode-Safe Normalization (`src/normalization.py`)
- **Unicode NFC Composition**: Standardizes decomposed characters (e.g. accented Latin `é`).
- **Devanagari & Non-ASCII Preservation**: Uses character-level Unicode General Category inspection (`unicodedata.category`) instead of destructive ASCII regular expressions (`[^a-z0-9]`), ensuring Indic matras and combining vowels remain intact.
- **Legal Suffix Canonicalization**: Standardizes legal suffixes (`LLC`, `L.L.C.`, `Inc.`, `Ltd.`, `Pvt Ltd`) *prior* to punctuation stripping so dotted abbreviations are accurately mapped.
- **Address Standardization**: Expands road and building abbreviations (`St` $\rightarrow$ `street`, `Ave` $\rightarrow$ `avenue`, `Rd` $\rightarrow$ `road`).
- **Missing Value Handling**: Handles `None`, `NaN`, and empty strings without throwing runtime errors.

### 2. Streaming Data Loader (`src/data_loader.py`)
- **Bounded RAM Consumption**: Streams TSV files in configurable row chunks (default: 50,000 rows $\approx$ 20 MB RAM).
- **Type Safety**: Enforces strict string dtypes on entity IDs (`S1-XXXXXX`, `S2-XXXXXX`, `S3-XXXXXX`) to prevent numeric truncation or scientific notation corruption.
- **Column Pruning**: Supports selective column loading at parse time to eliminate unnecessary memory decoding.

### 3. Multi-Strategy Blocking (`src/blocking.py`)
- **Country Blocking**: Hard constraint partitioning by country (0% missing). Cross-country pairs are strictly rejected.
- **Exact Name Blocking**: Inverted index over normalized business names.
- **Prefix Blocking**: Indexes first $N$ characters (default 5) to capture prefix variations.
- **Token Blocking**: Inverted index over significant tokens (min token length $\ge 3$) with customizable stopwords to capture word reorderings (`Global Tech Inc` vs `Tech Global Inc`).
- **Extensible API**: All strategies implement standard `build_<strategy>_index` and `get_<strategy>_candidates` interfaces.

### 4. Feature Engineering (`src/features.py`)
Computes 26 pairwise ML features across Country, Name, Address, and Interaction dimensions without modifying input records. Features are numeric (`int` or `float`) and robust to missing addresses.

### 5. Evaluation Engine (`src/evaluation.py`)
- Vectorized and sequence-based evaluation computing `tp`, `fp`, `fn`, `tn`, `precision`, `recall`, and $F_1$.
- `evaluate_thresholds`: Sweeps decision thresholds across $[0.0, 1.0]$ with zero-division safety.

### 6. Training Pair Generation & Hard Negative Mining (`src/training_pairs.py`)
- **GroundTruthIndex**: Compact streaming index providing $O(1)$ membership checks for $S_1 \rightarrow S_2$ and $S_1 \rightarrow S_3$ true matches.
- **Grouped S1 Splitting**: Splits records by S1 entity ID with a fixed random seed.
- **Hard Negative Prioritization**:
  1. *Tier 1*: Same country + identical normalized business name (non-GT match).
  2. *Tier 2*: Same country + similar name tokens/prefix + differing address.
  3. *Tier 3*: Ordinary same-country blocking index candidates.
- **Configurable Class Balance**: Supports ratios such as `1:1`, `1:2`, and `1:3`.
- **Chunked On-Disk Shards**: Writes partition files to `output/training_pairs/` (Parquet if `pyarrow` is available, else CSV/TSV).
- **Leakage Auditing**: Verifies zero cross-split S1 leakage, absence of mislabeled ground truth pairs, and pair uniqueness.

---

## Feature Schema (26 ML Features)

Every candidate pair generates the following 30-column schema (4 metadata fields + 26 ML features):

| Category | Feature Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **Metadata** | `s1_id` | `str` | Primary S1 entity ID (not used as an ML feature) |
| | `secondary_id` | `str` | Secondary S2 or S3 entity ID (not used as an ML feature) |
| | `source` | `str` | Secondary source identity (`'S2'` or `'S3'`) |
| | `label` | `int` | Binary target label (`1` = match, `0` = non-match) |
| **Country (2)** | `country_equal` | `int` | Binary indicator: normalized countries match |
| | `same_country` | `int` | Explicit binary indicator |
| **Name (9)** | `name_exact_raw` | `int` | 1 if raw name strings are identical |
| | `name_exact_norm` | `int` | 1 if normalized name strings are identical |
| | `name_jaro_winkler` | `float` | Jaro-Winkler similarity score in $[0.0, 1.0]$ |
| | `name_levenshtein_norm` | `float` | $1 - \frac{\text{edit\_dist}}{\max(\text{len}_a, \text{len}_b)}$ in $[0.0, 1.0]$ |
| | `name_token_similarity` | `float` | Jaccard token similarity over normalized name words |
| | `name_len_diff` | `int` | Absolute length difference of normalized names |
| | `name_len_ratio` | `float` | Length ratio $\frac{\min(\text{len}_a, \text{len}_b)}{\max(\text{len}_a, \text{len}_b)}$ |
| | `name_token_count_diff` | `int` | Difference in token count |
| | `name_char_bigram_sim` | `float` | Character bigram Jaccard (resilient to Indic transliterations) |
| **Address (9)** | `addr_missing` | `int` | 1 if either address is absent (~3.3% in S2/S3) |
| | `addr_both_present` | `int` | 1 if both addresses are present |
| | `addr_exact_norm` | `int` | 1 if normalized addresses are identical |
| | `addr_jaro_winkler` | `float` | Address Jaro-Winkler ($0.0$ when missing) |
| | `addr_levenshtein_norm` | `float` | Normalized address Levenshtein ($0.0$ when missing) |
| | `addr_token_similarity` | `float` | Jaccard token similarity over address words |
| | `addr_len_diff` | `int` | Address length difference ($0$ when missing) |
| | `addr_len_ratio` | `float` | Address length ratio ($0.0$ when missing) |
| | `addr_token_count_diff` | `int` | Address token count difference ($0$ when missing) |
| **Combined (6)** | `combined_weighted_sim` | `float` | Weighted combination ($w_{\text{name}} \cdot \text{name\_sim} + w_{\text{addr}} \cdot \text{addr\_sim}$) |
| | `combined_name_dominant_sim`| `float` | Name similarity fallback when address is absent |
| | `interaction_name_x_addr` | `float` | Multiplicative interaction: $\text{name\_exact\_norm} \times \text{addr\_jaro\_winkler}$ |
| | `country_strong_name` | `int` | High-precision rule: `country_equal & name_jw >= 0.9` |
| | `name_sim_composite` | `float` | Arithmetic mean of name JW, Levenshtein, and Jaccard |
| | `addr_sim_composite` | `float` | Arithmetic mean of address JW, Levenshtein, and Jaccard |

---

## Zero-Leakage & Data Integrity Guarantees

To ensure valid cross-validation and unbiased performance estimation:

1. **Entity-Level Group Partitioning**: Splitting is strictly performed on **unique S1 entity IDs**, never on individual candidate pairs. If an S1 entity is in `train`, all its pairs (positive matches and negatives across S2/S3) reside in `train`.
2. **Disjoint Sets**:
   $$\text{Train}_{S1} \cap \text{Val}_{S1} = \emptyset, \quad \text{Train}_{S1} \cap \text{Holdout}_{S1} = \emptyset, \quad \text{Val}_{S1} \cap \text{Holdout}_{S1} = \emptyset$$
3. **Automated Audit**: `validate_training_dataset()` verifies:
   - Zero S1 ID overlap across splits.
   - Zero ground-truth positives mislabeled as negative (`label == 0`).
   - Zero duplicate pairs within any split.
   - Strict source consistency (`secondary_id` starting with `S2` marked as `source="S2"`).
   - Read-only preservation of raw TSV files via pre- and post-processing MD5 checksums.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PowerShell (Windows) or Bash (Linux / macOS)

### Clone & Install
```bash
git clone https://github.com/shivaraj4791/ML-challenge.git
cd ML-challenge

# Install core dependencies
pip install -r requirements.txt
```

### Core Dependencies (`requirements.txt`)
```text
pandas
tqdm
rapidfuzz
jellyfish
pytest
```
*(Optional: `pyarrow` for Parquet export)*

---

## Running the Test Suite

The repository includes a comprehensive, self-contained unit test suite that executes against synthetic records without requiring the 2.52 GB production dataset:

```bash
# Run complete test suite (212 tests)
pytest tests/ -v

# Run specific module tests
pytest tests/test_normalization.py -v   # 44 normalization & Unicode tests
pytest tests/test_blocking.py -v        # 30 blocking index tests
pytest tests/test_features.py -v        # 60 feature engineering tests
pytest tests/test_evaluation.py -v      # 53 evaluation metric tests
pytest tests/test_training_pairs.py -v  # 21 pair generation & leakage tests
```

---

## Usage Examples

### 1. Unicode Normalization
```python
from src.normalization import normalize_business_name, normalize_address

# Handles Devanagari script safely without removing combining characters
name = normalize_business_name("रिलायंस इंडस्ट्रीज लिमिटेड")
print(name)  # Output: "रिलायंस इंडस्ट्रीज लिमिटेड"

# Handles legal suffix canonicalization before punctuation removal
name = normalize_business_name("Acme Tools, L.L.C.")
print(name)  # Output: "acme tools llc"

# Expands street abbreviations
addr = normalize_address("123 Main St, Apt 4B")
print(addr)  # Output: "123 main street apt 4b"
```

### 2. Multi-Strategy Bipartite Blocking
```python
from src.blocking import build_secondary_index

s2_records = [
    {"entity_id": "S2-001", "business_name": "Acme Tools Inc", "country": "US", "business_address": "100 Main St"},
    {"entity_id": "S2-002", "business_name": "Acme Tooling LLC", "country": "US", "business_address": "200 South Rd"},
    {"entity_id": "S2-003", "business_name": "Omega Tech", "country": "US", "business_address": "300 Oak Ave"},
]

# Build multi-index over secondary source
s2_index = build_secondary_index(s2_records, source_name="S2")

# Query with an S1 record
s1_query = {"entity_id": "S1-100", "business_name": "Acme Tools", "country": "US", "business_address": "100 Main St"}
exact_cands, similar_cands, country_cands = s2_index.get_candidates(s1_query)

print("Exact match candidates:", exact_cands)       # {'S2-001'}
print("Similar token candidates:", similar_cands)   # {'S2-002'}
print("Ordinary country candidates:", country_cands)# {'S2-003'}
```

### 3. Pairwise Feature Extraction
```python
from src.features import build_pair_features, feature_names

rec1 = {"entity_id": "S1-001", "business_name": "Acme Tools LLC", "country": "US", "business_address": "100 Main St"}
rec2 = {"entity_id": "S2-001", "business_name": "Acme Tools Inc", "country": "US", "business_address": "100 Main Street"}

features = build_pair_features(rec1, rec2)
print("Jaro-Winkler Name Similarity:", features["name_jaro_winkler"])
print("Address Exact Norm:", features["addr_exact_norm"])
print("Combined Weighted Similarity:", features["combined_weighted_sim"])
```

### 4. End-to-End Training Shard Generation
```python
from src.training_pairs import GroundTruthIndex, run_training_pair_pipeline

# Initialize ground truth index
gt = GroundTruthIndex.from_file("dataset/train/train_ground_truth.tsv")

# Run streaming pipeline with 1:2 class balance and 70/15/15 S1 split
result = run_training_pair_pipeline(
    s1_records=s1_stream,
    s2_records=s2_records,
    s3_records=s3_records,
    ground_truth=gt,
    output_dir="output/training_pairs",
    summary_path="output/training_pairs_summary.txt",
    train_ratio=0.70,
    val_ratio=0.15,
    holdout_ratio=0.15,
    neg_ratio="1:2",
    random_seed=42,
    shard_size=50_000,
)

print("Status:", result["status"])
print("Audit Report:", result["validation"]["status"])
```

---

## Roadmap & Next Steps

1. **Stage 1 (Completed)**: Dataset profiling, Unicode-safe normalization, streaming data loader, multi-strategy blocking index.
2. **Stage 2 (Completed)**: Pairwise feature engineering (26 features), multi-threshold evaluation engine.
3. **Stage 3 (Completed)**: Grouped S1 splitting, hard negative mining, class balance control, chunked shard output, and zero-leakage validation.
4. **Stage 4 (Upcoming)**: Entity resolution classifier training (LightGBM / XGBoost / CatBoost with cross-entropy and ranking loss).
5. **Stage 5 (Upcoming)**: Multi-source graph clustering / connected component resolution for global entity consolidation.
6. **Stage 6 (Upcoming)**: Final test set inference and submission generation.

---

## Authors & Maintainers

- **Amazon ML Challenge 2026 Team**
- Repository: [shivaraj4791/ML-challenge](https://github.com/shivaraj4791/ML-challenge)
