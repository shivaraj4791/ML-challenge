"""
training_pairs.py
=================
Memory-efficient pipeline that generates labeled candidate pairs for training
an entity-resolution classifier for the Amazon ML Challenge.

Key Design Principles
---------------------
1. Zero Cartesian Product: Candidate pairs are generated via bipartite blocking
   (S1 queried against S2 and S3 blocking indexes). Never performs S1 x S2 or S1 x S3.
2. Grouped S1 Splitting: Data is partitioned deterministically by S1 entity_id,
   never by individual pair, guaranteeing zero entity leakage between train, val,
   and holdout sets.
3. Hard Negatives Prioritization:
   - Tier 1: Same country + identical normalized business name, but not GT match.
   - Tier 2: Same country + similar name (prefix/tokens) + different address.
   - Tier 3: Ordinary same-country candidates from blocking index.
   No arbitrary cross-country pairs are used as the negative class.
4. Class Balance: Configurable positive:negative ratios (1:1, 1:2, 1:3, etc.).
5. Memory Safe & Chunked: Output written in chunked shards (Parquet if pyarrow
   available, else CSV/TSV) bounded by configurable shard size.
6. Schema Compliance: Emits all 26 feature columns from src/features.py plus
   metadata columns ('s1_id', 'secondary_id', 'source', 'label'). IDs are
   strictly metadata, not model features.
7. Verification & Audit: Full leakage validation and statistical summary export
   to output/training_pairs_summary.txt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from .blocking import (
    BlockIndex,
    build_country_index,
    build_name_index,
    build_prefix_index,
    build_token_index,
    get_country_candidates,
    get_name_candidates,
    get_prefix_candidates,
    get_token_candidates,
)
from .features import (
    SimilarityWeights,
    build_pair_features,
    feature_names,
)
from .normalization import (
    is_missing,
    normalize_address,
    normalize_business_name,
    normalize_country,
    safe_field,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema Constants
# ---------------------------------------------------------------------------

#: Metadata columns preceding feature columns in generated training shards.
METADATA_COLUMNS: list[str] = [
    "s1_id",
    "secondary_id",
    "source",
    "label",
]

#: Model feature columns (26 features) from src/features.py.
FEATURE_COLUMNS: list[str] = feature_names()

#: Complete ordered column list for output shards.
TRAINING_SHARD_COLUMNS: list[str] = METADATA_COLUMNS + FEATURE_COLUMNS

#: Check if parquet support is available in the current Python environment.
def has_parquet_support() -> bool:
    """Return True if pyarrow or fastparquet is available for parquet output."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Ground Truth Membership Index
# ---------------------------------------------------------------------------

class GroundTruthIndex:
    """Compact in-memory index for rapid ground truth membership lookups.

    Maps S1 entity IDs to their true matching S2 and S3 entity IDs.
    Provides O(1) membership checks and disjoint source access.
    """

    def __init__(self) -> None:
        self._s1_to_s2: dict[str, set[str]] = defaultdict(set)
        self._s1_to_s3: dict[str, set[str]] = defaultdict(set)
        self._all_s1: set[str] = set()
        self._total_s2_matches: int = 0
        self._total_s3_matches: int = 0

    def add_match(self, s1_id: str, matched_id: str) -> None:
        """Add a single true match pair (s1_id, matched_id)."""
        s1_id = str(s1_id).strip()
        matched_id = str(matched_id).strip()
        if not s1_id or not matched_id:
            return

        self._all_s1.add(s1_id)
        if matched_id.startswith("S2"):
            if matched_id not in self._s1_to_s2[s1_id]:
                self._s1_to_s2[s1_id].add(matched_id)
                self._total_s2_matches += 1
        elif matched_id.startswith("S3"):
            if matched_id not in self._s1_to_s3[s1_id]:
                self._s1_to_s3[s1_id].add(matched_id)
                self._total_s3_matches += 1
        else:
            # Fallback if prefix does not start with S2 or S3
            if matched_id not in self._s1_to_s2[s1_id]:
                self._s1_to_s2[s1_id].add(matched_id)
                self._total_s2_matches += 1

    def add_matches(self, s1_id: str, matched_ids: Iterable[str]) -> None:
        """Add multiple true matches for an S1 entity."""
        s1_id = str(s1_id).strip()
        if not s1_id:
            return
        self._all_s1.add(s1_id)
        for mid in matched_ids:
            self.add_match(s1_id, mid)

    def is_match(self, s1_id: str, secondary_id: str) -> bool:
        """Return True if (s1_id, secondary_id) is a ground-truth positive match."""
        s1_id = str(s1_id).strip()
        secondary_id = str(secondary_id).strip()
        if secondary_id.startswith("S2"):
            return secondary_id in self._s1_to_s2.get(s1_id, set())
        elif secondary_id.startswith("S3"):
            return secondary_id in self._s1_to_s3.get(s1_id, set())
        return (
            secondary_id in self._s1_to_s2.get(s1_id, set())
            or secondary_id in self._s1_to_s3.get(s1_id, set())
        )

    def get_s2_matches(self, s1_id: str) -> set[str]:
        """Return the set of S2 entity IDs matched to s1_id."""
        return set(self._s1_to_s2.get(str(s1_id).strip(), set()))

    def get_s3_matches(self, s1_id: str) -> set[str]:
        """Return the set of S3 entity IDs matched to s1_id."""
        return set(self._s1_to_s3.get(str(s1_id).strip(), set()))

    def get_all_matches(self, s1_id: str) -> set[str]:
        """Return all matching secondary IDs (S2 and S3) for s1_id."""
        s1_id = str(s1_id).strip()
        return self._s1_to_s2.get(s1_id, set()) | self._s1_to_s3.get(s1_id, set())

    def all_s1_ids(self) -> list[str]:
        """Return a sorted list of all unique S1 entity IDs recorded in the index."""
        return sorted(self._all_s1)

    @property
    def s1_count(self) -> int:
        """Total number of unique S1 entities."""
        return len(self._all_s1)

    @property
    def s2_match_count(self) -> int:
        """Total number of S1-S2 match edges."""
        return self._total_s2_matches

    @property
    def s3_match_count(self) -> int:
        """Total number of S1-S3 match edges."""
        return self._total_s3_matches

    @property
    def total_match_count(self) -> int:
        """Total number of all match edges (S1-S2 + S1-S3)."""
        return self._total_s2_matches + self._total_s3_matches

    @classmethod
    def from_file(
        cls,
        filepath: str | os.PathLike[str],
        sep: str = "\t",
        max_rows: Optional[int] = None,
        filter_s1_ids: Optional[Set[str]] = None,
    ) -> GroundTruthIndex:
        """Parse ground truth TSV into a GroundTruthIndex in a streaming manner.

        Args:
            filepath: Path to ground truth TSV file.
            sep: Column separator (default tab).
            max_rows: Optional row limit for testing or small-sample runs.
            filter_s1_ids: Optional set of S1 IDs to restrict loading to.

        Returns:
            Populated GroundTruthIndex instance.
        """
        gt = cls()
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Ground truth file not found: {p}")

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=sep)
            try:
                first_row = next(reader)
            except StopIteration:
                return gt

            # Detect header
            s1_idx = 0
            matches_idx = 1
            has_header = False
            for idx, col in enumerate(first_row):
                col_lower = col.lower()
                if "source1" in col_lower or col_lower == "entity_id" or "s1" in col_lower:
                    s1_idx = idx
                    has_header = True
                elif "match" in col_lower:
                    matches_idx = idx
                    has_header = True

            rows_to_process: Iterator[list[str]]
            if has_header:
                rows_to_process = reader
            else:
                # First row was actual data
                def _chain_first() -> Generator[list[str], None, None]:
                    yield first_row
                    yield from reader

                rows_to_process = _chain_first()

            row_count = 0
            for row in rows_to_process:
                if max_rows is not None and row_count >= max_rows:
                    break
                row_count += 1

                if not row:
                    continue
                s1_id = row[s1_idx].strip() if len(row) > s1_idx else ""
                if not s1_id:
                    continue
                if filter_s1_ids is not None and s1_id not in filter_s1_ids:
                    continue

                raw_matches = row[matches_idx].strip() if len(row) > matches_idx else ""
                gt._all_s1.add(s1_id)

                if raw_matches:
                    for tok in raw_matches.split(","):
                        mid = tok.strip()
                        if mid:
                            gt.add_match(s1_id, mid)

        return gt

    @classmethod
    def from_dict(cls, mapping: dict[str, Union[Sequence[str], Set[str]]]) -> GroundTruthIndex:
        """Create a GroundTruthIndex from an in-memory dict mapping s1_id -> [matched_ids]."""
        gt = cls()
        for s1_id, matches in mapping.items():
            gt._all_s1.add(str(s1_id).strip())
            for mid in matches:
                gt.add_match(s1_id, mid)
        return gt

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[str, str]]) -> GroundTruthIndex:
        """Create a GroundTruthIndex from an iterable of (s1_id, matched_id) tuples."""
        gt = cls()
        for s1_id, matched_id in pairs:
            gt.add_match(s1_id, matched_id)
        return gt


# ---------------------------------------------------------------------------
# Data Splitting by S1 Entity (Zero-Leakage Grouped Partition)
# ---------------------------------------------------------------------------

def split_s1_entities(
    s1_ids: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, set[str]]:
    """Partition unique S1 entity IDs deterministically into train, val, and holdout.

    CRITICAL: Splitting is performed strictly by S1 entity, never by individual
    pairs. No S1 entity will ever appear in more than one partition.

    Args:
        s1_ids: Sequence of S1 entity IDs.
        train_ratio: Proportion of S1 entities allocated to training (default 0.70).
        val_ratio: Proportion of S1 entities allocated to validation (default 0.15).
        holdout_ratio: Proportion of S1 entities allocated to holdout (default 0.15).
        random_seed: Random seed for deterministic reproducibility (default 42).

    Returns:
        Dict with keys 'train', 'val', 'holdout', each containing a set of S1 IDs.
    """
    total_ratio = train_ratio + val_ratio + holdout_ratio
    if not (0.9999 <= total_ratio <= 1.0001):
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {holdout_ratio} = {total_ratio}"
        )
    if train_ratio < 0 or val_ratio < 0 or holdout_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")

    # Deduplicate and sort for deterministic ordering before shuffling
    unique_s1 = sorted({str(x).strip() for x in s1_ids if str(x).strip()})
    n = len(unique_s1)
    if n == 0:
        return {"train": set(), "val": set(), "holdout": set()}

    import random
    rng = random.Random(random_seed)
    shuffled = unique_s1.copy()
    rng.shuffle(shuffled)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Remainder to holdout to avoid off-by-one errors from rounding
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    holdout_ids = set(shuffled[n_train + n_val :])

    # Validation: strictly disjoint partitions
    assert not (train_ids & val_ids), "Entity leakage detected between train and val"
    assert not (train_ids & holdout_ids), "Entity leakage detected between train and holdout"
    assert not (val_ids & holdout_ids), "Entity leakage detected between val and holdout"
    assert len(train_ids) + len(val_ids) + len(holdout_ids) == n, "Entity count mismatch"

    return {
        "train": train_ids,
        "val": val_ids,
        "holdout": holdout_ids,
    }


# ---------------------------------------------------------------------------
# Secondary Blocking & Candidate Search Engine
# ---------------------------------------------------------------------------

@dataclass
class SecondaryBlockIndex:
    """Multi-strategy blocking index built over records of a secondary source (S2 or S3).

    Enables high-speed candidate retrieval for an incoming S1 record without
    ever materializing full cross-source Cartesian joins.
    """
    source_name: str  # 'S2' or 'S3'
    records: dict[str, dict[str, Any]]
    country_index: BlockIndex
    name_index: BlockIndex
    prefix_index: BlockIndex
    token_index: BlockIndex
    prefix_len: int = 5
    min_token_len: int = 3

    def get_candidates(
        self, s1_rec: dict[str, Any]
    ) -> tuple[set[str], set[str], set[str]]:
        """Return candidate secondary entity IDs partitioned by difficulty tiers.

        All candidate IDs returned are guaranteed to be from the SAME country.

        Returns:
            Tuple of:
            - exact_name_cands: Share normalized country + exact normalized business name.
            - similar_name_cands: Share normalized country + prefix or token (excluding exact).
            - ordinary_country_cands: Same country, but neither exact name nor prefix/token.
        """
        # 1. Country candidates (MANDATORY constraint)
        country_cands = get_country_candidates(self.country_index, s1_rec)
        if not country_cands:
            return set(), set(), set()

        # 2. Exact normalized name match
        exact_name_cands = get_name_candidates(self.name_index, s1_rec) & country_cands

        # 3. Prefix match
        prefix_cands = (
            get_prefix_candidates(self.prefix_index, s1_rec, prefix_len=self.prefix_len)
            & country_cands
        )

        # 4. Token match
        token_cands = (
            get_token_candidates(self.token_index, s1_rec, min_token_len=self.min_token_len)
            & country_cands
        )

        similar_name_cands = (prefix_cands | token_cands) - exact_name_cands
        ordinary_country_cands = country_cands - exact_name_cands - similar_name_cands

        return exact_name_cands, similar_name_cands, ordinary_country_cands


def build_secondary_index(
    records: Iterable[dict[str, Any]],
    source_name: str = "S2",
    prefix_len: int = 5,
    min_token_len: int = 3,
    stopwords: Optional[set[str]] = None,
) -> SecondaryBlockIndex:
    """Build a SecondaryBlockIndex over records of a secondary source (S2 or S3).

    Args:
        records: Iterable of secondary record dicts.
        source_name: 'S2' or 'S3'.
        prefix_len: Prefix length for prefix blocking.
        min_token_len: Minimum token length for token blocking.
        stopwords: Optional token stopwords.

    Returns:
        SecondaryBlockIndex instance.
    """
    rec_dict: dict[str, dict[str, Any]] = {}
    record_list: list[dict[str, Any]] = []

    for r in records:
        eid = str(r["entity_id"]).strip()
        rec_dict[eid] = r
        record_list.append(r)

    country_idx = build_country_index(record_list)
    name_idx = build_name_index(record_list)
    prefix_idx = build_prefix_index(record_list, prefix_len=prefix_len)
    token_idx = build_token_index(
        record_list, min_token_len=min_token_len, stopwords=stopwords
    )

    return SecondaryBlockIndex(
        source_name=source_name,
        records=rec_dict,
        country_index=country_idx,
        name_index=name_idx,
        prefix_index=prefix_idx,
        token_index=token_idx,
        prefix_len=prefix_len,
        min_token_len=min_token_len,
    )


# ---------------------------------------------------------------------------
# Hard Negative & Positive Pair Generation
# ---------------------------------------------------------------------------

@dataclass
class CandidateGenerationStats:
    """Diagnostic counters for pair generation."""
    positives_s2: int = 0
    positives_s3: int = 0
    hard_exact_name_negatives: int = 0
    hard_similar_name_negatives: int = 0
    easy_same_country_negatives: int = 0
    total_candidates_examined: int = 0

    @property
    def total_positives(self) -> int:
        return self.positives_s2 + self.positives_s3

    @property
    def total_hard_negatives(self) -> int:
        return self.hard_exact_name_negatives + self.hard_similar_name_negatives

    @property
    def total_easy_negatives(self) -> int:
        return self.easy_same_country_negatives

    @property
    def total_negatives(self) -> int:
        return self.total_hard_negatives + self.total_easy_negatives

    @property
    def positive_negative_ratio(self) -> float:
        if self.total_positives == 0:
            return 0.0
        return round(self.total_negatives / self.total_positives, 4)


def parse_ratio(ratio: Union[float, int, str]) -> float:
    """Parse ratio expression (e.g. 2.0, 1, '1:2', '1:3', '1:1') into negative factor float."""
    if isinstance(ratio, (int, float)):
        val = float(ratio)
        if val <= 0:
            raise ValueError(f"Ratio must be positive; got {ratio}")
        return val

    s = str(ratio).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            try:
                pos_part = float(parts[0])
                neg_part = float(parts[1])
                if pos_part <= 0 or neg_part < 0:
                    raise ValueError
                return neg_part / pos_part
            except (ValueError, ZeroDivisionError):
                pass
    try:
        val = float(s)
        if val > 0:
            return val
    except ValueError:
        pass
    raise ValueError(f"Unable to parse class ratio: '{ratio}'. Expected float or '1:N'.")


def generate_candidate_pairs_for_s1(
    s1_rec: dict[str, Any],
    secondary_idx: SecondaryBlockIndex,
    ground_truth: GroundTruthIndex,
    neg_ratio: float = 2.0,
    rng: Optional[Any] = None,
    include_singleton_negatives: bool = True,
    stats: Optional[CandidateGenerationStats] = None,
) -> list[tuple[dict[str, Any], dict[str, Any], str, int, str]]:
    """Generate labeled candidate pairs for a single S1 record against a secondary source.

    Prioritizes HARD NEGATIVES:
    1. Same country + exact normalized business name (not GT match)
    2. Same country + similar name tokens / prefix (not GT match)
       Within tier 2, entities with different addresses are prioritized.
    3. Ordinary same-country candidates from the blocking index.

    Cross-country pairs are NEVER selected as negative pairs.

    Args:
        s1_rec: S1 record dict.
        secondary_idx: SecondaryBlockIndex for S2 or S3.
        ground_truth: GroundTruthIndex for match verification.
        neg_ratio: Ratio of negative pairs to positive pairs (e.g. 2.0 for 1:2).
        rng: Optional random.Random instance for deterministic sampling.
        include_singleton_negatives: If True, singletons with 0 matches still
            generate 1 hard negative candidate.
        stats: Optional diagnostic stats tracker.

    Returns:
        List of tuples: (s1_rec, secondary_rec, source_name, label, difficulty)
        where label is 1 for positive, 0 for negative.
    """
    s1_id = str(s1_rec["entity_id"]).strip()
    source_name = secondary_idx.source_name

    # 1. True Positives from Ground Truth
    if source_name == "S2":
        true_matches = ground_truth.get_s2_matches(s1_id)
    else:
        true_matches = ground_truth.get_s3_matches(s1_id)

    pairs: list[tuple[dict[str, Any], dict[str, Any], str, int, str]] = []

    for mid in sorted(true_matches):
        if mid in secondary_idx.records:
            sec_rec = secondary_idx.records[mid]
            pairs.append((s1_rec, sec_rec, source_name, 1, "positive"))
            if stats is not None:
                if source_name == "S2":
                    stats.positives_s2 += 1
                else:
                    stats.positives_s3 += 1

    num_pos = len(pairs)

    # 2. Determine target negative count for this S1 record
    if num_pos > 0:
        target_negs = max(1, int(round(num_pos * neg_ratio)))
    elif include_singleton_negatives:
        target_negs = 1
    else:
        target_negs = 0

    if target_negs <= 0:
        return pairs

    # 3. Retrieve blocked candidates
    exact_cands, similar_cands, country_cands = secondary_idx.get_candidates(s1_rec)

    if stats is not None:
        stats.total_candidates_examined += (
            len(exact_cands) + len(similar_cands) + len(country_cands)
        )

    # Filter out true positive matches from negative candidates
    neg_exact = [cid for cid in sorted(exact_cands) if cid not in true_matches and cid in secondary_idx.records]
    neg_similar = [cid for cid in sorted(similar_cands) if cid not in true_matches and cid in secondary_idx.records]
    neg_country = [cid for cid in sorted(country_cands) if cid not in true_matches and cid in secondary_idx.records]

    # Deterministic shuffling if rng provided
    if rng is not None:
        rng.shuffle(neg_exact)
        # Prioritize different addresses within similar name candidates
        s1_addr = normalize_address(s1_rec.get("business_address"))
        diff_addr_similar: list[str] = []
        same_addr_similar: list[str] = []
        for cid in neg_similar:
            c_addr = normalize_address(secondary_idx.records[cid].get("business_address"))
            if s1_addr is not None and c_addr is not None and s1_addr != c_addr:
                diff_addr_similar.append(cid)
            else:
                same_addr_similar.append(cid)
        rng.shuffle(diff_addr_similar)
        rng.shuffle(same_addr_similar)
        neg_similar = diff_addr_similar + same_addr_similar

        rng.shuffle(neg_country)

    chosen_negs: list[tuple[str, str]] = []  # (cid, difficulty)

    # Tier 1: Hardest negatives - exact normalized name match in same country
    for cid in neg_exact:
        if len(chosen_negs) >= target_negs:
            break
        chosen_negs.append((cid, "hard_exact_name"))

    # Tier 2: Hard negatives - similar name tokens / prefix in same country
    if len(chosen_negs) < target_negs:
        for cid in neg_similar:
            if len(chosen_negs) >= target_negs:
                break
            chosen_negs.append((cid, "hard_similar_name"))

    # Tier 3: Easy negatives - ordinary same-country candidates from blocking index
    if len(chosen_negs) < target_negs:
        for cid in neg_country:
            if len(chosen_negs) >= target_negs:
                break
            chosen_negs.append((cid, "easy_same_country"))

    # Add chosen negatives to output pairs
    for cid, diff in chosen_negs:
        sec_rec = secondary_idx.records[cid]
        pairs.append((s1_rec, sec_rec, source_name, 0, diff))
        if stats is not None:
            if diff == "hard_exact_name":
                stats.hard_exact_name_negatives += 1
            elif diff == "hard_similar_name":
                stats.hard_similar_name_negatives += 1
            else:
                stats.easy_same_country_negatives += 1

    return pairs


# ---------------------------------------------------------------------------
# Feature Extraction and Row Assembly
# ---------------------------------------------------------------------------

def build_labeled_pair(
    s1_rec: dict[str, Any],
    secondary_rec: dict[str, Any],
    source: str,
    label: int,
    difficulty: str = "",
    weights: Optional[SimilarityWeights] = None,
) -> dict[str, Any]:
    """Compute pair features and return a flat dict adhering to the shard schema.

    Contains all 26 features from src/features.py plus metadata columns
    's1_id', 'secondary_id', 'source', 'label'.
    """
    s1_id = str(s1_rec["entity_id"]).strip()
    sec_id = str(secondary_rec["entity_id"]).strip()

    feat = build_pair_features(s1_rec, secondary_rec, weights=weights)

    row: dict[str, Any] = {
        "s1_id": s1_id,
        "secondary_id": sec_id,
        "source": source,
        "label": int(label),
    }

    # Populate all 26 feature columns
    for fname in FEATURE_COLUMNS:
        row[fname] = feat.get(fname, 0.0)

    return row


# ---------------------------------------------------------------------------
# Output Shard Writing (Parquet / CSV / TSV)
# ---------------------------------------------------------------------------

def write_shards(
    pairs: Iterable[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    split_name: str,
    shard_size: int = 50_000,
    file_format: str = "auto",
) -> tuple[list[Path], int]:
    """Write an iterable of labeled pair dicts into chunked on-disk shards.

    Args:
        pairs: Iterable of row dicts matching TRAINING_SHARD_COLUMNS.
        output_dir: Directory where shards will be saved.
        split_name: Name of the split ('train', 'val', 'holdout').
        shard_size: Number of pairs per shard file (default 50,000).
        file_format: 'auto', 'parquet', 'csv', or 'tsv'.
            If 'auto', uses parquet if pyarrow/fastparquet available, else csv.

    Returns:
        Tuple of (list_of_created_shard_paths, total_pairs_written).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_parquet = False
    if file_format == "parquet":
        if not has_parquet_support():
            raise ImportError("Parquet requested but neither pyarrow nor fastparquet is installed.")
        use_parquet = True
    elif file_format == "auto":
        use_parquet = has_parquet_support()
    elif file_format in ("csv", "tsv"):
        use_parquet = False
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    ext = "parquet" if use_parquet else ("tsv" if file_format == "tsv" else "csv")
    sep = "\t" if ext == "tsv" else ","

    shard_paths: list[Path] = []
    total_written = 0
    buffer: list[dict[str, Any]] = []
    shard_idx = 0

    def _flush_buffer() -> None:
        nonlocal shard_idx, total_written
        if not buffer:
            return
        shard_filename = f"{split_name}_part_{shard_idx:03d}.{ext}"
        shard_file = out_path / shard_filename
        df = pd.DataFrame(buffer, columns=TRAINING_SHARD_COLUMNS)

        if use_parquet:
            df.to_parquet(shard_file, index=False)
        else:
            df.to_csv(shard_file, sep=sep, index=False)

        shard_paths.append(shard_file)
        total_written += len(buffer)
        shard_idx += 1
        buffer.clear()

    for pair in pairs:
        buffer.append(pair)
        if len(buffer) >= shard_size:
            _flush_buffer()

    # Flush any remaining rows
    _flush_buffer()

    return shard_paths, total_written


# ---------------------------------------------------------------------------
# Leakage Checks & Data Validation
# ---------------------------------------------------------------------------

def validate_training_dataset(
    splits_data: dict[str, Sequence[dict[str, Any]]],
    ground_truth: GroundTruthIndex,
    raw_file_paths: Optional[Sequence[str | os.PathLike[str]]] = None,
    raw_checksums_before: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Execute rigorous leakage and integrity audits on generated training data.

    Validates:
    1. Group isolation: No S1 entity appears in more than one split.
    2. Label integrity: No ground-truth positive match is mislabeled as negative (0).
    3. Pair uniqueness: No duplicate (s1_id, secondary_id) candidate pair exists
       within any split.
    4. Source consistency: S2 and S3 IDs are handled separately and labeled with
       matching source metadata.
    5. Raw data preservation: If raw file paths and checksums are provided,
       verifies that no raw data was modified.

    Raises:
        ValueError: If any leakage or integrity violation is detected.

    Returns:
        Dict summarizing audit results.
    """
    s1_by_split: dict[str, set[str]] = {}
    pair_counts_by_split: dict[str, int] = {}
    label_distribution_by_split: dict[str, dict[int, int]] = {}

    for split_name, pairs in splits_data.items():
        s1_set: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        label_dist: Counter[int] = Counter()

        for idx, row in enumerate(pairs):
            s1_id = str(row["s1_id"]).strip()
            sec_id = str(row["secondary_id"]).strip()
            source = str(row["source"]).strip()
            label = int(row["label"])

            s1_set.add(s1_id)
            label_dist[label] += 1

            # Check 3: Duplicate candidate pair within split
            pair_key = (s1_id, sec_id)
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate pair detected in split '{split_name}': {pair_key} at row {idx}"
                )
            seen_pairs.add(pair_key)

            # Check 2: Ground truth positive mislabeled as negative
            if label == 0:
                if ground_truth.is_match(s1_id, sec_id):
                    raise ValueError(
                        f"Leakage detected: ground truth positive match ({s1_id}, {sec_id}) "
                        f"was mislabeled as negative (0) in split '{split_name}'!"
                    )

            # Check 4: Source tagging consistency
            if sec_id.startswith("S2") and source != "S2":
                raise ValueError(f"Source mismatch for {sec_id}: source is '{source}', expected 'S2'")
            if sec_id.startswith("S3") and source != "S3":
                raise ValueError(f"Source mismatch for {sec_id}: source is '{source}', expected 'S3'")

        s1_by_split[split_name] = s1_set
        pair_counts_by_split[split_name] = len(pairs)
        label_distribution_by_split[split_name] = dict(label_dist)

    # Check 1: No S1 appears in multiple splits
    split_names = list(splits_data.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1_a = s1_by_split[split_names[i]]
            s1_b = s1_by_split[split_names[j]]
            intersection = s1_a & s1_b
            if intersection:
                sample_leak = list(intersection)[:5]
                raise ValueError(
                    f"Entity leakage detected between '{split_names[i]}' and '{split_names[j]}'! "
                    f"{len(intersection)} S1 entities shared: {sample_leak}"
                )

    # Check 5: Raw data untouched
    raw_status = "Not checked"
    if raw_file_paths and raw_checksums_before:
        for rpath in raw_file_paths:
            rp = Path(rpath)
            if not rp.exists():
                raise ValueError(f"Raw file {rp} disappeared!")
            current_md5 = compute_file_hash(rp)
            expected_md5 = raw_checksums_before.get(str(rp))
            if expected_md5 and current_md5 != expected_md5:
                raise ValueError(f"CRITICAL: Raw file {rp} was modified during processing!")
        raw_status = "Verified untouched"

    return {
        "status": "PASS",
        "s1_counts": {k: len(v) for k, v in s1_by_split.items()},
        "pair_counts": pair_counts_by_split,
        "label_distribution": label_distribution_by_split,
        "raw_data_check": raw_status,
    }


def compute_file_hash(filepath: str | os.PathLike[str], max_bytes: int = 10_000_000) -> str:
    """Compute MD5 checksum of the beginning of a file to verify it was not modified."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        chunk = f.read(max_bytes)
        h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Summary Report Generation
# ---------------------------------------------------------------------------

def write_summary(
    stats: CandidateGenerationStats,
    s1_split_counts: dict[str, int],
    shard_info: dict[str, list[dict[str, Any]]],
    output_path: str | os.PathLike[str] = "output/training_pairs_summary.txt",
) -> str:
    """Format and write the comprehensive training pairs generation summary file.

    Includes:
    - positive count
    - negative count
    - positive/negative ratio
    - S2/S3 breakdown
    - train/validation/holdout S1 counts
    - candidate-generation counts
    - number of hard negatives
    - number of easy negatives
    - feature columns
    - output shard sizes
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 75)
    lines.append("AMAZON ML CHALLENGE 2026 - TRAINING PAIR GENERATION SUMMARY")
    lines.append("=" * 75)
    lines.append("")
    lines.append("1. PAIR COUNTS & CLASS BALANCE:")
    lines.append(f"  - Positive pairs total:          {stats.total_positives:,}")
    lines.append(f"  - Negative pairs total:          {stats.total_negatives:,}")
    lines.append(f"  - Positive/Negative ratio:       1 : {stats.positive_negative_ratio:.2f}")
    lines.append("")
    lines.append("2. SOURCE BREAKDOWN (MATCHES TO S1):")
    lines.append(f"  - S1-S2 positive matches:        {stats.positives_s2:,}")
    lines.append(f"  - S1-S3 positive matches:        {stats.positives_s3:,}")
    lines.append("")
    lines.append("3. NEGATIVE DIFFICULTY BREAKDOWN:")
    lines.append(f"  - Total hard negatives:          {stats.total_hard_negatives:,}")
    lines.append(f"    * Hard (Exact normalized name) {stats.hard_exact_name_negatives:,}")
    lines.append(f"    * Hard (Similar name tokens/pfx){stats.hard_similar_name_negatives:,}")
    lines.append(f"  - Total easy negatives (country): {stats.total_easy_negatives:,}")
    lines.append(f"  - Total candidates examined:     {stats.total_candidates_examined:,}")
    lines.append("")
    lines.append("4. ENTITY SPLIT (GROUPED BY S1):")
    for split_name, count in s1_split_counts.items():
        lines.append(f"  - S1 entities in {split_name:<10}:    {count:,}")
    lines.append("")
    lines.append("5. MODEL FEATURE SCHEMA (26 FEATURES):")
    for idx, fname in enumerate(FEATURE_COLUMNS, 1):
        lines.append(f"  {idx:2d}. {fname}")
    lines.append("")
    lines.append("6. OUTPUT SHARDS & SIZES:")
    total_shard_bytes = 0
    for split_name, shards in shard_info.items():
        lines.append(f"  [{split_name.upper()} SHARDS]")
        for sh in shards:
            size_bytes = sh.get("size_bytes", 0)
            rows = sh.get("rows", 0)
            path_str = sh.get("path", "")
            total_shard_bytes += size_bytes
            lines.append(f"    - {path_str}: {rows:,} rows, {size_bytes / (1024 * 1024):.2f} MB ({size_bytes:,} bytes)")
    lines.append(f"  Total disk space: {total_shard_bytes / (1024 * 1024):.2f} MB")
    lines.append("=" * 75)

    summary_text = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text


# ---------------------------------------------------------------------------
# End-to-End Orchestrator Pipeline
# ---------------------------------------------------------------------------

def run_training_pair_pipeline(
    s1_records: Iterable[dict[str, Any]],
    s2_records: Iterable[dict[str, Any]],
    s3_records: Iterable[dict[str, Any]],
    ground_truth: GroundTruthIndex,
    output_dir: str | os.PathLike[str] = "output/training_pairs",
    summary_path: str | os.PathLike[str] = "output/training_pairs_summary.txt",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    neg_ratio: Union[float, int, str] = 2.0,
    random_seed: int = 42,
    shard_size: int = 50_000,
    file_format: str = "auto",
    include_singleton_negatives: bool = True,
    weights: Optional[SimilarityWeights] = None,
) -> dict[str, Any]:
    """Execute the end-to-end memory-efficient training pair generation pipeline.

    Args:
        s1_records: Iterable of S1 record dicts.
        s2_records: Iterable of S2 record dicts.
        s3_records: Iterable of S3 record dicts.
        ground_truth: GroundTruthIndex instance.
        output_dir: Directory for writing candidate pair shards.
        summary_path: Filepath for writing the human-readable summary.
        train_ratio: S1 split train proportion.
        val_ratio: S1 split val proportion.
        holdout_ratio: S1 split holdout proportion.
        neg_ratio: Positive to negative ratio (e.g. 2.0, '1:2', '1:3').
        random_seed: Deterministic seed for reproducible splits and negative sampling.
        shard_size: Maximum pairs per shard file.
        file_format: 'auto', 'parquet', 'csv', or 'tsv'.
        include_singleton_negatives: Whether to generate negatives for singletons.
        weights: Optional SimilarityWeights for feature calculation.

    Returns:
        Dict summarizing execution metrics, shards created, and leakage audits.
    """
    ratio_val = parse_ratio(neg_ratio)

    # 1. Build secondary blocking indices
    log.info("Building secondary blocking index for S2...")
    s2_idx = build_secondary_index(s2_records, source_name="S2")
    log.info("Building secondary blocking index for S3...")
    s3_idx = build_secondary_index(s3_records, source_name="S3")

    # 2. Materialize S1 records in memory or stream (if small / medium)
    s1_list = list(s1_records)
    all_s1_ids = [str(r["entity_id"]).strip() for r in s1_list]

    # 3. Deterministic entity split
    splits = split_s1_entities(
        all_s1_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        holdout_ratio=holdout_ratio,
        random_seed=random_seed,
    )

    s1_to_split: dict[str, str] = {}
    for sname, eids in splits.items():
        for eid in eids:
            s1_to_split[eid] = sname

    # 4. Generate pairs for each S1 record and route to split buffers
    import random
    rng = random.Random(random_seed)
    stats = CandidateGenerationStats()

    split_pairs: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "holdout": [],
    }

    shard_info: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "holdout": [],
    }

    for s1_rec in s1_list:
        s1_id = str(s1_rec["entity_id"]).strip()
        split_name = s1_to_split.get(s1_id, "train")

        # Generate against S2
        raw_pairs_s2 = generate_candidate_pairs_for_s1(
            s1_rec,
            s2_idx,
            ground_truth,
            neg_ratio=ratio_val,
            rng=rng,
            include_singleton_negatives=include_singleton_negatives,
            stats=stats,
        )

        # Generate against S3
        raw_pairs_s3 = generate_candidate_pairs_for_s1(
            s1_rec,
            s3_idx,
            ground_truth,
            neg_ratio=ratio_val,
            rng=rng,
            include_singleton_negatives=include_singleton_negatives,
            stats=stats,
        )

        for s1_r, sec_r, src, label, diff in raw_pairs_s2 + raw_pairs_s3:
            row = build_labeled_pair(
                s1_r, sec_r, src, label, difficulty=diff, weights=weights
            )
            split_pairs[split_name].append(row)

    # 5. Run rigorous leakage validation
    log.info("Running leakage and integrity validation...")
    validation_report = validate_training_dataset(
        split_pairs,
        ground_truth=ground_truth,
    )

    # 6. Write chunked shards for each split
    log.info("Writing output shards...")
    for split_name, pairs in split_pairs.items():
        shards, count = write_shards(
            pairs,
            output_dir=output_dir,
            split_name=split_name,
            shard_size=shard_size,
            file_format=file_format,
        )
        for sh in shards:
            shard_info[split_name].append({
                "path": str(sh),
                "rows": count if len(shards) == 1 else min(shard_size, count),
                "size_bytes": os.path.getsize(sh),
            })

    # 7. Write summary file
    s1_split_counts = {k: len(v) for k, v in splits.items()}
    summary_text = write_summary(
        stats=stats,
        s1_split_counts=s1_split_counts,
        shard_info=shard_info,
        output_path=summary_path,
    )

    return {
        "status": "SUCCESS",
        "stats": stats,
        "validation": validation_report,
        "shards": shard_info,
        "summary_path": str(summary_path),
    }
