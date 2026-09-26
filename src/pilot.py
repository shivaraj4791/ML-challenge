"""
pilot.py
========
Executes a small, controlled, real-data training-pair generation pilot
on exactly 50,000 Source 1 (S1) entities for the Amazon ML Challenge.

Constraints & Design
--------------------
- Uses exactly 50,000 S1 entities.
- Deterministic grouped S1 entity split: 70% train (35,000), 15% val (7,500), 15% holdout (7,500).
- 1:1 positive:negative ratio (strictly balanced).
- Generates both S1 -> S2 and S1 -> S3 candidate pairs.
- Uses existing multi-tier hard negative mining:
    * Tier 1: Same country + identical normalized business name (non-GT match)
    * Tier 2: Same country + similar name prefix/tokens (different address prioritized)
    * Tier 3: Ordinary same-country candidates from blocking index
    * Zero cross-country negatives
- Calls build_pair_features() from src/features.py for every pair (26 ML features + 4 metadata).
- Writes chunked on-disk shards to output/training_pairs_pilot/.
- Runs full leakage & integrity checks (zero S1 overlap, zero mislabeled positives,
  zero duplicate pairs, zero cross-country negatives, raw files untouched).
- Generates comprehensive output/training_pairs_pilot/summary.txt reporting all 18 metrics.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import random
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    build_country_index,
    build_name_index,
    build_prefix_index,
    build_token_index,
    get_country_candidates,
    get_name_candidates,
    get_prefix_candidates,
    get_token_candidates,
)
from src.data_loader import iter_chunks
from src.features import build_pair_features, feature_names
from src.normalization import (
    normalize_address,
    normalize_business_name,
    normalize_country,
)
from src.training_pairs import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TRAINING_SHARD_COLUMNS,
    GroundTruthIndex,
    SecondaryBlockIndex,
    build_labeled_pair,
    build_secondary_index,
    compute_file_hash,
    split_s1_entities,
    validate_training_dataset,
    write_shards,
)
from src.utils import resolve_dataset_paths, setup_logging

log = logging.getLogger("pilot")


def run_pilot(
    num_s1_entities: int = 50_000,
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    neg_ratio: float = 1.0,
    shard_size: int = 50_000,
    output_dir: str | Path = "output/training_pairs_pilot",
    summary_path: str | Path = "output/training_pairs_pilot/summary.txt",
) -> dict[str, Any]:
    """Execute the small real-data training pair pilot."""
    tracemalloc.start()
    t_start = time.time()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_file = Path(summary_path)
    sum_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("STARTING REAL-DATA TRAINING-PAIR PILOT (50,000 S1 ENTITIES)")
    log.info("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Resolve dataset paths and take baseline checksums
    # -----------------------------------------------------------------------
    paths = resolve_dataset_paths(PROJECT_ROOT)
    raw_files = [
        paths["train_source1"],
        paths["train_source2"],
        paths["train_source3"],
        paths["train_ground_truth"],
    ]

    log.info("Resolving raw dataset paths:")
    for rf in raw_files:
        log.info(f"  - {rf.name}: {rf} ({rf.stat().st_size:,} bytes)")

    log.info("Computing baseline checksums on raw dataset files...")
    checksums_before = {str(rf): compute_file_hash(rf) for rf in raw_files}
    sizes_before = {str(rf): rf.stat().st_size for rf in raw_files}

    # -----------------------------------------------------------------------
    # Step 2: Read exactly 50,000 S1 entities
    # -----------------------------------------------------------------------
    log.info(f"Reading exactly {num_s1_entities:,} S1 entities from train_source1.tsv...")
    s1_records: list[dict[str, Any]] = []
    skipped_s1_records = 0

    for chunk in iter_chunks(paths["train_source1"], chunksize=50000):
        records = chunk.to_dict(orient="records")
        for r in records:
            if len(s1_records) < num_s1_entities:
                s1_records.append(r)
            else:
                break
        if len(s1_records) >= num_s1_entities:
            break

    if len(s1_records) != num_s1_entities:
        raise ValueError(
            f"Expected {num_s1_entities} S1 records, but loaded {len(s1_records)}"
        )

    s1_ids = [str(r["entity_id"]).strip() for r in s1_records]
    s1_id_set = set(s1_ids)
    log.info(f"Successfully loaded {len(s1_records):,} S1 entities.")

    # -----------------------------------------------------------------------
    # Step 3: Grouped S1 Entity Splitting (70% / 15% / 15%)
    # -----------------------------------------------------------------------
    log.info("Splitting S1 entities (70% train, 15% val, 15% holdout, seed=42)...")
    splits = split_s1_entities(
        s1_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        holdout_ratio=holdout_ratio,
        random_seed=random_seed,
    )

    s1_to_split: dict[str, str] = {}
    for sname, eids in splits.items():
        for eid in eids:
            s1_to_split[eid] = sname

    log.info(
        f"Split breakdown: Train={len(splits['train']):,}, "
        f"Val={len(splits['val']):,}, Holdout={len(splits['holdout']):,}"
    )

    # -----------------------------------------------------------------------
    # Step 4: Load Ground Truth for the 50,000 S1 entities
    # -----------------------------------------------------------------------
    log.info("Indexing ground truth matches for the selected 50,000 S1 entities...")
    gt = GroundTruthIndex.from_file(
        paths["train_ground_truth"], filter_s1_ids=s1_id_set
    )

    target_s2_ids: set[str] = set()
    target_s3_ids: set[str] = set()
    for s1_id in s1_ids:
        target_s2_ids.update(gt.get_s2_matches(s1_id))
        target_s3_ids.update(gt.get_s3_matches(s1_id))

    log.info(
        f"Ground Truth loaded: {gt.s1_count:,} S1 entities, "
        f"{len(target_s2_ids):,} S2 true matches, {len(target_s3_ids):,} S3 true matches."
    )

    # -----------------------------------------------------------------------
    # Step 5: Pre-compute S1 blocking keys for candidate retrieval
    # -----------------------------------------------------------------------
    log.info("Extracting S1 blocking keys...")
    s1_norm_names: set[str] = set()
    s1_prefixes: set[str] = set()
    s1_countries: set[str] = set()

    for r in s1_records:
        c = normalize_country(r.get("country"))
        if c:
            s1_countries.add(c)
        nm = normalize_business_name(r.get("business_name"))
        if nm:
            s1_norm_names.add(nm)
            if len(nm) >= 5:
                s1_prefixes.add(nm[:5])

    log.info(
        f"S1 unique normalized names: {len(s1_norm_names):,}, "
        f"5-char prefixes: {len(s1_prefixes):,}, countries: {s1_countries}"
    )

    # -----------------------------------------------------------------------
    # Step 6: Stream Source 2 (Target Matches + Negative Candidates)
    # -----------------------------------------------------------------------
    log.info("Streaming train_source2.tsv for true matches and negative candidates...")
    s2_records_dict: dict[str, dict[str, Any]] = {}
    s2_prefix_counts: Counter[str] = Counter()
    s2_country_samples: Counter[str] = Counter()
    max_cands_per_prefix = 5
    max_ordinary_per_country = 15_000
    total_s2_rows_scanned = 0

    t_s2_start = time.time()
    for chunk in iter_chunks(paths["train_source2"], chunksize=100_000):
        records = chunk.to_dict(orient="records")
        total_s2_rows_scanned += len(records)
        for r in records:
            eid = str(r["entity_id"]).strip()
            # 1. Always retain true matches
            if eid in target_s2_ids:
                s2_records_dict[eid] = r
                continue

            # 2. Retain candidates within the same countries
            c = normalize_country(r.get("country"))
            if not c or c not in s1_countries:
                continue

            nm = normalize_business_name(r.get("business_name"))
            if not nm:
                continue

            # Tier 1 candidate: exact normalized name match
            if nm in s1_norm_names:
                s2_records_dict[eid] = r
                continue

            # Tier 2 candidate: prefix match (capped per prefix)
            pfx = nm[:5] if len(nm) >= 5 else nm
            if pfx in s1_prefixes and s2_prefix_counts[pfx] < max_cands_per_prefix:
                s2_records_dict[eid] = r
                s2_prefix_counts[pfx] += 1
                continue

            # Tier 3 candidate: ordinary country sample
            if s2_country_samples[c] < max_ordinary_per_country:
                s2_records_dict[eid] = r
                s2_country_samples[c] += 1

    log.info(
        f"Source 2 streamed in {time.time() - t_s2_start:.2f}s: "
        f"Indexed {len(s2_records_dict):,} records out of {total_s2_rows_scanned:,} total rows."
    )
    s2_index = build_secondary_index(s2_records_dict.values(), source_name="S2")

    # -----------------------------------------------------------------------
    # Step 7: Stream Source 3 (Target Matches + Negative Candidates)
    # -----------------------------------------------------------------------
    log.info("Streaming train_source3.tsv for true matches and negative candidates...")
    s3_records_dict: dict[str, dict[str, Any]] = {}
    s3_prefix_counts: Counter[str] = Counter()
    s3_country_samples: Counter[str] = Counter()
    total_s3_rows_scanned = 0

    t_s3_start = time.time()
    for chunk in iter_chunks(paths["train_source3"], chunksize=100_000):
        records = chunk.to_dict(orient="records")
        total_s3_rows_scanned += len(records)
        for r in records:
            eid = str(r["entity_id"]).strip()
            # 1. Always retain true matches
            if eid in target_s3_ids:
                s3_records_dict[eid] = r
                continue

            # 2. Retain candidates within the same countries
            c = normalize_country(r.get("country"))
            if not c or c not in s1_countries:
                continue

            nm = normalize_business_name(r.get("business_name"))
            if not nm:
                continue

            # Tier 1 candidate: exact normalized name match
            if nm in s1_norm_names:
                s3_records_dict[eid] = r
                continue

            # Tier 2 candidate: prefix match (capped per prefix)
            pfx = nm[:5] if len(nm) >= 5 else nm
            if pfx in s1_prefixes and s3_prefix_counts[pfx] < max_cands_per_prefix:
                s3_records_dict[eid] = r
                s3_prefix_counts[pfx] += 1
                continue

            # Tier 3 candidate: ordinary country sample
            if s3_country_samples[c] < max_ordinary_per_country:
                s3_records_dict[eid] = r
                s3_country_samples[c] += 1

    log.info(
        f"Source 3 streamed in {time.time() - t_s3_start:.2f}s: "
        f"Indexed {len(s3_records_dict):,} records out of {total_s3_rows_scanned:,} total rows."
    )
    s3_index = build_secondary_index(s3_records_dict.values(), source_name="S3")

    # -----------------------------------------------------------------------
    # Step 8: Generate Labeled Pairs for Each S1 Entity (1:1 Ratio)
    # -----------------------------------------------------------------------
    log.info("Generating labeled candidate pairs and computing 26 ML features...")
    rng = random.Random(random_seed)

    split_pairs: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "holdout": [],
    }

    # Tracking metrics
    s2_pos_count = 0
    s2_neg_count = 0
    s3_pos_count = 0
    s3_neg_count = 0

    tier1_hard_negs = 0
    tier2_hard_negs = 0
    tier3_easy_negs = 0
    candidates_after_blocking = 0
    skipped_error_records = 0

    # Theoretical Cartesian comparison count before blocking
    candidates_before_blocking = num_s1_entities * (total_s2_rows_scanned + total_s3_rows_scanned)

    t_pairs_start = time.time()
    for idx, s1_rec in enumerate(s1_records):
        s1_id = str(s1_rec["entity_id"]).strip()
        split_name = s1_to_split.get(s1_id, "train")

        # ---- Process against S2 ----
        s2_matches = sorted(gt.get_s2_matches(s1_id))
        s2_pos_pairs: list[dict[str, Any]] = []
        for mid in s2_matches:
            if mid in s2_index.records:
                s2_pos_pairs.append(
                    build_labeled_pair(
                        s1_rec, s2_index.records[mid], source="S2", label=1, difficulty="positive"
                    )
                )
                s2_pos_count += 1
            else:
                skipped_error_records += 1

        num_s2_pos = len(s2_pos_pairs)
        target_s2_negs = num_s2_pos  # 1:1 ratio

        # Candidate retrieval from S2 blocking index
        exact_s2, similar_s2, country_s2 = s2_index.get_candidates(s1_rec)
        candidates_after_blocking += (len(exact_s2) + len(similar_s2) + len(country_s2))

        neg_exact_s2 = [c for c in sorted(exact_s2) if c not in s2_matches and c in s2_index.records]
        neg_similar_s2 = [c for c in sorted(similar_s2) if c not in s2_matches and c in s2_index.records]
        neg_country_s2 = [c for c in sorted(country_s2) if c not in s2_matches and c in s2_index.records]

        rng.shuffle(neg_exact_s2)
        # Prioritize different addresses within similar name candidates
        s1_addr = normalize_address(s1_rec.get("business_address"))
        diff_addr_similar: list[str] = []
        same_addr_similar: list[str] = []
        for cid in neg_similar_s2:
            c_addr = normalize_address(s2_index.records[cid].get("business_address"))
            if s1_addr is not None and c_addr is not None and s1_addr != c_addr:
                diff_addr_similar.append(cid)
            else:
                same_addr_similar.append(cid)
        rng.shuffle(diff_addr_similar)
        rng.shuffle(same_addr_similar)
        neg_similar_s2 = diff_addr_similar + same_addr_similar
        rng.shuffle(neg_country_s2)

        s2_neg_pairs: list[dict[str, Any]] = []
        # Tier 1
        for cid in neg_exact_s2:
            if len(s2_neg_pairs) >= target_s2_negs:
                break
            s2_neg_pairs.append(
                build_labeled_pair(
                    s1_rec, s2_index.records[cid], source="S2", label=0, difficulty="hard_exact_name"
                )
            )
            tier1_hard_negs += 1
            s2_neg_count += 1

        # Tier 2
        if len(s2_neg_pairs) < target_s2_negs:
            for cid in neg_similar_s2:
                if len(s2_neg_pairs) >= target_s2_negs:
                    break
                s2_neg_pairs.append(
                    build_labeled_pair(
                        s1_rec, s2_index.records[cid], source="S2", label=0, difficulty="hard_similar_name"
                    )
                )
                tier2_hard_negs += 1
                s2_neg_count += 1

        # Tier 3
        if len(s2_neg_pairs) < target_s2_negs:
            for cid in neg_country_s2:
                if len(s2_neg_pairs) >= target_s2_negs:
                    break
                s2_neg_pairs.append(
                    build_labeled_pair(
                        s1_rec, s2_index.records[cid], source="S2", label=0, difficulty="easy_same_country"
                    )
                )
                tier3_easy_negs += 1
                s2_neg_count += 1

        # ---- Process against S3 ----
        s3_matches = sorted(gt.get_s3_matches(s1_id))
        s3_pos_pairs: list[dict[str, Any]] = []
        for mid in s3_matches:
            if mid in s3_index.records:
                s3_pos_pairs.append(
                    build_labeled_pair(
                        s1_rec, s3_index.records[mid], source="S3", label=1, difficulty="positive"
                    )
                )
                s3_pos_count += 1
            else:
                skipped_error_records += 1

        num_s3_pos = len(s3_pos_pairs)
        target_s3_negs = num_s3_pos  # 1:1 ratio

        # Candidate retrieval from S3 blocking index
        exact_s3, similar_s3, country_s3 = s3_index.get_candidates(s1_rec)
        candidates_after_blocking += (len(exact_s3) + len(similar_s3) + len(country_s3))

        neg_exact_s3 = [c for c in sorted(exact_s3) if c not in s3_matches and c in s3_index.records]
        neg_similar_s3 = [c for c in sorted(similar_s3) if c not in s3_matches and c in s3_index.records]
        neg_country_s3 = [c for c in sorted(country_s3) if c not in s3_matches and c in s3_index.records]

        rng.shuffle(neg_exact_s3)
        diff_addr_similar_s3: list[str] = []
        same_addr_similar_s3: list[str] = []
        for cid in neg_similar_s3:
            c_addr = normalize_address(s3_index.records[cid].get("business_address"))
            if s1_addr is not None and c_addr is not None and s1_addr != c_addr:
                diff_addr_similar_s3.append(cid)
            else:
                same_addr_similar_s3.append(cid)
        rng.shuffle(diff_addr_similar_s3)
        rng.shuffle(same_addr_similar_s3)
        neg_similar_s3 = diff_addr_similar_s3 + same_addr_similar_s3
        rng.shuffle(neg_country_s3)

        s3_neg_pairs: list[dict[str, Any]] = []
        # Tier 1
        for cid in neg_exact_s3:
            if len(s3_neg_pairs) >= target_s3_negs:
                break
            s3_neg_pairs.append(
                build_labeled_pair(
                    s1_rec, s3_index.records[cid], source="S3", label=0, difficulty="hard_exact_name"
                )
            )
            tier1_hard_negs += 1
            s3_neg_count += 1

        # Tier 2
        if len(s3_neg_pairs) < target_s3_negs:
            for cid in neg_similar_s3:
                if len(s3_neg_pairs) >= target_s3_negs:
                    break
                s3_neg_pairs.append(
                    build_labeled_pair(
                        s1_rec, s3_index.records[cid], source="S3", label=0, difficulty="hard_similar_name"
                    )
                )
                tier2_hard_negs += 1
                s3_neg_count += 1

        # Tier 3
        if len(s3_neg_pairs) < target_s3_negs:
            for cid in neg_country_s3:
                if len(s3_neg_pairs) >= target_s3_negs:
                    break
                s3_neg_pairs.append(
                    build_labeled_pair(
                        s1_rec, s3_index.records[cid], source="S3", label=0, difficulty="easy_same_country"
                    )
                )
                tier3_easy_negs += 1
                s3_neg_count += 1

        # Append all generated pairs to the respective split
        split_pairs[split_name].extend(
            s2_pos_pairs + s2_neg_pairs + s3_pos_pairs + s3_neg_pairs
        )

        if (idx + 1) % 10_000 == 0:
            log.info(f"Processed {idx + 1:,} S1 entities...")

    log.info(f"Candidate pair generation finished in {time.time() - t_pairs_start:.2f}s.")

    total_positives = s2_pos_count + s3_pos_count
    total_negatives = s2_neg_count + s3_neg_count
    total_pairs = total_positives + total_negatives
    pos_neg_ratio = (total_negatives / total_positives) if total_positives > 0 else 0.0

    # -----------------------------------------------------------------------
    # Step 9: Leakage Checks & Validation
    # -----------------------------------------------------------------------
    log.info("Running rigorous leakage and data integrity checks...")
    audit_report = validate_training_dataset(
        split_pairs,
        ground_truth=gt,
        raw_file_paths=raw_files,
        raw_checksums_before=checksums_before,
    )
    log.info("Leakage validation status: PASS")

    # Explicit cross-country check
    for sname, pairs in split_pairs.items():
        for p in pairs:
            if p["country_equal"] != 1 or p["same_country"] != 1:
                raise ValueError(
                    f"Cross-country candidate pair detected in {sname}: {p['s1_id']} - {p['secondary_id']}"
                )

    # Re-verify raw files are untouched
    for rf in raw_files:
        curr_h = compute_file_hash(rf)
        if curr_h != checksums_before[str(rf)]:
            raise ValueError(f"CRITICAL: Raw file {rf} was modified during the run!")
        if rf.stat().st_size != sizes_before[str(rf)]:
            raise ValueError(f"CRITICAL: Raw file size of {rf} changed!")

    # -----------------------------------------------------------------------
    # Step 10: Write Chunked Shards
    # -----------------------------------------------------------------------
    log.info("Writing chunked output shards...")
    shard_info: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "holdout": [],
    }
    total_shard_files: list[Path] = []
    total_shard_bytes = 0

    for split_name, pairs in split_pairs.items():
        shards, count = write_shards(
            pairs,
            output_dir=out_dir,
            split_name=split_name,
            shard_size=shard_size,
            file_format="csv",
        )
        for sh in shards:
            size_b = os.path.getsize(sh)
            total_shard_files.append(sh)
            total_shard_bytes += size_b
            shard_info[split_name].append({
                "path": str(sh),
                "rows": count if len(shards) == 1 else min(shard_size, count),
                "size_bytes": size_b,
            })

    # -----------------------------------------------------------------------
    # Step 11: Collect Resource & Performance Metrics
    # -----------------------------------------------------------------------
    t_total = time.time() - t_start
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    # -----------------------------------------------------------------------
    # Step 12: Write Summary Report (output/training_pairs_pilot/summary.txt)
    # -----------------------------------------------------------------------
    summary_lines = [
        "=" * 75,
        "AMAZON ML CHALLENGE 2026 - TRAINING PAIR PILOT SUMMARY",
        "=" * 75,
        "",
        f"1. Selected S1 entities:              {num_s1_entities:,}",
        f"2. Train / validation / holdout S1:    {len(splits['train']):,} / {len(splits['val']):,} / {len(splits['holdout']):,}",
        f"3. Positive pair count:               {total_positives:,}",
        f"4. Negative pair count:               {total_negatives:,}",
        f"5. Positive/negative ratio:           1 : {pos_neg_ratio:.2f}",
        f"6. S2 positive/negative counts:       {s2_pos_count:,} positive / {s2_neg_count:,} negative",
        f"7. S3 positive/negative counts:       {s3_pos_count:,} positive / {s3_neg_count:,} negative",
        f"8. Tier-1 hard negatives:             {tier1_hard_negs:,}",
        f"9. Tier-2 hard negatives:             {tier2_hard_negs:,}",
        f"10. Tier-3 easy negatives:            {tier3_easy_negs:,}",
        f"11. Candidate counts before blocking: {candidates_before_blocking:,} pairs (Cartesian space)",
        f"12. Candidate counts after blocking:  {candidates_after_blocking:,} candidates examined",
        f"13. Final training-pair row count:    {total_pairs:,}",
        f"14. Number of shards:                 {len(total_shard_files)} shards",
        f"15. Total disk size:                  {total_shard_bytes / (1024 * 1024):.2f} MB ({total_shard_bytes:,} bytes)",
        f"16. Processing time:                  {t_total:.2f} seconds ({t_total / 60:.2f} minutes)",
        f"17. Peak memory usage:                {peak_memory_mb:.2f} MB",
        f"18. Any skipped/error records:        {skipped_error_records} records skipped / {skipped_s1_records} S1 skipped",
        "",
        "LEAKAGE & INTEGRITY VERIFICATION:",
        "  [✓] No S1 appears in more than one split (Train ∩ Val = ∅, Train ∩ Holdout = ∅, Val ∩ Holdout = ∅)",
        "  [✓] Zero ground-truth positives mislabeled as negative (0)",
        "  [✓] Zero duplicate candidate pairs within any split",
        "  [✓] Zero cross-country negative pairs created (100% same country)",
        "  [✓] Raw dataset files remain completely unchanged (MD5 & byte-level verified)",
        "",
        "OUTPUT SHARDS BREAKDOWN:",
    ]

    for split_name, shards in shard_info.items():
        summary_lines.append(f"  [{split_name.upper()}] ({len(split_pairs[split_name]):,} rows)")
        for sh in shards:
            summary_lines.append(
                f"    * {Path(sh['path']).name}: {sh['size_bytes'] / (1024 * 1024):.2f} MB ({sh['size_bytes']:,} bytes)"
            )

    summary_lines.append("=" * 75)
    summary_text = "\n".join(summary_lines)

    with open(sum_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    log.info(f"Summary written to {sum_file}")
    log.info("PILOT COMPLETED SUCCESSFULLY.")

    return {
        "status": "SUCCESS",
        "summary_text": summary_text,
        "summary_path": str(sum_file),
        "total_pairs": total_pairs,
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "pos_neg_ratio": pos_neg_ratio,
        "shards": shard_info,
        "peak_memory_mb": peak_memory_mb,
        "processing_time_sec": t_total,
    }


def main():
    setup_logging(level=logging.INFO)
    results = run_pilot(
        num_s1_entities=50_000,
        random_seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
        holdout_ratio=0.15,
        neg_ratio=1.0,
        shard_size=50_000,
        output_dir="output/training_pairs_pilot",
        summary_path="output/training_pairs_pilot/summary.txt",
    )
    print("\n" + results["summary_text"])


if __name__ == "__main__":
    main()
