"""
tests/test_training_pairs.py
============================
Comprehensive unit tests for src/training_pairs.py.

Uses tiny synthetic datasets (< 25 records).
Does NOT touch or process any large production dataset files.
"""

import csv
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import feature_names
from src.training_pairs import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TRAINING_SHARD_COLUMNS,
    CandidateGenerationStats,
    GroundTruthIndex,
    SecondaryBlockIndex,
    build_labeled_pair,
    build_secondary_index,
    compute_file_hash,
    generate_candidate_pairs_for_s1,
    has_parquet_support,
    parse_ratio,
    run_training_pair_pipeline,
    split_s1_entities,
    validate_training_dataset,
    write_shards,
    write_summary,
)


def _make_rec(eid: str, name: str, country: str, address: str = "") -> dict:
    return {
        "entity_id": eid,
        "business_name": name,
        "business_address": address or None,
        "country": country,
    }


# ===========================================================================
# 1. Ground Truth Index Tests
# ===========================================================================

class TestGroundTruthIndex(unittest.TestCase):
    def test_from_dict_and_membership(self):
        gt = GroundTruthIndex.from_dict({
            "S1-001": ["S2-001", "S3-001"],
            "S1-002": ["S2-002"],
            "S1-003": [],  # Singleton
        })

        self.assertEqual(gt.s1_count, 3)
        self.assertEqual(gt.s2_match_count, 2)
        self.assertEqual(gt.s3_match_count, 1)
        self.assertEqual(gt.total_match_count, 3)

        self.assertTrue(gt.is_match("S1-001", "S2-001"))
        self.assertTrue(gt.is_match("S1-001", "S3-001"))
        self.assertTrue(gt.is_match("S1-002", "S2-002"))

        self.assertFalse(gt.is_match("S1-001", "S2-999"))
        self.assertFalse(gt.is_match("S1-003", "S2-001"))
        self.assertFalse(gt.is_match("S1-999", "S2-001"))

        self.assertEqual(gt.get_s2_matches("S1-001"), {"S2-001"})
        self.assertEqual(gt.get_s3_matches("S1-001"), {"S3-001"})
        self.assertEqual(gt.get_all_matches("S1-001"), {"S2-001", "S3-001"})
        self.assertEqual(gt.get_all_matches("S1-003"), set())

    def test_from_file_streaming(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-100\tS2-100,S3-100\n")
            f.write("S1-200\t\n")  # Singleton
            f.write("S1-300\tS2-300\n")
            tmp_path = f.name

        try:
            gt = GroundTruthIndex.from_file(tmp_path)
            self.assertEqual(gt.s1_count, 3)
            self.assertEqual(gt.total_match_count, 3)
            self.assertTrue(gt.is_match("S1-100", "S2-100"))
            self.assertTrue(gt.is_match("S1-100", "S3-100"))
            self.assertTrue(gt.is_match("S1-300", "S2-300"))
            self.assertFalse(gt.is_match("S1-200", "S2-100"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_from_pairs(self):
        pairs = [("S1-01", "S2-01"), ("S1-01", "S3-01"), ("S1-02", "S2-02")]
        gt = GroundTruthIndex.from_pairs(pairs)
        self.assertEqual(gt.s1_count, 2)
        self.assertEqual(gt.total_match_count, 3)
        self.assertTrue(gt.is_match("S1-01", "S2-01"))


# ===========================================================================
# 2. Entity Splitting Tests (Grouped by S1, Zero Leakage)
# ===========================================================================

class TestEntitySplitting(unittest.TestCase):
    def test_split_proportions_and_disjointness(self):
        s1_ids = [f"S1-{i:04d}" for i in range(100)]
        splits = split_s1_entities(s1_ids, train_ratio=0.70, val_ratio=0.15, holdout_ratio=0.15, random_seed=42)

        train_set = splits["train"]
        val_set = splits["val"]
        holdout_set = splits["holdout"]

        # No overlap between any splits
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & holdout_set), 0)
        self.assertEqual(len(val_set & holdout_set), 0)

        # Total count matches
        self.assertEqual(len(train_set) + len(val_set) + len(holdout_set), 100)
        self.assertEqual(len(train_set), 70)
        self.assertEqual(len(val_set), 15)
        self.assertEqual(len(holdout_set), 15)

    def test_split_deterministic_reproducibility(self):
        s1_ids = [f"S1-{i:03d}" for i in range(50)]
        split1 = split_s1_entities(s1_ids, random_seed=123)
        split2 = split_s1_entities(s1_ids, random_seed=123)
        split3 = split_s1_entities(s1_ids, random_seed=999)

        self.assertEqual(split1["train"], split2["train"])
        self.assertEqual(split1["val"], split2["val"])
        self.assertEqual(split1["holdout"], split2["holdout"])

        # Different seed produces different partition
        self.assertNotEqual(split1["train"], split3["train"])

    def test_invalid_split_ratios(self):
        with self.assertRaises(ValueError):
            split_s1_entities(["S1-01"], train_ratio=0.8, val_ratio=0.3, holdout_ratio=0.1)

        with self.assertRaises(ValueError):
            split_s1_entities(["S1-01"], train_ratio=-0.1, val_ratio=0.6, holdout_ratio=0.5)


# ===========================================================================
# 3. Secondary Blocking & Candidate Difficulty Tier Tests
# ===========================================================================

class TestSecondaryBlocking(unittest.TestCase):
    def setUp(self):
        self.s2_records = [
            _make_rec("S2-001", "Acme Corporation", "US", "100 Main St"),
            _make_rec("S2-002", "Acme Corp", "US", "200 Market St"),
            _make_rec("S2-003", "Acme Logistics LLC", "US", "300 Industrial Rd"),
            _make_rec("S2-004", "Omega Solutions", "US", "400 Oak Ave"),
            _make_rec("S2-005", "Acme Corp", "IN", "Mumbai Highway"),  # Different country!
        ]
        self.s2_index = build_secondary_index(self.s2_records, source_name="S2")

    def test_candidate_difficulty_tiers(self):
        s1_rec = _make_rec("S1-001", "Acme Corp", "US", "100 Main St")
        exact, similar, ordinary = self.s2_index.get_candidates(s1_rec)

        # S2-001 & S2-002 share normalized name "acme" in US (exact match tier)
        self.assertIn("S2-002", exact)
        self.assertIn("S2-001", exact)

        # S2-003 shares prefix/token "acme" in US (similar name tier)
        self.assertIn("S2-003", similar)

        # S2-004 is in US but completely different name (ordinary country tier)
        self.assertIn("S2-004", ordinary)

        # S2-005 is IN (different country) -> must NEVER appear anywhere
        self.assertNotIn("S2-005", exact)
        self.assertNotIn("S2-005", similar)
        self.assertNotIn("S2-005", ordinary)


# ===========================================================================
# 4. Positive and Hard Negative Generation Tests
# ===========================================================================

class TestPositiveAndHardNegativeGeneration(unittest.TestCase):
    def setUp(self):
        # S1 query
        self.s1_rec = _make_rec("S1-001", "Acme Tools LLC", "US", "100 Main St")

        # S2 candidates
        self.s2_records = [
            # True match:
            _make_rec("S2-TRUE", "Acme Tools Inc", "US", "100 Main St"),
            # Hard negative tier 1 (exact normalized name, but not match):
            _make_rec("S2-HARD1", "Acme Tools L.L.C.", "US", "999 Other Blvd"),
            # Hard negative tier 2 (similar name prefix/token, different address):
            _make_rec("S2-HARD2", "Acme Tooling Solutions", "US", "888 South Way"),
            # Easy negative tier 3 (ordinary same country):
            _make_rec("S2-EASY", "Beta Systems", "US", "777 West Ave"),
            # Cross-country record (should never be selected):
            _make_rec("S2-CROSS", "Acme Tools", "IN", "Delhi"),
        ]
        self.s2_index = build_secondary_index(self.s2_records, source_name="S2")
        self.gt = GroundTruthIndex.from_dict({"S1-001": ["S2-TRUE"]})

    def test_positive_extraction(self):
        pairs = generate_candidate_pairs_for_s1(
            self.s1_rec, self.s2_index, self.gt, neg_ratio=1.0
        )
        positives = [p for p in pairs if p[3] == 1]
        self.assertEqual(len(positives), 1)
        self.assertEqual(positives[0][1]["entity_id"], "S2-TRUE")
        self.assertEqual(positives[0][4], "positive")

    def test_hard_negative_prioritization(self):
        import random
        rng = random.Random(42)
        # Request 1 negative (1:1 ratio)
        pairs = generate_candidate_pairs_for_s1(
            self.s1_rec, self.s2_index, self.gt, neg_ratio=1.0, rng=rng
        )
        negatives = [p for p in pairs if p[3] == 0]
        self.assertEqual(len(negatives), 1)
        # Tier 1 hard negative should be chosen first
        self.assertEqual(negatives[0][1]["entity_id"], "S2-HARD1")
        self.assertEqual(negatives[0][4], "hard_exact_name")

    def test_no_cross_country_negatives(self):
        # Even with high negative ratio, S2-CROSS (India) must NEVER be chosen
        pairs = generate_candidate_pairs_for_s1(
            self.s1_rec, self.s2_index, self.gt, neg_ratio=10.0
        )
        selected_ids = {p[1]["entity_id"] for p in pairs}
        self.assertNotIn("S2-CROSS", selected_ids)


# ===========================================================================
# 5. Class Balancing Tests
# ===========================================================================

class TestClassBalancing(unittest.TestCase):
    def test_ratio_parsing(self):
        self.assertAlmostEqual(parse_ratio("1:1"), 1.0)
        self.assertAlmostEqual(parse_ratio("1:2"), 2.0)
        self.assertAlmostEqual(parse_ratio("1:3"), 3.0)
        self.assertAlmostEqual(parse_ratio(2.5), 2.5)

        with self.assertRaises(ValueError):
            parse_ratio("invalid")
        with self.assertRaises(ValueError):
            parse_ratio(-1)

    def test_configurable_ratios(self):
        s1 = _make_rec("S1-A", "Delta Corp", "US", "101 Pine")
        s2_list = [
            _make_rec("S2-TRUE", "Delta Corp", "US", "101 Pine"),
            _make_rec("S2-NEG1", "Delta Corp", "US", "202 Elm"),
            _make_rec("S2-NEG2", "Delta Services", "US", "303 Maple"),
            _make_rec("S2-NEG3", "Delta Global", "US", "404 Cedar"),
            _make_rec("S2-NEG4", "Delta Tech", "US", "505 Birch"),
        ]
        s2_idx = build_secondary_index(s2_list, source_name="S2")
        gt = GroundTruthIndex.from_dict({"S1-A": ["S2-TRUE"]})

        # Ratio 1:1 -> 1 pos, 1 neg
        p1 = generate_candidate_pairs_for_s1(s1, s2_idx, gt, neg_ratio=1.0)
        self.assertEqual(len([p for p in p1 if p[3] == 1]), 1)
        self.assertEqual(len([p for p in p1 if p[3] == 0]), 1)

        # Ratio 1:2 -> 1 pos, 2 neg
        p2 = generate_candidate_pairs_for_s1(s1, s2_idx, gt, neg_ratio=2.0)
        self.assertEqual(len([p for p in p2 if p[3] == 1]), 1)
        self.assertEqual(len([p for p in p2 if p[3] == 0]), 2)

        # Ratio 1:3 -> 1 pos, 3 neg
        p3 = generate_candidate_pairs_for_s1(s1, s2_idx, gt, neg_ratio=3.0)
        self.assertEqual(len([p for p in p3 if p[3] == 1]), 1)
        self.assertEqual(len([p for p in p3 if p[3] == 0]), 3)


# ===========================================================================
# 6. Feature Extraction & Shard Schema Tests
# ===========================================================================

class TestFeatureExtractionAndSchema(unittest.TestCase):
    def test_schema_conformance(self):
        s1 = _make_rec("S1-1", "Alpha Inc", "US", "1st Street")
        s2 = _make_rec("S2-1", "Alpha LLC", "US", "1st St")

        row = build_labeled_pair(s1, s2, source="S2", label=1, difficulty="positive")

        # Check metadata columns
        self.assertEqual(row["s1_id"], "S1-1")
        self.assertEqual(row["secondary_id"], "S2-1")
        self.assertEqual(row["source"], "S2")
        self.assertEqual(row["label"], 1)

        # Check that all 26 feature columns are present
        for fname in FEATURE_COLUMNS:
            self.assertIn(fname, row)
            self.assertIsInstance(row[fname], (int, float))

        # Check total column count
        self.assertEqual(len(row), len(TRAINING_SHARD_COLUMNS))
        self.assertEqual(len(TRAINING_SHARD_COLUMNS), 30)

        # IDs must NOT be part of FEATURE_COLUMNS
        self.assertNotIn("s1_id", FEATURE_COLUMNS)
        self.assertNotIn("secondary_id", FEATURE_COLUMNS)
        self.assertNotIn("source", FEATURE_COLUMNS)
        self.assertNotIn("label", FEATURE_COLUMNS)


# ===========================================================================
# 7. Shard Writing Tests
# ===========================================================================

class TestShardWriting(unittest.TestCase):
    def test_write_multiple_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_pairs = []
            for i in range(12):
                s1 = _make_rec(f"S1-{i}", f"Company {i}", "US", f"{i} Street")
                s2 = _make_rec(f"S2-{i}", f"Company {i}", "US", f"{i} St")
                sample_pairs.append(build_labeled_pair(s1, s2, source="S2", label=1))

            # Shard size = 5 -> should produce 3 shards: 5, 5, 2
            shards, count = write_shards(
                sample_pairs,
                output_dir=tmpdir,
                split_name="train",
                shard_size=5,
                file_format="csv",
            )

            self.assertEqual(count, 12)
            self.assertEqual(len(shards), 3)
            self.assertTrue(all(p.exists() for p in shards))
            self.assertTrue(shards[0].name.endswith("train_part_000.csv"))
            self.assertTrue(shards[1].name.endswith("train_part_001.csv"))
            self.assertTrue(shards[2].name.endswith("train_part_002.csv"))


# ===========================================================================
# 8. Leakage Checks and Integrity Validation Tests
# ===========================================================================

class TestLeakageAndIntegrityChecks(unittest.TestCase):
    def setUp(self):
        self.gt = GroundTruthIndex.from_dict({
            "S1-1": ["S2-1"],
            "S1-2": ["S2-2"],
        })

    def test_valid_dataset_passes(self):
        splits_data = {
            "train": [
                {"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S2", "label": 1},
                {"s1_id": "S1-1", "secondary_id": "S2-9", "source": "S2", "label": 0},
            ],
            "val": [
                {"s1_id": "S1-2", "secondary_id": "S2-2", "source": "S2", "label": 1},
            ],
        }
        report = validate_training_dataset(splits_data, self.gt)
        self.assertEqual(report["status"], "PASS")

    def test_s1_entity_leakage_detected(self):
        # S1-1 is in both train and val
        splits_data = {
            "train": [{"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S2", "label": 1}],
            "val": [{"s1_id": "S1-1", "secondary_id": "S2-2", "source": "S2", "label": 0}],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_training_dataset(splits_data, self.gt)
        self.assertIn("Entity leakage detected", str(ctx.exception))

    def test_mislabeled_positive_detected(self):
        # (S1-1, S2-1) is a true match in GT, but labeled as 0!
        splits_data = {
            "train": [{"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S2", "label": 0}],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_training_dataset(splits_data, self.gt)
        self.assertIn("mislabeled as negative", str(ctx.exception))

    def test_duplicate_candidate_pair_detected(self):
        splits_data = {
            "train": [
                {"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S2", "label": 1},
                {"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S2", "label": 1},
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_training_dataset(splits_data, self.gt)
        self.assertIn("Duplicate pair detected", str(ctx.exception))

    def test_source_mismatch_detected(self):
        splits_data = {
            "train": [
                # S2-1 entity tagged with source="S3"
                {"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S3", "label": 1},
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_training_dataset(splits_data, self.gt)
        self.assertIn("Source mismatch", str(ctx.exception))

    def test_raw_data_untouched_check(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("test raw content")
            raw_path = f.name

        try:
            h_before = compute_file_hash(raw_path)
            splits_data = {
                "train": [{"s1_id": "S1-1", "secondary_id": "S2-1", "source": "S2", "label": 1}]
            }
            # Verify clean check
            res = validate_training_dataset(
                splits_data,
                self.gt,
                raw_file_paths=[raw_path],
                raw_checksums_before={raw_path: h_before},
            )
            self.assertEqual(res["raw_data_check"], "Verified untouched")

            # Simulate modification
            with open(raw_path, "a") as f:
                f.write("tampered!")

            with self.assertRaises(ValueError) as ctx:
                validate_training_dataset(
                    splits_data,
                    self.gt,
                    raw_file_paths=[raw_path],
                    raw_checksums_before={raw_path: h_before},
                )
            self.assertIn("modified", str(ctx.exception))
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)


# ===========================================================================
# 9. End-to-End Synthetic Pipeline Tests
# ===========================================================================

class TestEndToEndPipeline(unittest.TestCase):
    def test_synthetic_run_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output" / "training_pairs"
            sum_path = Path(tmpdir) / "output" / "training_pairs_summary.txt"

            s1_records = [
                _make_rec("S1-01", "Apex Global", "US", "100 Broadway"),
                _make_rec("S1-02", "Nova Corp", "US", "200 Fifth Ave"),
                _make_rec("S1-03", "Apex Logistics", "US", "300 Madison"),
                _make_rec("S1-04", "Solaris Tech", "IN", "MG Road"),
            ]
            s2_records = [
                _make_rec("S2-01", "Apex Global LLC", "US", "100 Broadway"),
                _make_rec("S2-02", "Nova Corp", "US", "200 Fifth Ave"),
                _make_rec("S2-03", "Apex Global Inc", "US", "999 Other St"),  # Hard negative for S1-01
                _make_rec("S2-04", "Solaris Tech Ltd", "IN", "MG Road"),
            ]
            s3_records = [
                _make_rec("S3-01", "Apex Global Inc", "US", "100 Broadway"),
                _make_rec("S3-02", "Nova Corporation", "US", "200 5th Ave"),
            ]
            gt = GroundTruthIndex.from_dict({
                "S1-01": ["S2-01", "S3-01"],
                "S1-02": ["S2-02", "S3-02"],
                "S1-03": [],  # Singleton
                "S1-04": ["S2-04"],
            })

            result = run_training_pair_pipeline(
                s1_records=s1_records,
                s2_records=s2_records,
                s3_records=s3_records,
                ground_truth=gt,
                output_dir=out_dir,
                summary_path=sum_path,
                train_ratio=0.50,
                val_ratio=0.25,
                holdout_ratio=0.25,
                neg_ratio="1:2",
                random_seed=42,
                shard_size=10,
                file_format="csv",
            )

            self.assertEqual(result["status"], "SUCCESS")
            self.assertTrue(sum_path.exists())

            # Read summary file to verify required content
            with open(sum_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("TRAINING PAIR GENERATION SUMMARY", content)
            self.assertIn("Positive pairs total:", content)
            self.assertIn("Negative pairs total:", content)
            self.assertIn("Positive/Negative ratio:", content)
            self.assertIn("S1-S2 positive matches:", content)
            self.assertIn("S1-S3 positive matches:", content)
            self.assertIn("Total hard negatives:", content)
            self.assertIn("Total easy negatives", content)
            self.assertIn("MODEL FEATURE SCHEMA (26 FEATURES):", content)
            self.assertIn("OUTPUT SHARDS & SIZES:", content)


if __name__ == "__main__":
    unittest.main()
