"""
tests/test_blocking.py
======================
Unit tests for src/blocking.py.

All tests use small synthetic datasets (< 30 records).
No real dataset files are touched.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blocking import (
    build_country_index,
    build_name_index,
    build_prefix_index,
    build_token_index,
    generate_candidate_pairs,
    get_country_candidates,
    get_name_candidates,
    get_prefix_candidates,
    get_token_candidates,
    pairs_from_index,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_record(eid: str, name: str, country: str, address: str = "") -> dict:
    return {
        "entity_id": eid,
        "business_name": name,
        "business_address": address or None,
        "country": country,
    }


# Small synthetic dataset: 3 US, 2 IN, 1 FR
SYNTHETIC_RECORDS = [
    _make_record("S1-001", "Acme Corporation",       "US", "123 Main St"),
    _make_record("S2-001", "Acme Corp",               "US", "123 Main Street"),
    _make_record("S1-002", "Global Tech LLC",         "US", "456 Park Ave"),
    _make_record("S2-002", "Reliance Industries Ltd", "IN", "Mumbai"),
    _make_record("S3-001", "Reliance Industries Ltd", "IN", "Mumbai"),
    _make_record("S1-003", "Société Générale",        "FR", "Paris"),
]


# ===========================================================================
# A. Country blocking
# ===========================================================================


class TestCountryBlocking(unittest.TestCase):
    def setUp(self):
        self.index = build_country_index(SYNTHETIC_RECORDS)

    def test_index_has_expected_keys(self):
        self.assertIn("US", self.index)
        self.assertIn("IN", self.index)
        self.assertIn("FR", self.index)

    def test_us_has_three_entities(self):
        self.assertEqual(len(self.index["US"]), 3)

    def test_india_has_two_entities(self):
        self.assertEqual(len(self.index["IN"]), 2)

    def test_fr_has_one_entity(self):
        self.assertEqual(len(self.index["FR"]), 1)

    def test_candidates_exclude_self(self):
        rec = SYNTHETIC_RECORDS[0]  # S1-001 US
        candidates = get_country_candidates(self.index, rec)
        self.assertNotIn("S1-001", candidates)

    def test_us_entity_only_gets_us_candidates(self):
        rec = SYNTHETIC_RECORDS[0]  # S1-001 US
        candidates = get_country_candidates(self.index, rec)
        # Must contain other US entities.
        self.assertIn("S2-001", candidates)
        self.assertIn("S1-002", candidates)
        # Must NOT contain IN or FR entities.
        self.assertNotIn("S2-002", candidates)
        self.assertNotIn("S3-001", candidates)
        self.assertNotIn("S1-003", candidates)

    def test_cross_country_isolation(self):
        """US and India entities must never appear in each other's candidate sets."""
        us_rec = SYNTHETIC_RECORDS[0]   # S1-001 US
        in_rec = SYNTHETIC_RECORDS[3]   # S2-002 IN

        us_candidates = get_country_candidates(self.index, us_rec)
        in_candidates = get_country_candidates(self.index, in_rec)

        for us_eid in ["S1-001", "S2-001", "S1-002"]:
            self.assertNotIn(us_eid, in_candidates)
        for in_eid in ["S2-002", "S3-001"]:
            self.assertNotIn(in_eid, us_candidates)

    def test_missing_country_returns_empty(self):
        rec = {"entity_id": "S1-999", "country": None}
        candidates = get_country_candidates(self.index, rec)
        self.assertEqual(candidates, set())


# ===========================================================================
# B. Exact name blocking
# ===========================================================================


class TestExactNameBlocking(unittest.TestCase):
    def setUp(self):
        self.index = build_name_index(SYNTHETIC_RECORDS)

    def test_reliance_entities_share_block(self):
        """S2-002 and S3-001 both have 'Reliance Industries Ltd'."""
        rec = SYNTHETIC_RECORDS[3]  # S2-002
        candidates = get_name_candidates(self.index, rec)
        self.assertIn("S3-001", candidates)

    def test_acme_corp_canonicalized_to_same_key(self):
        """'Acme Corporation' and 'Acme Corp' both normalize to 'acme corp'."""
        # After normalization:
        # 'Acme Corporation' -> 'acme corp'  (corporation -> corp canonical)
        # 'Acme Corp'        -> 'acme corp'
        rec_a = SYNTHETIC_RECORDS[0]  # Acme Corporation
        rec_b = SYNTHETIC_RECORDS[1]  # Acme Corp
        cands_a = get_name_candidates(self.index, rec_a)
        self.assertIn("S2-001", cands_a, "Acme Corp should be a candidate for Acme Corporation")

    def test_unique_name_has_no_candidates(self):
        rec = SYNTHETIC_RECORDS[2]  # Global Tech LLC -- unique name
        candidates = get_name_candidates(self.index, rec)
        # S1-003 (Société Générale) should not match.
        self.assertNotIn("S1-003", candidates)

    def test_candidates_exclude_self(self):
        rec = SYNTHETIC_RECORDS[3]  # S2-002
        candidates = get_name_candidates(self.index, rec)
        self.assertNotIn("S2-002", candidates)

    def test_missing_name_returns_empty(self):
        rec = {"entity_id": "S1-999", "business_name": None, "country": "US"}
        candidates = get_name_candidates(self.index, rec)
        self.assertEqual(candidates, set())


# ===========================================================================
# C. Prefix blocking
# ===========================================================================


class TestPrefixBlocking(unittest.TestCase):
    def test_acme_shares_prefix_block(self):
        records = [
            _make_record("S1-001", "Acme Corporation", "US"),
            _make_record("S2-001", "Acme Corp",        "US"),
            _make_record("S1-002", "Global Tech LLC",  "US"),
        ]
        index = build_prefix_index(records, prefix_len=5)
        # "Acme " (first 5 chars of "acme corporation") vs "acme " (same)
        # Both normalize to start with "acme " -> same prefix block.
        rec = records[0]
        candidates = get_prefix_candidates(index, rec, prefix_len=5)
        self.assertIn("S2-001", candidates)
        self.assertNotIn("S1-002", candidates)

    def test_missing_name_returns_empty(self):
        records = [_make_record("S1-001", "Acme Corp", "US")]
        index = build_prefix_index(records, prefix_len=5)
        rec = {"entity_id": "S9-999", "business_name": None, "country": "US"}
        self.assertEqual(get_prefix_candidates(index, rec, prefix_len=5), set())


# ===========================================================================
# D. Token blocking
# ===========================================================================


class TestTokenBlocking(unittest.TestCase):
    def test_shared_token_creates_candidates(self):
        records = [
            _make_record("S1-001", "Global Tech Solutions", "US"),
            _make_record("S2-001", "Tech Innovations Inc",  "US"),
            _make_record("S1-002", "Unrelated Widgets",     "US"),
        ]
        index = build_token_index(records, min_token_len=4)
        rec = records[0]
        candidates = get_token_candidates(index, rec, min_token_len=4)
        # 'tech' is a shared token.
        self.assertIn("S2-001", candidates)
        # 'unrelated' / 'widgets' are not shared.
        self.assertNotIn("S1-002", candidates)

    def test_stopwords_excluded(self):
        records = [
            _make_record("S1-001", "The Best Company",   "US"),
            _make_record("S2-001", "The Good Company",   "US"),
            _make_record("S1-002", "An Average Company", "US"),
        ]
        # 'the', 'and', 'company' are stopwords by default.
        index = build_token_index(records, min_token_len=3)
        rec = records[0]  # "The Best Company"
        candidates = get_token_candidates(index, rec, min_token_len=3)
        # 'the' is a stopword; 'best' is unique – so no match via stopword.
        # 'company' is a stopword so should NOT create spurious matches.
        # S2-001 shares 'the' (stopword) but NOT 'best'.
        self.assertNotIn("S2-001", candidates)


# ===========================================================================
# E. generate_candidate_pairs (combined)
# ===========================================================================


class TestGenerateCandidatePairs(unittest.TestCase):
    """Integration tests for the combined blocking function."""

    def test_pairs_are_unique_and_ordered(self):
        pairs = list(generate_candidate_pairs(
            SYNTHETIC_RECORDS,
            use_country=True,
            use_exact_name=True,
        ))
        # No duplicates.
        self.assertEqual(len(pairs), len(set(pairs)))
        # All pairs are ordered (id_a < id_b).
        for a, b in pairs:
            self.assertLess(a, b)

    def test_no_self_pairs(self):
        pairs = list(generate_candidate_pairs(SYNTHETIC_RECORDS))
        for a, b in pairs:
            self.assertNotEqual(a, b)

    def test_cross_country_pairs_absent(self):
        """With country blocking enabled, US <-> IN pairs must not appear."""
        pairs = list(generate_candidate_pairs(
            SYNTHETIC_RECORDS,
            use_country=True,
            use_exact_name=True,
        ))
        us_ids = {"S1-001", "S2-001", "S1-002"}
        in_ids = {"S2-002", "S3-001"}
        for a, b in pairs:
            in_us = a in us_ids or b in us_ids
            in_in = a in in_ids or b in in_ids
            self.assertFalse(in_us and in_in, f"Cross-country pair found: ({a}, {b})")

    def test_reliance_india_pair_present(self):
        """S2-002 and S3-001 share the same name and country -> must be paired."""
        pairs = set(generate_candidate_pairs(
            SYNTHETIC_RECORDS,
            use_country=True,
            use_exact_name=True,
        ))
        self.assertIn(("S2-002", "S3-001"), pairs)

    def test_pair_count_much_less_than_all_pairs(self):
        """Blocking must reduce pairs significantly vs all-pairs comparison."""
        n = len(SYNTHETIC_RECORDS)
        all_pairs = n * (n - 1) // 2  # 15 for n=6

        actual_pairs = list(generate_candidate_pairs(
            SYNTHETIC_RECORDS,
            use_country=True,
            use_exact_name=True,
        ))
        self.assertLess(len(actual_pairs), all_pairs,
                        "Blocking should produce fewer pairs than all-pairs")


# ===========================================================================
# F. pairs_from_index
# ===========================================================================


class TestPairsFromIndex(unittest.TestCase):
    def test_generates_correct_pairs(self):
        index = {"block1": ["A", "B", "C"], "block2": ["D", "E"]}
        pairs = set(pairs_from_index(index))
        # block1: (A,B), (A,C), (B,C) -- 3 pairs
        # block2: (D,E)               -- 1 pair
        self.assertEqual(len(pairs), 4)
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("A", "C"), pairs)
        self.assertIn(("B", "C"), pairs)
        self.assertIn(("D", "E"), pairs)

    def test_singleton_block_produces_no_pairs(self):
        index = {"solo": ["X"]}
        pairs = list(pairs_from_index(index))
        self.assertEqual(pairs, [])

    def test_pairs_are_deduplicated(self):
        """Duplicate entity IDs within a block must be deduplicated."""
        index = {"dup": ["A", "A", "B"]}
        pairs = list(pairs_from_index(index))
        # Deduplicated: only (A, B)
        self.assertEqual(len(pairs), 1)
        self.assertIn(("A", "B"), pairs)


if __name__ == "__main__":
    unittest.main()
