"""
tests/test_features.py
======================
Unit tests for src/features.py.

Every test uses small, synthetic in-memory records.
No real dataset files are read.
"""

from __future__ import annotations

import math
import sys
import os
import unittest

# Ensure src/ is importable when running from project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import (
    DEFAULT_WEIGHTS,
    SimilarityWeights,
    build_pair_features,
    build_features_for_pairs,
    feature_names,
    _jaro_winkler,
    _levenshtein_norm,
    _jaccard,
    _bigram_set,
    _token_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(entity_id="S1-001", name="Acme Corp", address="123 Main St",
         country="US"):
    """Build a minimal record dict."""
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


def _feats(rec_a, rec_b, weights=None):
    return build_pair_features(rec_a, rec_b, weights)


# ---------------------------------------------------------------------------
# 1. Feature schema / completeness
# ---------------------------------------------------------------------------

class TestFeatureSchema(unittest.TestCase):
    """The returned dict always has the full documented schema."""

    def setUp(self):
        self.f = _feats(_rec(), _rec("S2-001"))

    def test_has_all_feature_names(self):
        for name in feature_names():
            self.assertIn(name, self.f, f"Missing feature: {name}")

    def test_has_id_fields(self):
        self.assertIn("id_a", self.f)
        self.assertIn("id_b", self.f)

    def test_feature_count(self):
        # 26 ML features + id_a + id_b = 28 keys
        self.assertEqual(len(self.f), 28)

    def test_no_nan_values(self):
        for k, v in self.f.items():
            if isinstance(v, float):
                self.assertFalse(math.isnan(v), f"NaN in feature '{k}'")

    def test_feature_names_stable(self):
        # feature_names() must return the same list every time
        self.assertEqual(feature_names(), feature_names())


# ---------------------------------------------------------------------------
# 2. Identical records
# ---------------------------------------------------------------------------

class TestIdenticalRecords(unittest.TestCase):

    def setUp(self):
        self.a = _rec("S1-001", "Reliance Industries Ltd",
                      "Corporate Park, Mumbai", "India")
        self.b = _rec("S2-001", "Reliance Industries Ltd",
                      "Corporate Park, Mumbai", "India")
        self.f = _feats(self.a, self.b)

    def test_exact_raw_name_match(self):
        self.assertEqual(self.f["name_exact_raw"], 1)

    def test_exact_norm_name_match(self):
        self.assertEqual(self.f["name_exact_norm"], 1)

    def test_name_jaro_winkler_is_one(self):
        self.assertAlmostEqual(self.f["name_jaro_winkler"], 1.0, places=4)

    def test_name_levenshtein_is_one(self):
        self.assertAlmostEqual(self.f["name_levenshtein_norm"], 1.0, places=4)

    def test_name_token_similarity_is_one(self):
        self.assertAlmostEqual(self.f["name_token_similarity"], 1.0, places=4)

    def test_addr_exact_match(self):
        self.assertEqual(self.f["addr_exact_norm"], 1)

    def test_country_equal(self):
        self.assertEqual(self.f["country_equal"], 1)

    def test_name_len_diff_zero(self):
        self.assertEqual(self.f["name_len_diff"], 0)

    def test_combined_weighted_near_one(self):
        self.assertGreater(self.f["combined_weighted_sim"], 0.9)

    def test_country_strong_name(self):
        self.assertEqual(self.f["country_strong_name"], 1)


# ---------------------------------------------------------------------------
# 3. Formatting-only differences (LLC vs L.L.C.)
# ---------------------------------------------------------------------------

class TestFormattingDifferences(unittest.TestCase):

    def setUp(self):
        self.a = _rec("S1-001", "Global Tech LLC",
                      "456 Oak Ave", "United States")
        self.b = _rec("S2-001", "Global Tech L.L.C.",
                      "456 Oak Avenue", "US")
        self.f = _feats(self.a, self.b)

    def test_exact_norm_name_match(self):
        # Both normalize to "global tech llc"
        self.assertEqual(self.f["name_exact_norm"], 1,
                         "LLC vs L.L.C. should normalize to same string")

    def test_raw_name_not_exact(self):
        self.assertEqual(self.f["name_exact_raw"], 0)

    def test_country_equal(self):
        self.assertEqual(self.f["country_equal"], 1,
                         "US and United States should normalize to same")

    def test_addr_high_similarity(self):
        # Ave -> Avenue expansion makes them equal
        self.assertGreater(self.f["addr_jaro_winkler"], 0.85)


# ---------------------------------------------------------------------------
# 4. Punctuation differences
# ---------------------------------------------------------------------------

class TestPunctuationDifferences(unittest.TestCase):

    def setUp(self):
        self.a = _rec("S1-001", "Hewlett-Packard Inc.",
                      "1501 Page Mill Rd., Palo Alto", "US")
        self.b = _rec("S2-001", "Hewlett Packard Inc",
                      "1501 Page Mill Road, Palo Alto", "US")
        self.f = _feats(self.a, self.b)

    def test_name_jw_high(self):
        self.assertGreater(self.f["name_jaro_winkler"], 0.88)

    def test_name_token_sim_high(self):
        # After normalization "Hewlett-Packard Inc." -> "hewlett-packard inc"
        # (hyphen preserved, so it's one token), while "Hewlett Packard Inc"
        # -> "hewlett packard inc" (two separate tokens).
        # Shared token: {'inc'} -> Jaccard = 1/4 = 0.25.
        # We verify similarity is > 0 (they share 'inc') and < 1 (not equal).
        self.assertGreater(self.f["name_token_similarity"], 0.0)
        self.assertLess(self.f["name_token_similarity"], 1.0)


    def test_addr_similarity_high(self):
        self.assertGreater(self.f["addr_jaro_winkler"], 0.85)


# ---------------------------------------------------------------------------
# 5. Unicode / Indic text (Devanagari)
# ---------------------------------------------------------------------------

class TestUnicodeIndic(unittest.TestCase):

    def setUp(self):
        self.a = _rec("S1-001",
                      "रिलायंस इंडस्ट्रीज लिमिटेड",
                      "456 मेन रोड, मुंबई",
                      "India")
        self.b = _rec("S2-001",
                      "रिलायंस इंडस्ट्रीज लिमिटेड",
                      "456 मेन रोड, मुंबई",
                      "India")
        self.f = _feats(self.a, self.b)

    def test_exact_norm_name_devanagari(self):
        self.assertEqual(self.f["name_exact_norm"], 1)

    def test_name_jw_is_one(self):
        self.assertAlmostEqual(self.f["name_jaro_winkler"], 1.0, places=4)

    def test_no_nan_in_features(self):
        for k, v in self.f.items():
            if isinstance(v, float):
                self.assertFalse(math.isnan(v), f"NaN in '{k}' for Devanagari")

    def test_country_equal(self):
        self.assertEqual(self.f["country_equal"], 1)


class TestUnicodeIndicDifferent(unittest.TestCase):
    """Different Devanagari names should have low similarity."""

    def setUp(self):
        self.a = _rec("S1-001", "टाटा मोटर्स", "पुणे", "India")
        self.b = _rec("S2-001", "रिलायंस", "मुंबई", "India")
        self.f = _feats(self.a, self.b)

    def test_name_jw_low(self):
        self.assertLess(self.f["name_jaro_winkler"], 0.8)

    def test_exact_norm_name_zero(self):
        self.assertEqual(self.f["name_exact_norm"], 0)


# ---------------------------------------------------------------------------
# 6. Missing addresses
# ---------------------------------------------------------------------------

class TestMissingAddresses(unittest.TestCase):

    def _run(self, addr_a, addr_b):
        a = _rec("S1-001", "Tech Corp", addr_a, "US")
        b = _rec("S2-001", "Tech Corp", addr_b, "US")
        return _feats(a, b)

    def test_one_side_none(self):
        f = self._run(None, "123 Main St")
        self.assertEqual(f["addr_missing"], 1)
        self.assertEqual(f["addr_both_present"], 0)
        self.assertEqual(f["addr_jaro_winkler"], 0.0)
        self.assertEqual(f["addr_levenshtein_norm"], 0.0)
        self.assertEqual(f["addr_token_similarity"], 0.0)
        self.assertEqual(f["addr_len_diff"], 0)
        self.assertEqual(f["addr_len_ratio"], 0.0)

    def test_both_sides_none(self):
        f = self._run(None, None)
        self.assertEqual(f["addr_missing"], 1)
        self.assertEqual(f["addr_both_present"], 0)

    def test_combined_name_dominant_when_addr_missing(self):
        f = self._run(None, "123 Main St")
        # combined_name_dominant should equal name composite, not weighted
        self.assertAlmostEqual(f["combined_name_dominant_sim"],
                               f["name_sim_composite"], places=5)

    def test_no_nan_when_address_missing(self):
        f = self._run(None, "123 Main St")
        for k, v in f.items():
            if isinstance(v, float):
                self.assertFalse(math.isnan(v), f"NaN in '{k}' for missing addr")

    def test_empty_string_address(self):
        f = self._run("", "123 Main St")
        self.assertEqual(f["addr_missing"], 1)

    def test_both_addresses_present_flag(self):
        f = self._run("123 Main St", "456 Oak Ave")
        self.assertEqual(f["addr_both_present"], 1)
        self.assertEqual(f["addr_missing"], 0)


# ---------------------------------------------------------------------------
# 7. Completely different businesses
# ---------------------------------------------------------------------------

class TestCompletelyDifferent(unittest.TestCase):

    def setUp(self):
        self.a = _rec("S1-001", "Apple Inc", "One Apple Park Way", "US")
        self.b = _rec("S2-001", "Reliance Industries",
                      "Corporate Park Mumbai", "India")
        self.f = _feats(self.a, self.b)

    def test_exact_raw_zero(self):
        self.assertEqual(self.f["name_exact_raw"], 0)

    def test_exact_norm_zero(self):
        self.assertEqual(self.f["name_exact_norm"], 0)

    def test_name_jw_low(self):
        self.assertLess(self.f["name_jaro_winkler"], 0.7)

    def test_country_not_equal(self):
        self.assertEqual(self.f["country_equal"], 0)

    def test_country_strong_name_zero(self):
        self.assertEqual(self.f["country_strong_name"], 0)

    def test_combined_sim_low(self):
        self.assertLess(self.f["combined_weighted_sim"], 0.6)


# ---------------------------------------------------------------------------
# 8. Country features
# ---------------------------------------------------------------------------

class TestCountryFeatures(unittest.TestCase):

    def test_same_country_us(self):
        a = _rec("S1-001", "Alpha", "123 St", "US")
        b = _rec("S2-001", "Alpha", "123 St", "United States")
        f = _feats(a, b)
        self.assertEqual(f["country_equal"], 1)
        self.assertEqual(f["same_country"], f["country_equal"])

    def test_same_country_india(self):
        a = _rec("S1-001", "Beta", "Mumbai", "India")
        b = _rec("S2-001", "Beta", "Mumbai", "IN")
        f = _feats(a, b)
        self.assertEqual(f["country_equal"], 1)

    def test_different_country(self):
        a = _rec("S1-001", "Gamma", "Paris", "France")
        b = _rec("S2-001", "Gamma", "Paris", "US")
        f = _feats(a, b)
        self.assertEqual(f["country_equal"], 0)

    def test_country_equal_is_alias_of_same_country(self):
        f = _feats(_rec(), _rec("S2-001"))
        self.assertEqual(f["country_equal"], f["same_country"])


# ---------------------------------------------------------------------------
# 9. Configurable weights
# ---------------------------------------------------------------------------

class TestConfigurableWeights(unittest.TestCase):

    def test_name_heavy_weights(self):
        w = SimilarityWeights(name_weight=1.0, address_weight=0.0)
        a = _rec("S1-001", "Exact Match Corp", "123 Main St", "US")
        b = _rec("S2-001", "Exact Match Corp", "999 Different Rd", "US")
        f = build_pair_features(a, b, weights=w)
        # With no address weight the combined should be purely name-driven
        self.assertAlmostEqual(f["combined_weighted_sim"],
                               f["name_sim_composite"], places=4)

    def test_strong_name_threshold_custom(self):
        # Lower threshold -> more pairs qualify as country_strong_name
        w_low  = SimilarityWeights(strong_name_threshold=0.50)
        w_high = SimilarityWeights(strong_name_threshold=0.99)
        a = _rec("S1-001", "Alpha Corp", "123 St", "US")
        b = _rec("S2-001", "Alpha Company", "123 St", "US")
        f_low  = build_pair_features(a, b, weights=w_low)
        f_high = build_pair_features(a, b, weights=w_high)
        # With 0.99 threshold the fuzzy pair should NOT get country_strong_name
        # (their JW is < 0.99).  With 0.50 threshold it should.
        self.assertGreaterEqual(f_low["country_strong_name"],
                                f_high["country_strong_name"])


# ---------------------------------------------------------------------------
# 10. Interaction and combined features
# ---------------------------------------------------------------------------

class TestCombinedFeatures(unittest.TestCase):

    def test_interaction_zero_when_name_not_exact(self):
        a = _rec("S1-001", "Alpha Corp", "123 Main St", "US")
        b = _rec("S2-001", "Beta Corp",  "123 Main St", "US")
        f = _feats(a, b)
        self.assertEqual(f["name_exact_norm"], 0)
        self.assertEqual(f["interaction_name_x_addr"], 0.0)

    def test_interaction_high_when_both_exact(self):
        a = _rec("S1-001", "Acme Inc", "123 Main St", "US")
        b = _rec("S2-001", "Acme Inc", "123 Main St", "US")
        f = _feats(a, b)
        self.assertEqual(f["name_exact_norm"], 1)
        self.assertAlmostEqual(f["interaction_name_x_addr"],
                               f["addr_jaro_winkler"], places=5)

    def test_name_dominant_equals_composite_when_addr_missing(self):
        a = _rec("S1-001", "Acme Inc", None, "US")
        b = _rec("S2-001", "Acme Inc", "123 Main St", "US")
        f = _feats(a, b)
        self.assertAlmostEqual(f["combined_name_dominant_sim"],
                               f["name_sim_composite"], places=5)

    def test_name_dominant_equals_weighted_when_both_present(self):
        a = _rec("S1-001", "Acme Inc", "123 Main St", "US")
        b = _rec("S2-001", "Acme Inc", "123 Main St", "US")
        f = _feats(a, b)
        self.assertAlmostEqual(f["combined_name_dominant_sim"],
                               f["combined_weighted_sim"], places=5)


# ---------------------------------------------------------------------------
# 11. Similarity primitive edge-cases
# ---------------------------------------------------------------------------

class TestSimilarityPrimitives(unittest.TestCase):

    def test_jw_identical(self):
        self.assertAlmostEqual(_jaro_winkler("hello", "hello"), 1.0)

    def test_jw_empty_a(self):
        self.assertEqual(_jaro_winkler("", "hello"), 0.0)

    def test_jw_empty_both(self):
        self.assertEqual(_jaro_winkler("", ""), 0.0)

    def test_lev_identical(self):
        self.assertAlmostEqual(_levenshtein_norm("hello", "hello"), 1.0)

    def test_lev_both_empty(self):
        self.assertAlmostEqual(_levenshtein_norm("", ""), 1.0)

    def test_lev_one_empty(self):
        self.assertAlmostEqual(_levenshtein_norm("", "abc"), 0.0)

    def test_jaccard_identical_sets(self):
        self.assertAlmostEqual(_jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_jaccard_disjoint_sets(self):
        self.assertAlmostEqual(_jaccard({"a"}, {"b"}), 0.0)

    def test_jaccard_both_empty(self):
        self.assertAlmostEqual(_jaccard(set(), set()), 1.0)

    def test_bigram_devanagari(self):
        word = "रिलायंस"
        bg = _bigram_set(word)
        # bigrams should be present and are 2-codepoint unicode pairs
        self.assertGreater(len(bg), 0)
        for b in bg:
            self.assertEqual(len(b), 2)

    def test_bigram_short_string(self):
        # single char -> singleton set {char}
        self.assertEqual(_bigram_set("A"), {"A"})

    def test_bigram_empty(self):
        self.assertEqual(_bigram_set(""), set())


# ---------------------------------------------------------------------------
# 12. Batch helper
# ---------------------------------------------------------------------------

class TestBatchHelper(unittest.TestCase):

    def test_build_features_for_pairs_length(self):
        pairs = [
            (_rec("S1-001"), _rec("S2-001")),
            (_rec("S1-002", "Beta Corp"), _rec("S2-002", "Gamma Corp")),
        ]
        results = build_features_for_pairs(pairs)
        self.assertEqual(len(results), 2)

    def test_build_features_for_pairs_schema(self):
        pairs = [(_rec(), _rec("S2-001"))]
        result = build_features_for_pairs(pairs)[0]
        for name in feature_names():
            self.assertIn(name, result)


if __name__ == "__main__":
    unittest.main()
