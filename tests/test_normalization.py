"""
tests/test_normalization.py
===========================
Unit tests for src/normalization.py.

All tests use small synthetic examples — never the real 2.5 GB dataset.
"""

import math
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.normalization import (
    _is_valid,
    is_missing,
    normalize_address,
    normalize_business_name,
    normalize_country,
    normalize_punctuation,
    normalize_whitespace,
    safe_field,
    to_lowercase,
    unicode_normalize,
)


# ===========================================================================
# _is_valid / is_missing
# ===========================================================================


class TestIsValid(unittest.TestCase):
    """Tests for the internal _is_valid helper and public is_missing."""

    def test_none_is_invalid(self):
        self.assertFalse(_is_valid(None))

    def test_nan_float_is_invalid(self):
        self.assertFalse(_is_valid(float("nan")))

    def test_empty_string_is_invalid(self):
        self.assertFalse(_is_valid(""))
        self.assertFalse(_is_valid("   "))

    def test_sentinel_strings_are_invalid(self):
        for sentinel in ("nan", "NaN", "none", "None", "null", "NULL", "N/A", "na", "-"):
            with self.subTest(sentinel=sentinel):
                self.assertFalse(_is_valid(sentinel))

    def test_valid_strings(self):
        self.assertTrue(_is_valid("hello"))
        self.assertTrue(_is_valid("ABC Corp"))
        self.assertTrue(_is_valid("  ABC  "))

    def test_is_missing_mirrors_is_valid(self):
        self.assertTrue(is_missing(None))
        self.assertTrue(is_missing(""))
        self.assertFalse(is_missing("Google LLC"))


# ===========================================================================
# safe_field
# ===========================================================================


class TestSafeField(unittest.TestCase):
    def test_returns_stripped_string(self):
        self.assertEqual(safe_field("  Acme Corp  "), "Acme Corp")

    def test_returns_default_for_none(self):
        self.assertEqual(safe_field(None, default="UNKNOWN"), "UNKNOWN")

    def test_returns_default_for_nan(self):
        self.assertEqual(safe_field(float("nan"), default=""), "")

    def test_empty_default(self):
        self.assertEqual(safe_field(None), "")


# ===========================================================================
# unicode_normalize
# ===========================================================================


class TestUnicodeNormalize(unittest.TestCase):
    """Verify that Unicode normalization does not destroy content."""

    def test_nfc_composes_characters(self):
        # NFD: e + combining acute -> NFC: single precomposed e-acute
        nfd = "e\u0301cafe"  # e + combining acute + cafe
        result = unicode_normalize(nfd, form="NFC")
        self.assertIsNotNone(result)
        self.assertIn("\xe9", result)  # precomposed é

    def test_preserves_french_accents(self):
        french = "Société Générale"
        result = unicode_normalize(french)
        self.assertIsNotNone(result)
        # Accented characters must survive.
        self.assertIn("é", result)
        self.assertIn("é", result)

    def test_preserves_devanagari(self):
        devanagari = "रिलायंस इंडस्ट्रीज"
        result = unicode_normalize(devanagari)
        self.assertIsNotNone(result)
        self.assertEqual(result, devanagari)  # NFC is idempotent for Devanagari

    def test_returns_none_for_missing(self):
        self.assertIsNone(unicode_normalize(None))
        self.assertIsNone(unicode_normalize(""))


# ===========================================================================
# to_lowercase
# ===========================================================================


class TestToLowercase(unittest.TestCase):
    def test_ascii_lowercase(self):
        self.assertEqual(to_lowercase("HELLO WORLD"), "hello world")

    def test_french_lowercase(self):
        self.assertEqual(to_lowercase("SOCIÉTÉ"), "société")

    def test_preserves_devanagari(self):
        # Devanagari has no case concept; should be returned unchanged.
        s = "रिलायंस"
        self.assertEqual(to_lowercase(s), s)

    def test_none_returns_none(self):
        self.assertIsNone(to_lowercase(None))


# ===========================================================================
# normalize_whitespace
# ===========================================================================


class TestNormalizeWhitespace(unittest.TestCase):
    def test_collapses_multiple_spaces(self):
        self.assertEqual(normalize_whitespace("A   B   C"), "A B C")

    def test_strips_leading_trailing(self):
        self.assertEqual(normalize_whitespace("  hello  "), "hello")

    def test_handles_tabs_and_newlines(self):
        self.assertEqual(normalize_whitespace("A\tB\nC"), "A B C")

    def test_handles_nbsp(self):
        result = normalize_whitespace("A\u00a0B")  # Non-breaking space
        self.assertEqual(result, "A B")

    def test_none_returns_none(self):
        self.assertIsNone(normalize_whitespace(None))


# ===========================================================================
# normalize_punctuation
# ===========================================================================


class TestNormalizePunctuation(unittest.TestCase):
    def test_replaces_comma_with_space(self):
        result = normalize_punctuation("ABC,Inc")
        self.assertIn("ABC", result)
        self.assertIn("Inc", result)

    def test_keeps_hyphens_by_default(self):
        result = normalize_punctuation("Hewlett-Packard")
        self.assertIn("-", result)

    def test_removes_hyphens_when_disabled(self):
        result = normalize_punctuation("Hewlett-Packard", keep_hyphens=False)
        self.assertNotIn("-", result)

    def test_preserves_non_ascii_tokens(self):
        # Punctuation removal should not destroy Devanagari letters.
        result = normalize_punctuation("रिलायंस, लिमिटेड")
        self.assertIn("रिलायंस", result)
        self.assertIn("लिमिटेड", result)

    def test_none_returns_none(self):
        self.assertIsNone(normalize_punctuation(None))


# ===========================================================================
# normalize_business_name
# ===========================================================================


class TestNormalizeBusinessName(unittest.TestCase):
    """Core tests for the full business-name normalization pipeline."""

    def test_basic_normalization(self):
        self.assertEqual(normalize_business_name("ACME CORP"), "acme corp")

    def test_llc_variants_canonicalized(self):
        for variant in ["LLC", "L.L.C.", "L.L.C", "llc."]:
            with self.subTest(variant=variant):
                result = normalize_business_name(f"Global Tech {variant}")
                self.assertIn("llc", result)
                # No literal dot should remain in the canonical form.
                self.assertNotIn(".", result)

    def test_inc_variants_canonicalized(self):
        self.assertEqual(
            normalize_business_name("Acme Incorporated"),
            normalize_business_name("Acme Inc"),
        )

    def test_preserves_hyphen_in_name(self):
        result = normalize_business_name("Hewlett-Packard Inc.")
        self.assertIn("-", result)

    def test_french_accents_preserved(self):
        result = normalize_business_name("Société Anonyme")
        self.assertIsNotNone(result)
        self.assertIn("société", result)  # lowercased, accents kept

    def test_devanagari_preserved(self):
        result = normalize_business_name("रिलायंस इंडस्ट्रीज लिमिटेड")
        self.assertIsNotNone(result)
        self.assertIn("रिलायंस", result)

    def test_missing_returns_none(self):
        self.assertIsNone(normalize_business_name(None))
        self.assertIsNone(normalize_business_name(""))
        self.assertIsNone(normalize_business_name("nan"))

    def test_raw_value_not_mutated(self):
        raw = "ACME LLC"
        original = raw
        normalize_business_name(raw)
        self.assertEqual(raw, original)

    def test_comma_separated_suffix(self):
        result = normalize_business_name("Widgets, Inc.")
        self.assertIsNotNone(result)
        self.assertIn("widgets", result)
        self.assertIn("inc", result)

    def test_extra_whitespace_collapsed(self):
        result = normalize_business_name("  Big   Business   Corp  ")
        self.assertEqual(result, "big business corp")


# ===========================================================================
# normalize_address
# ===========================================================================


class TestNormalizeAddress(unittest.TestCase):
    """Tests for address normalization."""

    def test_abbrev_expansion_street(self):
        result = normalize_address("123 Main St")
        self.assertIn("street", result)
        # Original abbreviation should be gone.
        self.assertNotIn(" st", result.split())

    def test_abbrev_expansion_avenue(self):
        result = normalize_address("45 Park Ave")
        self.assertIn("avenue", result)

    def test_missing_address_returns_none(self):
        """Covers the ~3.3% missing address case in Source 2/3."""
        self.assertIsNone(normalize_address(None))
        self.assertIsNone(normalize_address(""))
        self.assertIsNone(normalize_address(float("nan")))

    def test_french_address_preserved(self):
        result = normalize_address("12 Rue de la Paix, Paris")
        self.assertIsNotNone(result)
        self.assertIn("rue", result)  # lowercase
        self.assertIn("paix", result)

    def test_hyphen_in_address_preserved(self):
        result = normalize_address("Suite 10-B, Main Street")
        self.assertIn("-", result)

    def test_unicode_address_preserved(self):
        addr = "456 मेन रोड, मुंबई"
        result = normalize_address(addr)
        self.assertIsNotNone(result)
        self.assertIn("मेन", result)


# ===========================================================================
# normalize_country
# ===========================================================================


class TestNormalizeCountry(unittest.TestCase):
    """Tests for country canonicalization."""

    def test_us_variants(self):
        for raw in ["US", "us", "USA", "United States", "united states of america"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), "US")

    def test_india_variants(self):
        for raw in ["India", "IN", "india"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), "IN")

    def test_france_variants(self):
        for raw in ["France", "FR", "france", "fr"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), "FR")

    def test_unknown_country_uppercased(self):
        result = normalize_country("Brazil")
        self.assertEqual(result, "BRAZIL")

    def test_none_returns_none(self):
        self.assertIsNone(normalize_country(None))
        self.assertIsNone(normalize_country(""))


if __name__ == "__main__":
    unittest.main()
