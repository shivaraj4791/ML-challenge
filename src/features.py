"""
features.py
===========
Feature engineering for the Amazon ML Challenge entity-resolution pipeline.

Each feature is computed for a *candidate pair* (record_a, record_b) and
returned as a plain Python dict with a stable, documented schema.  The dict
can be consumed directly by scikit-learn, XGBoost, or any other ML framework.

Design principles
-----------------
* All string comparisons go through the Unicode-safe normalization functions
  in src/normalization.py.  No ASCII-only stripping (no [^a-z0-9]).
* Missing addresses (~3.3% in S2/S3) are handled explicitly as features.
* Weights for combined features are configurable via SimilarityWeights.
* Functions are pure and stateless -- safe for parallel / chunked use.
* No dataset files are opened here.

Feature schema (37 features)
-----------------------------
Country (2):
  country_equal               int  1 if normalized countries match
  same_country                int  alias of country_equal (explicit binary)

Name (9):
  name_exact_raw              int  1 if raw strings are identical
  name_exact_norm             int  1 if normalized strings are identical
  name_jaro_winkler           float Jaro-Winkler similarity in [0, 1]
  name_levenshtein_norm       float 1 - edit_distance / max_len  (in [0,1])
  name_token_similarity       float Jaccard over name token sets
  name_len_diff               int  abs(len_a - len_b) on normalized strings
  name_len_ratio              float min/max len ratio in [0, 1]
  name_token_count_diff       int  abs(token_count_a - token_count_b)
  name_char_bigram_sim        float character-bigram Jaccard (handles Indic)

Address (9):
  addr_missing                int  1 if either address is missing
  addr_both_present           int  1 if both addresses are present
  addr_exact_norm             int  1 if normalized addresses are identical
  addr_jaro_winkler           float Jaro-Winkler (0.0 when either missing)
  addr_levenshtein_norm       float normalised Levenshtein (0.0 when missing)
  addr_token_similarity       float Jaccard over address token sets
  addr_len_diff               int  abs(len_a - len_b); 0 when missing
  addr_len_ratio              float min/max len ratio; 0.0 when missing
  addr_token_count_diff       int  token count diff; 0 when missing

Combined (6):
  combined_weighted_sim       float w_name*name_sim + w_addr*addr_sim
  combined_name_dominant_sim  float name_sim when addr missing, else weighted
  interaction_name_x_addr     float name_exact_norm * addr_jaro_winkler
  country_strong_name         int  country_equal AND name_jaro_winkler >= 0.9
  name_sim_composite          float mean(jw, lev_norm, token) for name
  addr_sim_composite          float mean(jw, lev_norm, token) for address

Source metadata (2 -- not used as ML features, kept for traceability):
  id_a                        str  entity_id of record a
  id_b                        str  entity_id of record b

Total features fed to ML: 35  (excluding id_a, id_b)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Lazy imports for similarity libraries (graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    import jellyfish as _jellyfish  # type: ignore[import-untyped]
    _HAS_JELLYFISH = True
except ImportError:  # pragma: no cover
    _HAS_JELLYFISH = False

try:
    from rapidfuzz.distance import Levenshtein as _RFLev  # type: ignore
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

from .normalization import (
    is_missing,
    normalize_address,
    normalize_business_name,
    normalize_country,
    safe_field,
)

# ---------------------------------------------------------------------------
# Configurable weights
# ---------------------------------------------------------------------------


@dataclass
class SimilarityWeights:
    """Weights used in combined similarity features.

    All weights are normalised internally so they don't need to sum to 1.
    Tune these for your matching task -- do NOT hard-code in feature logic.

    Attributes:
        name_weight:    Weight given to name similarity in combined score.
        address_weight: Weight given to address similarity in combined score.
        jw_weight:      Sub-weight for Jaro-Winkler within name/addr composite.
        lev_weight:     Sub-weight for normalised Levenshtein within composite.
        token_weight:   Sub-weight for token Jaccard within composite.
        strong_name_threshold: Jaro-Winkler >= this => "strong name match".
    """
    name_weight: float = 0.6
    address_weight: float = 0.4
    jw_weight: float = 0.4
    lev_weight: float = 0.3
    token_weight: float = 0.3
    strong_name_threshold: float = 0.90


# Module-level default weights (callers can pass their own).
DEFAULT_WEIGHTS = SimilarityWeights()


# ---------------------------------------------------------------------------
# Internal string-similarity primitives
# ---------------------------------------------------------------------------


def _jaro_winkler(a: str, b: str) -> float:
    """Return Jaro-Winkler similarity in [0, 1].

    Uses ``jellyfish`` if available, otherwise falls back to a pure-Python
    implementation so the module always works.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _HAS_JELLYFISH:
        return float(_jellyfish.jaro_winkler_similarity(a, b))
    return _jaro_winkler_pure(a, b)


def _jaro_winkler_pure(s1: str, s2: str) -> float:
    """Pure-Python Jaro-Winkler (fallback when jellyfish not installed)."""
    if s1 == s2:
        return 1.0
    len_s1, len_s2 = len(s1), len(s2)
    match_dist = max(len_s1, len_s2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    matches = transpositions = 0
    for i, ch in enumerate(s1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len_s2)
        for j in range(lo, hi):
            if s2_matches[j] or ch != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / len_s1 + matches / len_s2 +
            (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for ch1, ch2 in zip(s1[:4], s2[:4]):
        if ch1 == ch2:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def _levenshtein_norm(a: str, b: str) -> float:
    """Return normalised Levenshtein similarity in [0, 1].

    similarity = 1 - edit_distance / max(len_a, len_b)
    Uses rapidfuzz if available for speed; falls back to stdlib.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _HAS_RAPIDFUZZ:
        dist = _RFLev.distance(a, b)
    else:
        dist = _levenshtein_pure(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len


def _levenshtein_pure(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein distance (O(len_s1 * len_s2) space)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def _token_set(text: str) -> set[str]:
    """Split *text* into a set of tokens (whitespace-split, length >= 1)."""
    return set(text.split())


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _bigram_set(text: str) -> set[str]:
    """Return the set of character bigrams from *text*.

    Works correctly for Unicode (Devanagari, accented Latin, etc.) because
    Python string indexing is code-point based.
    """
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _composite_sim(jw: float, lev: float, token: float,
                   weights: SimilarityWeights) -> float:
    """Weighted composite of three similarity scores."""
    w = weights.jw_weight + weights.lev_weight + weights.token_weight
    if w == 0:
        return 0.0
    return (weights.jw_weight * jw + weights.lev_weight * lev +
            weights.token_weight * token) / w


def _len_ratio(a: str, b: str) -> float:
    """min/max length ratio in [0, 1]; 1.0 if both empty."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


# ---------------------------------------------------------------------------
# Main feature-building function
# ---------------------------------------------------------------------------


def build_pair_features(
    rec_a: dict[str, Any],
    rec_b: dict[str, Any],
    weights: Optional[SimilarityWeights] = None,
) -> dict[str, Any]:
    """Compute all features for a candidate (rec_a, rec_b) pair.

    Records must be plain dicts with keys:
        entity_id, business_name, business_address, country

    Values may be raw strings, None, or float NaN (as loaded by pandas).
    All normalization is done internally; raw values are never mutated.

    Args:
        rec_a: First entity record (typically Source 1).
        rec_b: Second entity record (Source 2 or 3).
        weights: Optional :class:`SimilarityWeights` instance.  Defaults to
            :data:`DEFAULT_WEIGHTS`.

    Returns:
        Dict with the stable feature schema described in the module docstring.
        All numeric values are Python float or int -- no numpy types.
    """
    w = weights or DEFAULT_WEIGHTS

    # ------------------------------------------------------------------
    # 1. Resolve raw and normalized field values
    # ------------------------------------------------------------------
    raw_name_a = safe_field(rec_a.get("business_name"), "")
    raw_name_b = safe_field(rec_b.get("business_name"), "")

    norm_name_a = normalize_business_name(rec_a.get("business_name")) or ""
    norm_name_b = normalize_business_name(rec_b.get("business_name")) or ""

    raw_addr_a = rec_a.get("business_address")
    raw_addr_b = rec_b.get("business_address")
    addr_a_missing = is_missing(raw_addr_a)
    addr_b_missing = is_missing(raw_addr_b)
    addr_missing = int(addr_a_missing or addr_b_missing)
    addr_both_present = int(not addr_a_missing and not addr_b_missing)

    norm_addr_a = normalize_address(raw_addr_a) or ""
    norm_addr_b = normalize_address(raw_addr_b) or ""

    norm_country_a = normalize_country(rec_a.get("country")) or ""
    norm_country_b = normalize_country(rec_b.get("country")) or ""

    # ------------------------------------------------------------------
    # 2. Country features
    # ------------------------------------------------------------------
    country_equal = int(norm_country_a == norm_country_b and
                        norm_country_a != "")

    # ------------------------------------------------------------------
    # 3. Name features
    # ------------------------------------------------------------------
    name_exact_raw  = int(raw_name_a == raw_name_b and raw_name_a != "")
    name_exact_norm = int(norm_name_a == norm_name_b and norm_name_a != "")

    name_jw  = _jaro_winkler(norm_name_a, norm_name_b)
    name_lev = _levenshtein_norm(norm_name_a, norm_name_b)

    tok_a = _token_set(norm_name_a)
    tok_b = _token_set(norm_name_b)
    name_token_sim = _jaccard(tok_a, tok_b)

    name_len_diff  = abs(len(norm_name_a) - len(norm_name_b))
    name_len_ratio = _len_ratio(norm_name_a, norm_name_b)
    name_tok_count_diff = abs(len(tok_a) - len(tok_b))

    bg_a = _bigram_set(norm_name_a)
    bg_b = _bigram_set(norm_name_b)
    name_bigram_sim = _jaccard(bg_a, bg_b)

    name_composite = _composite_sim(name_jw, name_lev, name_token_sim, w)

    # ------------------------------------------------------------------
    # 4. Address features (gracefully handle missing)
    # ------------------------------------------------------------------
    if addr_both_present:
        addr_exact_norm  = int(norm_addr_a == norm_addr_b and norm_addr_a != "")
        addr_jw          = _jaro_winkler(norm_addr_a, norm_addr_b)
        addr_lev         = _levenshtein_norm(norm_addr_a, norm_addr_b)
        atok_a = _token_set(norm_addr_a)
        atok_b = _token_set(norm_addr_b)
        addr_token_sim   = _jaccard(atok_a, atok_b)
        addr_len_diff    = abs(len(norm_addr_a) - len(norm_addr_b))
        addr_len_ratio   = _len_ratio(norm_addr_a, norm_addr_b)
        addr_tok_count_diff = abs(len(atok_a) - len(atok_b))
        addr_composite   = _composite_sim(addr_jw, addr_lev, addr_token_sim, w)
    else:
        # Encode absence as 0.0 / 0 -- NOT as NaN -- so tree-based models
        # can split on addr_missing directly.
        addr_exact_norm     = 0
        addr_jw             = 0.0
        addr_lev            = 0.0
        addr_token_sim      = 0.0
        addr_len_diff       = 0
        addr_len_ratio      = 0.0
        addr_tok_count_diff = 0
        addr_composite      = 0.0

    # ------------------------------------------------------------------
    # 5. Combined / interaction features
    # ------------------------------------------------------------------
    # Weighted combined (addr contribution zeroed when missing -- model sees
    # addr_missing=1 as a separate signal).
    total_w = w.name_weight + (w.address_weight if addr_both_present else 0.0)
    if total_w > 0:
        combined_weighted = (w.name_weight * name_composite +
                             (w.address_weight * addr_composite
                              if addr_both_present else 0.0)) / total_w
    else:
        combined_weighted = 0.0

    # Name-dominant: use name only when address missing.
    combined_name_dominant = (
        name_composite if addr_missing
        else combined_weighted
    )

    # Interaction: exact norm name AND address JW (high when both are strong)
    interaction_name_x_addr = float(name_exact_norm) * addr_jw

    # Binary indicator: country matches AND name is strong
    country_strong_name = int(
        country_equal == 1 and name_jw >= w.strong_name_threshold
    )

    # ------------------------------------------------------------------
    # 6. Assemble and return
    # ------------------------------------------------------------------
    return {
        # -- metadata (not ML features) --
        "id_a": str(rec_a.get("entity_id", "")),
        "id_b": str(rec_b.get("entity_id", "")),

        # -- country --
        "country_equal": country_equal,
        "same_country":  country_equal,   # explicit alias

        # -- name --
        "name_exact_raw":        name_exact_raw,
        "name_exact_norm":       name_exact_norm,
        "name_jaro_winkler":     round(name_jw, 6),
        "name_levenshtein_norm": round(name_lev, 6),
        "name_token_similarity": round(name_token_sim, 6),
        "name_len_diff":         name_len_diff,
        "name_len_ratio":        round(name_len_ratio, 6),
        "name_token_count_diff": name_tok_count_diff,
        "name_char_bigram_sim":  round(name_bigram_sim, 6),

        # -- address --
        "addr_missing":               addr_missing,
        "addr_both_present":          addr_both_present,
        "addr_exact_norm":            addr_exact_norm,
        "addr_jaro_winkler":          round(addr_jw, 6),
        "addr_levenshtein_norm":      round(addr_lev, 6),
        "addr_token_similarity":      round(addr_token_sim, 6),
        "addr_len_diff":              addr_len_diff,
        "addr_len_ratio":             round(addr_len_ratio, 6),
        "addr_token_count_diff":      addr_tok_count_diff,

        # -- combined --
        "combined_weighted_sim":      round(combined_weighted, 6),
        "combined_name_dominant_sim": round(combined_name_dominant, 6),
        "interaction_name_x_addr":    round(interaction_name_x_addr, 6),
        "country_strong_name":        country_strong_name,
        "name_sim_composite":         round(name_composite, 6),
        "addr_sim_composite":         round(addr_composite, 6),
    }


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def build_features_for_pairs(
    pairs: list[tuple[dict, dict]],
    weights: Optional[SimilarityWeights] = None,
) -> list[dict[str, Any]]:
    """Return a list of feature dicts for a list of (rec_a, rec_b) pairs.

    This is a convenience wrapper around :func:`build_pair_features`.
    For the full 2.52 GB dataset, call this chunk-by-chunk via the
    data_loader generators rather than all at once.

    Args:
        pairs: List of (record_a, record_b) dicts.
        weights: Optional weights.

    Returns:
        List of feature dicts in the same order as *pairs*.
    """
    return [build_pair_features(a, b, weights) for a, b in pairs]


def feature_names() -> list[str]:
    """Return the ordered list of ML feature names (excludes id_a / id_b).

    Stable across all calls -- can be used as column names for a DataFrame.
    """
    return [
        "country_equal", "same_country",
        "name_exact_raw", "name_exact_norm",
        "name_jaro_winkler", "name_levenshtein_norm",
        "name_token_similarity", "name_len_diff", "name_len_ratio",
        "name_token_count_diff", "name_char_bigram_sim",
        "addr_missing", "addr_both_present", "addr_exact_norm",
        "addr_jaro_winkler", "addr_levenshtein_norm",
        "addr_token_similarity", "addr_len_diff", "addr_len_ratio",
        "addr_token_count_diff",
        "combined_weighted_sim", "combined_name_dominant_sim",
        "interaction_name_x_addr", "country_strong_name",
        "name_sim_composite", "addr_sim_composite",
    ]
