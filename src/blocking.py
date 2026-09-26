"""
blocking.py
===========
Candidate-pair generation (blocking) for the entity-resolution pipeline.

Blocking reduces the O(N^2) comparison space by only pairing entities that
share a cheap, high-recall key.  We never compare all pairs.

Strategies implemented
----------------------
A. Country blocking        -- only entities from the same country are paired.
B. Exact-name blocking     -- entities sharing the same normalized name key.
C. Name-prefix blocking    -- entities sharing the same N-character name prefix
                              (optional, additive on top of A+B).
D. Token blocking          -- entities sharing at least one name token.

Extension pattern
-----------------
All strategies follow the same interface:

    def build_<strategy>_index(records, ...) -> dict[key, list[entity_id]]
    def get_<strategy>_candidates(index, record) -> set[entity_id]

This makes it trivial to add new strategies (address token, zip-code, etc.)
without touching existing code.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Generator, Iterable, Optional

from .normalization import (
    is_missing,
    normalize_business_name,
    normalize_country,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: entity_id -> normalized record dict
RecordDict = dict[str, Any]

#: blocking key -> list of entity_ids sharing that key
BlockIndex = dict[str, list[str]]


# ---------------------------------------------------------------------------
# A. Country blocking
# ---------------------------------------------------------------------------


def build_country_index(records: Iterable[RecordDict]) -> BlockIndex:
    """Build an index mapping country -> [entity_ids].

    Only entities within the same country will be compared.  This is the
    first and most important blocking key because:
    - The dataset guarantees 0% missing countries.
    - Train set has US + India; test adds France.
    - A US entity cannot match an India entity by definition.

    Args:
        records: Iterable of record dicts with at least 'entity_id' and
            'country' keys.  Values may be raw or already normalized.

    Returns:
        Dict mapping normalized country code -> list of entity_ids.
    """
    index: BlockIndex = defaultdict(list)
    for rec in records:
        eid = str(rec["entity_id"])
        country = normalize_country(rec.get("country"))
        if country is None:
            log.warning("Missing country for entity_id=%s; skipping.", eid)
            continue
        index[country].append(eid)
    return dict(index)


def get_country_candidates(
    index: BlockIndex, record: RecordDict
) -> set[str]:
    """Return entity_ids in the same country as *record*.

    Args:
        index: Output of :func:`build_country_index`.
        record: Query record dict.

    Returns:
        Set of candidate entity_ids (excludes the query entity itself).
    """
    country = normalize_country(record.get("country"))
    if country is None:
        return set()
    candidates = set(index.get(country, []))
    candidates.discard(str(record.get("entity_id", "")))
    return candidates


# ---------------------------------------------------------------------------
# B. Exact normalized-name blocking
# ---------------------------------------------------------------------------


def build_name_index(records: Iterable[RecordDict]) -> BlockIndex:
    """Build an index mapping normalized_name -> [entity_ids].

    Entities that share the same normalized business name are considered
    high-confidence candidates and must be compared.

    Args:
        records: Iterable of record dicts with 'entity_id' and
            'business_name' keys.

    Returns:
        Dict mapping normalized name -> list of entity_ids.
    """
    index: BlockIndex = defaultdict(list)
    for rec in records:
        eid = str(rec["entity_id"])
        name = normalize_business_name(rec.get("business_name"))
        if name is None:
            log.debug("Missing business_name for entity_id=%s; skipping.", eid)
            continue
        index[name].append(eid)
    return dict(index)


def get_name_candidates(index: BlockIndex, record: RecordDict) -> set[str]:
    """Return entity_ids sharing the exact normalized name with *record*.

    Args:
        index: Output of :func:`build_name_index`.
        record: Query record dict.

    Returns:
        Set of candidate entity_ids.
    """
    name = normalize_business_name(record.get("business_name"))
    if name is None:
        return set()
    candidates = set(index.get(name, []))
    candidates.discard(str(record.get("entity_id", "")))
    return candidates


# ---------------------------------------------------------------------------
# C. Name-prefix blocking  (optional, additive)
# ---------------------------------------------------------------------------


def build_prefix_index(
    records: Iterable[RecordDict], prefix_len: int = 5
) -> BlockIndex:
    """Build an index mapping the first *prefix_len* chars of name -> [eids].

    This is a softer blocking key than exact name: catches partial matches and
    minor prefix differences.  Increase *prefix_len* to reduce false positives.

    Args:
        records: Iterable of record dicts.
        prefix_len: Number of characters in the prefix key (default 5).

    Returns:
        Dict mapping name prefix -> list of entity_ids.
    """
    index: BlockIndex = defaultdict(list)
    for rec in records:
        eid = str(rec["entity_id"])
        name = normalize_business_name(rec.get("business_name"))
        if name is None or len(name) < 2:
            continue
        prefix = name[:prefix_len]
        index[prefix].append(eid)
    return dict(index)


def get_prefix_candidates(
    index: BlockIndex, record: RecordDict, prefix_len: int = 5
) -> set[str]:
    """Return entity_ids sharing the name prefix with *record*.

    Args:
        index: Output of :func:`build_prefix_index`.
        record: Query record dict.
        prefix_len: Must match the value used when building the index.

    Returns:
        Set of candidate entity_ids.
    """
    name = normalize_business_name(record.get("business_name"))
    if name is None or len(name) < 2:
        return set()
    prefix = name[:prefix_len]
    candidates = set(index.get(prefix, []))
    candidates.discard(str(record.get("entity_id", "")))
    return candidates


# ---------------------------------------------------------------------------
# D. Token blocking  (optional, additive)
# ---------------------------------------------------------------------------


def build_token_index(
    records: Iterable[RecordDict],
    min_token_len: int = 3,
    stopwords: Optional[set[str]] = None,
) -> BlockIndex:
    """Build an index mapping individual name tokens -> [entity_ids].

    Entities sharing at least one significant token (word) are candidates.
    This catches rearrangements like "Global Tech Inc" vs "Tech Global Inc".

    Args:
        records: Iterable of record dicts.
        min_token_len: Minimum token length to be indexed (filters noise).
        stopwords: Optional set of tokens to ignore (e.g. {'the', 'and'}).

    Returns:
        Dict mapping token -> list of entity_ids.
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    index: BlockIndex = defaultdict(list)
    for rec in records:
        eid = str(rec["entity_id"])
        name = normalize_business_name(rec.get("business_name"))
        if name is None:
            continue
        tokens = _tokenize(name, min_token_len=min_token_len, stopwords=stopwords)
        for token in tokens:
            index[token].append(eid)
    return dict(index)


def get_token_candidates(
    index: BlockIndex,
    record: RecordDict,
    min_token_len: int = 3,
    stopwords: Optional[set[str]] = None,
) -> set[str]:
    """Return entity_ids sharing at least one name token with *record*.

    Args:
        index: Output of :func:`build_token_index`.
        record: Query record dict.
        min_token_len: Must match the value used when building the index.
        stopwords: Must match the value used when building the index.

    Returns:
        Set of candidate entity_ids.
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    name = normalize_business_name(record.get("business_name"))
    if name is None:
        return set()

    tokens = _tokenize(name, min_token_len=min_token_len, stopwords=stopwords)
    candidates: set[str] = set()
    for token in tokens:
        candidates.update(index.get(token, []))
    candidates.discard(str(record.get("entity_id", "")))
    return candidates


# ---------------------------------------------------------------------------
# Combined blocking: generate candidate pairs
# ---------------------------------------------------------------------------


def generate_candidate_pairs(
    records: list[RecordDict],
    *,
    use_country: bool = True,
    use_exact_name: bool = True,
    use_prefix: bool = False,
    prefix_len: int = 5,
    use_tokens: bool = False,
    min_token_len: int = 3,
) -> Generator[tuple[str, str], None, None]:
    """Yield (entity_id_a, entity_id_b) candidate pairs using the chosen strategies.

    Pairs are deduplicated (each unordered pair appears exactly once) and
    entity self-pairs are excluded.

    This function is designed for *medium-scale* use (hundreds of thousands
    of records).  For the full 2.52 GB dataset, build indexes incrementally
    via the individual ``build_*_index`` functions and intersect keys.

    Args:
        records: List of record dicts (in-memory).
        use_country: If True, restrict candidates to same-country pairs.
        use_exact_name: If True, include pairs with identical normalized names.
        use_prefix: If True, include pairs sharing a name prefix.
        prefix_len: Prefix length for prefix blocking.
        use_tokens: If True, include pairs sharing at least one name token.
        min_token_len: Minimum token length for token blocking.

    Yields:
        Ordered ``(id_a, id_b)`` tuples where id_a < id_b (lexicographic).
    """
    # Build all requested indexes in one pass over the records.
    country_idx: Optional[BlockIndex] = build_country_index(records) if use_country else None
    name_idx:    Optional[BlockIndex] = build_name_index(records)    if use_exact_name else None
    prefix_idx:  Optional[BlockIndex] = build_prefix_index(records, prefix_len) if use_prefix else None
    token_idx:   Optional[BlockIndex] = build_token_index(records, min_token_len) if use_tokens else None

    seen: set[tuple[str, str]] = set()

    def _emit(a: str, b: str) -> Optional[tuple[str, str]]:
        pair = (a, b) if a < b else (b, a)
        if pair not in seen:
            seen.add(pair)
            return pair
        return None

    for rec in records:
        eid = str(rec["entity_id"])

        # Collect candidates from all active strategies.
        candidates: set[str] = set()

        if country_idx is not None:
            country_cands = get_country_candidates(country_idx, rec)
        else:
            # If not using country blocking, everyone is a potential candidate.
            country_cands = {str(r["entity_id"]) for r in records} - {eid}

        if name_idx is not None:
            name_cands = get_name_candidates(name_idx, rec) & country_cands
            candidates |= name_cands

        if prefix_idx is not None:
            candidates |= get_prefix_candidates(prefix_idx, rec, prefix_len) & country_cands

        if token_idx is not None:
            candidates |= get_token_candidates(token_idx, rec, min_token_len) & country_cands

        for cid in candidates:
            pair = _emit(eid, cid)
            if pair is not None:
                yield pair


# ---------------------------------------------------------------------------
# Pair-generation from a pre-built index (streaming, low memory)
# ---------------------------------------------------------------------------


def pairs_from_index(index: BlockIndex) -> Generator[tuple[str, str], None, None]:
    """Yield all unique pairs within each block of an index.

    For large blocks this is still O(B^2) per block, so combine with country
    blocking to keep block sizes small.  A block with B members yields
    B*(B-1)/2 pairs.

    Args:
        index: Any ``BlockIndex`` (name, prefix, token, ...).

    Yields:
        Unique ``(id_a, id_b)`` pairs where id_a < id_b.
    """
    for key, eids in index.items():
        if len(eids) < 2:
            continue
        # Deduplicate entity IDs within this block first.
        unique_ids = sorted(set(eids))
        for a, b in combinations(unique_ids, 2):
            yield (a, b)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_STOPWORDS: set[str] = {
    "the", "and", "of", "a", "an", "in", "at", "for", "to", "by",
    "inc", "llc", "ltd", "corp", "co", "pvt", "pte", "lp", "llp",
}


def _tokenize(
    text: str,
    min_token_len: int = 3,
    stopwords: Optional[set[str]] = None,
) -> list[str]:
    """Split *text* into significant tokens.

    Args:
        text: Already-normalized (lowercased) text.
        min_token_len: Minimum number of characters for a token to be kept.
        stopwords: Tokens to exclude.

    Returns:
        List of unique tokens.
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS
    tokens = text.split()
    return list({
        t for t in tokens
        if len(t) >= min_token_len and t not in stopwords
    })
