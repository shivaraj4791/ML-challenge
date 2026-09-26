"""
normalization.py
================
Reusable text-normalization functions for the Amazon ML Challenge entity-
resolution pipeline.

Design principles
-----------------
* NEVER destroy useful information.
* NEVER blindly strip non-ASCII characters – the dataset contains France
  (accented Latin), India (Devanagari / other Indic scripts), and the US.
* Keep the *raw* field values untouched; every function returns a *new*
  normalized string.
* All functions are pure (no side-effects) and accept None / NaN / empty
  strings gracefully.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Punctuation regex that targets only Unicode punctuation/symbol categories.
# Using \p{} style is not natively supported in stdlib re, so we enumerate
# the General Categories we want to remove: Pc, Pd (handled separately),
# Pe, Pf, Pi, Po, Ps, Sc, Sk, Sm, So.
# Critically, this does NOT remove letters (Ll, Lu, Lo, Lm, Lt, Nd, Nl, No)
# so Devanagari, accented Latin, etc. are all safe.
# We use a negative-lookahead to protect hyphens when keep_hyphens=True.
_PUNCT_CATEGORIES = frozenset([
    "Pc", "Pe", "Pf", "Pi", "Po", "Ps",  # punctuation (excl. Pd = dash)
    "Sc", "Sk", "Sm", "So",              # symbols
])


def _char_is_punct(c: str) -> bool:
    """Return True if *c* is a punctuation or symbol Unicode character."""
    return unicodedata.category(c) in _PUNCT_CATEGORIES


# Collapse any run of whitespace (including NBSP, tabs, zero-width spaces)
# to a single ASCII space.
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

# Legal suffixes we want to standardize so that "LLC" == "L.L.C." etc.
# Mapping: normalised canonical -> set of variants (all already lowercased).
_LEGAL_SUFFIX_MAP: dict[str, list[str]] = {
    "llc": ["l.l.c.", "l.l.c", "llc.", "limited liability company"],
    "inc": ["inc.", "incorporated", "inc,"],
    "corp": ["corp.", "corporation", "corp,"],
    "ltd": ["ltd.", "limited", "ltd,"],
    "co": ["co.", "company", "co,"],
    "lp": ["l.p.", "limited partnership"],
    "llp": ["l.l.p.", "limited liability partnership"],
    "pvt": ["pvt.", "private", "pvt,"],
    "pte": ["pte.", "private limited"],  # Singapore / India
}

# Build a flat list of (pattern, canonical) tuples – longest variants first
# to avoid partial replacement.
_suffix_patterns: list[tuple[re.Pattern[str], str]] = []
for _canonical, _variants in _LEGAL_SUFFIX_MAP.items():
    for _variant in sorted(_variants, key=len, reverse=True):
        _suffix_patterns.append(
            (
                re.compile(
                    r"(?<!\w)" + re.escape(_variant) + r"(?!\w)",
                    flags=re.IGNORECASE | re.UNICODE,
                ),
                _canonical,
            )
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_valid(value: object) -> bool:
    """Return True when value is a non-empty, non-NaN string."""
    if value is None:
        return False
    # Detect float NaN without importing numpy.
    try:
        if isinstance(value, float) and (value != value):  # NaN check
            return False
    except TypeError:
        pass
    s = str(value).strip()
    return s != "" and s.lower() not in {"nan", "none", "null", "n/a", "na", "-"}


# ---------------------------------------------------------------------------
# Low-level primitive helpers
# ---------------------------------------------------------------------------


def unicode_normalize(text: Optional[str], form: str = "NFC") -> Optional[str]:
    """Apply Unicode normalization (default NFC).

    NFC is the safest form for multilingual text: it composes characters so
    that e-acute is stored as a *single* code-point rather than e + combining
    acute.  This avoids false inequality between two representations of the
    same glyph.

    Args:
        text: Input string or None/NaN.
        form: Unicode normalization form -- NFC, NFD, NFKC, NFKD.

    Returns:
        Normalized string, or None if input is None/NaN/empty.
    """
    if not _is_valid(text):
        return None
    return unicodedata.normalize(form, str(text))


def to_lowercase(text: Optional[str]) -> Optional[str]:
    """Return a lowercased copy; handles None gracefully."""
    if not _is_valid(text):
        return None
    return str(text).lower()


def normalize_whitespace(text: Optional[str]) -> Optional[str]:
    """Collapse all whitespace runs to a single space and strip edges.

    Handles NBSP (\\u00a0), tab, zero-width space, and other Unicode spaces.
    """
    if not _is_valid(text):
        return None
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def normalize_punctuation(
    text: Optional[str], *, keep_hyphens: bool = True
) -> Optional[str]:
    """Replace punctuation with spaces (collapse, do NOT delete).

    We replace rather than delete so that 'ABC,Inc' -> 'ABC Inc' preserving
    the token boundary.  Hyphens are kept by default because they appear in
    legitimate business names (e.g. 'Hewlett-Packard').

    Implementation uses Unicode General Category checks so that:
    - Devanagari combining marks (matras) are NEVER split from their base.
    - Accented Latin characters are NEVER removed.
    - Only true punctuation/symbol code-points are replaced with spaces.

    Args:
        text: Input string.
        keep_hyphens: When True, hyphens/en-dashes/em-dashes are preserved.

    Returns:
        Punctuation-normalized string, or None.
    """
    if not _is_valid(text):
        return None
    s = str(text)
    chars: list[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == "Pd":  # dash / hyphen category
            if keep_hyphens:
                chars.append("-")
            else:
                chars.append(" ")
        elif _char_is_punct(ch):
            chars.append(" ")
        else:
            chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Missing-value safety
# ---------------------------------------------------------------------------


def safe_field(value: Optional[str], default: str = "") -> str:
    """Return a stripped string, falling back to *default* for None/NaN.

    Args:
        value: Raw field value (may be None, float NaN, or empty string).
        default: Value to return when the field is missing.

    Returns:
        Stripped string, or *default*.
    """
    if not _is_valid(value):
        return default
    cleaned = str(value).strip()
    return cleaned if cleaned else default


def is_missing(value: Optional[str]) -> bool:
    """Return True when a field is genuinely absent (None, NaN, empty)."""
    return not _is_valid(value) or str(value).strip() == ""


# ---------------------------------------------------------------------------
# Business-name normalization
# ---------------------------------------------------------------------------


def normalize_business_name(name: Optional[str]) -> Optional[str]:
    """Full normalization pipeline for business names.

    Steps (in order):
    1. Handle missing values -> None.
    2. Unicode NFC normalization.
    3. Lowercase.
    4. Normalize whitespace.
    5. Standardize common legal suffixes (LLC, Inc, Ltd ...) -- BEFORE
       punctuation stripping so dotted forms like L.L.C. are caught.
    6. Normalize punctuation (keep hyphens).
    7. Final whitespace collapse.

    The raw value is **never** modified -- this function returns a new string.

    Args:
        name: Raw business name.

    Returns:
        Normalized name string, or None if the input is missing.
    """
    if not _is_valid(name):
        return None

    s = unicode_normalize(name, form="NFC")
    s = to_lowercase(s)
    s = normalize_whitespace(s)

    # Standardize legal suffixes FIRST while dots are still present.
    # e.g. "L.L.C." -> "llc" before punctuation stripping removes the dots.
    for pattern, canonical in _suffix_patterns:
        s = pattern.sub(canonical, s)

    # Now strip remaining punctuation (hyphens preserved).
    s = normalize_punctuation(s, keep_hyphens=True)

    # Final whitespace tidy-up after suffix substitution + punctuation pass.
    s = normalize_whitespace(s)
    return s



# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

# Common address abbreviations -- expand so that "St" == "Street".
_ADDR_ABBREV: dict[str, str] = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\brd\b": "road",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bhwy\b": "highway",
    r"\bfwy\b": "freeway",
    r"\bpkwy\b": "parkway",
    r"\bfte\b": "suite",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bflr\b": "floor",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bnw\b": "northwest",
    r"\bne\b": "northeast",
    r"\bsw\b": "southwest",
    r"\bse\b": "southeast",
}
_ADDR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, flags=re.IGNORECASE | re.UNICODE), repl)
    for pat, repl in _ADDR_ABBREV.items()
]


def normalize_address(address: Optional[str]) -> Optional[str]:
    """Normalize a business address.

    Steps:
    1. Handle missing values -> None.
    2. Unicode NFC normalization (preserves Devanagari / accented Latin).
    3. Lowercase.
    4. Normalize whitespace.
    5. Normalize punctuation (keep hyphens for house-number ranges).
    6. Expand common English address abbreviations.
    7. Final whitespace collapse.

    Args:
        address: Raw address string.

    Returns:
        Normalized address, or None if the input is missing.

    Note:
        ~3.3% of Source 2/3 addresses are missing.  Callers should use
        ``is_missing()`` to detect absent addresses before calling this
        function, or simply check for a None return value.
    """
    if not _is_valid(address):
        return None

    s = unicode_normalize(address, form="NFC")
    s = to_lowercase(s)
    s = normalize_whitespace(s)
    s = normalize_punctuation(s, keep_hyphens=True)

    # Expand address abbreviations (English only; non-ASCII tokens left as-is).
    for pattern, replacement in _ADDR_PATTERNS:
        s = pattern.sub(replacement, s)

    s = normalize_whitespace(s)
    return s


# ---------------------------------------------------------------------------
# Country normalization
# ---------------------------------------------------------------------------

_COUNTRY_ALIASES: dict[str, str] = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "india": "IN",
    "in": "IN",
    "france": "FR",
    "fr": "FR",
}


def normalize_country(country: Optional[str]) -> Optional[str]:
    """Canonicalize a country field to a 2-letter ISO code.

    The dataset has 0% missing country values, but None is handled safely.

    Args:
        country: Raw country string.

    Returns:
        ISO-3166 2-letter code ('US', 'IN', 'FR'), or the stripped/uppercased
        input if not in the alias table (future-proofs new countries).
    """
    if not _is_valid(country):
        return None
    stripped = str(country).strip().lower()
    if stripped in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[stripped]
    # Unknown country: return as uppercased stripped string.
    return str(country).strip().upper()
