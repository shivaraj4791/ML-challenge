"""
data_loader.py
==============
Memory-efficient TSV loading for the Amazon ML Challenge (2.52 GB dataset).

Key design choices
------------------
* Never loads the entire dataset into RAM.
* Uses pandas read_csv with chunksize so each chunk is processed and discarded.
* Exposes generators / iterators throughout -- callers pull one chunk at a time.
* Column selection is supported so unused columns (if any are added later) are
  never even decoded from disk.
* Dtype overrides ensure entity_id stays as str (not parsed as int/float).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Canonical column names in every source TSV.
REQUIRED_COLUMNS: list[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

#: Dtypes that must be preserved as strings (never coerced to numeric).
_STRING_DTYPES: dict[str, str] = {
    "entity_id": "str",
    "business_name": "str",
    "business_address": "str",
    "country": "str",
}

#: Default chunk size (rows).  Adjust for available RAM.
DEFAULT_CHUNKSIZE: int = 50_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_chunks(
    filepath: str | os.PathLike[str],
    columns: Optional[Sequence[str]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    sep: str = "\t",
    encoding: str = "utf-8",
) -> Generator[pd.DataFrame, None, None]:
    """Yield successive DataFrame chunks from a TSV file.

    This is the *primary* entry point for reading source files.  It streams
    the file so that peak RAM usage is bounded by ``chunksize`` rows, not the
    total file size.

    Args:
        filepath: Path to the TSV file.
        columns: Optional list of column names to keep.  If None, all columns
            are returned.  Use this to avoid decoding unnecessary columns.
        chunksize: Number of rows per chunk.  Tune this to your RAM budget:
            - 50,000 rows ~= 20 MB RAM for this schema.
            - 200,000 rows ~= 80 MB RAM.
        sep: Field separator (default tab).
        encoding: File encoding (default UTF-8 to handle multilingual text).

    Yields:
        pandas.DataFrame chunks.  Each chunk has the selected columns with
        string dtypes for ID fields.

    Example::

        for chunk in iter_chunks("data/source1.tsv", columns=["entity_id", "business_name"]):
            process(chunk)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TSV file not found: {filepath}")

    # Resolve usecols once so pandas skips unwanted columns at parse time.
    usecols: Optional[list[str]] = list(columns) if columns is not None else None

    # Determine which string-dtype overrides are relevant for the requested cols.
    dtype_map = {
        col: _STRING_DTYPES[col]
        for col in (usecols or REQUIRED_COLUMNS)
        if col in _STRING_DTYPES
    }

    reader = pd.read_csv(
        filepath,
        sep=sep,
        encoding=encoding,
        chunksize=chunksize,
        usecols=usecols,
        dtype=dtype_map,
        keep_default_na=True,   # keep NaN for missing addresses
        on_bad_lines="warn",    # log malformed rows, don't crash
    )

    for chunk in reader:
        yield chunk


def iter_rows(
    filepath: str | os.PathLike[str],
    columns: Optional[Sequence[str]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    sep: str = "\t",
    encoding: str = "utf-8",
) -> Generator[dict, None, None]:
    """Yield individual row dicts from a TSV file (lower memory per item).

    Wraps :func:`iter_chunks` and iterates each chunk row-by-row.  Useful
    when downstream processing is row-oriented (e.g., building a dict index).

    Args:
        filepath: Path to the TSV file.
        columns: Optional column subset.
        chunksize: Rows per internal chunk.
        sep: Field separator.
        encoding: File encoding.

    Yields:
        dict mapping column name -> raw value for each row.
    """
    for chunk in iter_chunks(filepath, columns=columns, chunksize=chunksize,
                              sep=sep, encoding=encoding):
        for row in chunk.to_dict(orient="records"):
            yield row


def load_full(
    filepath: str | os.PathLike[str],
    columns: Optional[Sequence[str]] = None,
    sep: str = "\t",
    encoding: str = "utf-8",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a complete TSV into a single DataFrame.

    .. warning::
        For the 2.52 GB production dataset this will consume several GB of
        RAM.  Use :func:`iter_chunks` in production pipelines.  This function
        is intended for:

        * Unit tests with synthetic small files.
        * Loading ground-truth / label files (which are much smaller).
        * Interactive exploration of small samples.

    Args:
        filepath: Path to the TSV file.
        columns: Optional column subset.
        sep: Field separator.
        encoding: File encoding.
        max_rows: Optional row limit (useful for sampling / debugging).

    Returns:
        pandas.DataFrame with the full file content.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TSV file not found: {filepath}")

    usecols: Optional[list[str]] = list(columns) if columns is not None else None
    dtype_map = {
        col: _STRING_DTYPES[col]
        for col in (usecols or REQUIRED_COLUMNS)
        if col in _STRING_DTYPES
    }

    return pd.read_csv(
        filepath,
        sep=sep,
        encoding=encoding,
        usecols=usecols,
        dtype=dtype_map,
        keep_default_na=True,
        on_bad_lines="warn",
        nrows=max_rows,
    )


def stream_source_files(
    source_dir: str | os.PathLike[str],
    pattern: str = "*.tsv",
    columns: Optional[Sequence[str]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Generator[tuple[str, pd.DataFrame], None, None]:
    """Stream chunks from all matching files in a directory.

    Useful for iterating over source1.tsv, source2.tsv, source3.tsv without
    loading them all at once.

    Args:
        source_dir: Directory containing source TSV files.
        pattern: Glob pattern to match files (default ``*.tsv``).
        columns: Optional column subset.
        chunksize: Rows per chunk.

    Yields:
        ``(filename, chunk_dataframe)`` tuples.
    """
    source_dir = Path(source_dir)
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {source_dir}"
        )
    for filepath in files:
        for chunk in iter_chunks(filepath, columns=columns, chunksize=chunksize):
            yield filepath.name, chunk


def sample_file(
    filepath: str | os.PathLike[str],
    n: int = 1000,
    columns: Optional[Sequence[str]] = None,
    sep: str = "\t",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Return the first *n* rows of a TSV for quick inspection.

    Reads only *n* rows, so this is safe even for the 2.5 GB dataset.

    Args:
        filepath: Path to the TSV file.
        n: Number of rows to return.
        columns: Optional column subset.
        sep: Field separator.
        encoding: File encoding.

    Returns:
        pandas.DataFrame with at most *n* rows.
    """
    return load_full(filepath, columns=columns, sep=sep,
                     encoding=encoding, max_rows=n)
