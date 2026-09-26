"""
utils.py
========
Reusable helper utilities for the Amazon ML Challenge pipeline.

Contents
--------
* Logging setup
* Progress reporting (tqdm-aware, gracefully degrades without tqdm)
* File discovery
* Schema / record validation
* Pair-count estimation
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, TypeVar

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | os.PathLike[str]] = None,
) -> logging.Logger:
    """Configure root logger with console (and optional file) handlers.

    Safe to call multiple times – existing handlers are replaced.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``).
        log_file: Optional path to write logs to in addition to stderr.

    Returns:
        Root ``logging.Logger`` instance.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing handlers to avoid duplication.
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stderr).
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler.
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (convenience wrapper)."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

T = TypeVar("T")

try:
    from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def progress(
    iterable: Iterable[T],
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "it",
    disable: bool = False,
) -> Iterator[T]:
    """Wrap an iterable with a progress bar (tqdm if available, else plain).

    Args:
        iterable: Any iterable to wrap.
        desc: Description label shown on the left of the progress bar.
        total: Optional total item count (enables ETA calculation).
        unit: Label for each item (e.g. 'rows', 'pairs').
        disable: When True, pass through without wrapping.

    Yields:
        Items from *iterable* unchanged.
    """
    if disable:
        yield from iterable
        return

    if _HAS_TQDM:
        yield from _tqdm(iterable, desc=desc, total=total, unit=unit)
    else:
        # Minimal fallback: log progress every N items.
        log = logging.getLogger("progress")
        log.info("Starting: %s", desc or "iteration")
        start = time.monotonic()
        count = 0
        report_every = max(1, (total or 10_000) // 10)
        for item in iterable:
            yield item
            count += 1
            if count % report_every == 0:
                elapsed = time.monotonic() - start
                rate = count / elapsed if elapsed > 0 else 0
                log.info(
                    "%s: %d%s [%.1f items/s]",
                    desc or "Progress",
                    count,
                    f"/{total}" if total else "",
                    rate,
                )
        elapsed = time.monotonic() - start
        log.info("%s: done (%d items in %.1f s)", desc or "Finished", count, elapsed)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_files(
    directory: str | os.PathLike[str],
    pattern: str = "*.tsv",
    recursive: bool = False,
) -> list[Path]:
    """Return a sorted list of files matching *pattern* in *directory*.

    Args:
        directory: Root directory to search.
        pattern: Glob pattern (e.g. ``*.tsv``, ``source*.tsv``).
        recursive: If True, search subdirectories as well.

    Returns:
        Sorted list of matching ``pathlib.Path`` objects.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if recursive:
        return sorted(directory.rglob(pattern))
    return sorted(directory.glob(pattern))


def file_size_mb(filepath: str | os.PathLike[str]) -> float:
    """Return the file size in megabytes."""
    return Path(filepath).stat().st_size / (1024 ** 2)


def describe_file(filepath: str | os.PathLike[str]) -> str:
    """Return a human-readable description of a file (name + size)."""
    p = Path(filepath)
    return f"{p.name} ({file_size_mb(p):.1f} MB)"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)

_SOURCE_ID_PREFIXES: tuple[str, ...] = ("S1-", "S2-", "S3-")


def validate_record(record: dict[str, Any]) -> list[str]:
    """Validate a single record dict against the expected schema.

    Args:
        record: Dict with field names as keys.

    Returns:
        List of validation error messages.  Empty list means the record is
        valid.
    """
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: '{field}'")

    eid = record.get("entity_id", "")
    if eid and not any(str(eid).startswith(p) for p in _SOURCE_ID_PREFIXES):
        errors.append(
            f"entity_id '{eid}' does not match expected prefixes "
            f"({', '.join(_SOURCE_ID_PREFIXES)})"
        )

    return errors


def validate_dataframe(df: Any) -> dict[str, Any]:
    """Run basic validation on a pandas DataFrame chunk.

    Args:
        df: pandas.DataFrame with columns corresponding to the source schema.

    Returns:
        Dict with validation summary:
        - ``n_rows``: total rows in the chunk.
        - ``missing_per_column``: dict column -> count of missing values.
        - ``unknown_id_prefixes``: count of entity_ids with unexpected prefix.
        - ``errors``: list of error messages.
    """
    import pandas as pd  # local import to keep utils.py pandas-optional

    errors: list[str] = []
    missing_per_column: dict[str, int] = {}

    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            errors.append(f"Column '{field}' not found in DataFrame")
        else:
            n_missing = int(df[field].isna().sum())
            if n_missing > 0:
                missing_per_column[field] = n_missing

    # Check entity_id prefix format.
    unknown_prefix_count = 0
    if "entity_id" in df.columns:
        mask = ~df["entity_id"].astype(str).str.startswith(
            _SOURCE_ID_PREFIXES  # type: ignore[arg-type]
        )
        unknown_prefix_count = int(mask.sum())
        if unknown_prefix_count > 0:
            errors.append(
                f"{unknown_prefix_count} entity_ids have unexpected prefixes"
            )

    return {
        "n_rows": len(df),
        "missing_per_column": missing_per_column,
        "unknown_id_prefixes": unknown_prefix_count,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Pair-count estimation
# ---------------------------------------------------------------------------


def estimate_pairs(n: int) -> int:
    """Return the maximum number of pairs from *n* entities (all-pairs).

    Use this to sanity-check blocking effectiveness:
    ``actual_pairs << estimate_pairs(n)`` means blocking is working.

    Args:
        n: Number of entities.

    Returns:
        n * (n - 1) // 2
    """
    return n * (n - 1) // 2


def blocking_reduction_ratio(n_total: int, n_candidate_pairs: int) -> float:
    """Return the fraction of pairs eliminated by blocking.

    A value close to 1.0 means blocking is very aggressive (few pairs kept).
    A value of 0.0 means all pairs are kept (no reduction).

    Args:
        n_total: Total number of entities.
        n_candidate_pairs: Number of candidate pairs generated by blocking.

    Returns:
        Reduction ratio in [0, 1].
    """
    max_pairs = estimate_pairs(n_total)
    if max_pairs == 0:
        return 1.0
    kept = min(n_candidate_pairs, max_pairs)
    return 1.0 - (kept / max_pairs)


# ---------------------------------------------------------------------------
# Dataset Path Resolution
# ---------------------------------------------------------------------------


def resolve_dataset_paths(
    project_root: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Path]:
    """Resolve and return verified paths to the raw dataset TSV files.

    Resolution precedence:
    1. Environment variable ``DATASET_DIR`` or ``AMAZON_ML_DATASET_DIR``.
    2. Configuration file ``config/dataset_config.json``.
    3. Standard candidate paths relative to project root or user profile.

    Returns:
        Dict with keys: 'dataset_dir', 'train_source1', 'train_source2',
        'train_source3', 'train_ground_truth', each mapping to a verified Path.

    Raises:
        FileNotFoundError: If the dataset files cannot be located.
    """
    import json

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root).resolve()

    candidate_dirs: list[Path] = []

    # 1. Environment variables
    for env_var in ("DATASET_DIR", "AMAZON_ML_DATASET_DIR"):
        val = os.environ.get(env_var)
        if val:
            candidate_dirs.append(Path(val))

    # 2. Config file
    config_file = project_root / "config" / "dataset_config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "dataset_dir" in cfg:
                candidate_dirs.append(Path(cfg["dataset_dir"]))
        except Exception:
            pass

    # 3. Standard paths
    candidate_dirs.extend([
        project_root / "dataset",
        Path(r"C:\Users\somes\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"),
        Path(r"D:\amazon-ml-challenge\amazon-ml-challenge\dataset"),
    ])

    for base_dir in candidate_dirs:
        train_dir = base_dir / "train" if (base_dir / "train").exists() else base_dir
        s1 = train_dir / "train_source1.tsv"
        s2 = train_dir / "train_source2.tsv"
        s3 = train_dir / "train_source3.tsv"
        gt = train_dir / "train_ground_truth.tsv"

        if s1.exists() and s2.exists() and s3.exists() and gt.exists():
            return {
                "dataset_dir": base_dir,
                "train_source1": s1,
                "train_source2": s2,
                "train_source3": s3,
                "train_ground_truth": gt,
            }

    raise FileNotFoundError(
        "Could not resolve raw dataset directory. Looked in: "
        + ", ".join(str(d) for d in candidate_dirs)
    )

