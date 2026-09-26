"""
evaluation.py
=============
Reusable evaluation utilities for the Amazon ML Challenge entity-resolution
pipeline.

This module is intentionally decoupled from any specific model.  It accepts
lists of scores/probabilities and ground-truth labels and computes standard
IR/binary-classification metrics.

Functions
---------
confusion_counts        -- tp, fp, fn, tn for a single threshold
precision               -- tp / (tp + fp)
recall                  -- tp / (tp + fn)
f1_score                -- 2 * P * R / (P + R)
evaluate_at_threshold   -- all metrics for one threshold
evaluate_thresholds     -- table over a sweep of thresholds
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Label = int   # 1 = match, 0 = non-match
Score = float  # predicted probability or similarity score in [0, 1]


# ---------------------------------------------------------------------------
# Low-level confusion counts
# ---------------------------------------------------------------------------


def confusion_counts(
    y_true: Sequence[Label],
    y_scores: Sequence[Score],
    threshold: float,
) -> dict[str, int]:
    """Compute TP, FP, FN, TN for a given decision threshold.

    A pair is predicted *positive* (match) when ``score >= threshold``.

    Args:
        y_true:    Ground-truth labels (1 = match, 0 = non-match).
        y_scores:  Predicted scores / probabilities in [0, 1].
        threshold: Decision cutoff.  Predictions >= threshold are positive.

    Returns:
        Dict with keys 'tp', 'fp', 'fn', 'tn'.

    Raises:
        ValueError: If y_true and y_scores have different lengths.
    """
    if len(y_true) != len(y_scores):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_scores ({len(y_scores)}) "
            "must have the same length."
        )
    tp = fp = fn = tn = 0
    for label, score in zip(y_true, y_scores):
        pred = int(score >= threshold)
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP).

    Returns 0.0 when TP + FP == 0 (no positive predictions).

    Args:
        tp: True positives.
        fp: False positives.

    Returns:
        Precision in [0, 1].
    """
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN).

    Returns 0.0 when TP + FN == 0 (no actual positives).

    Args:
        tp: True positives.
        fn: False negatives.

    Returns:
        Recall in [0, 1].
    """
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(p: float, r: float) -> float:
    """F1 = 2 * P * R / (P + R).

    Returns 0.0 when P + R == 0.

    Args:
        p: Precision.
        r: Recall.

    Returns:
        F1 score in [0, 1].
    """
    denom = p + r
    return 2 * p * r / denom if denom > 0 else 0.0


def f_beta_score(p: float, r: float, beta: float = 1.0) -> float:
    """F-beta score = (1 + beta^2) * P * R / (beta^2 * P + R).

    F1 is the special case where beta=1.  Use beta < 1 to weight precision
    higher; beta > 1 to weight recall higher.

    Args:
        p:    Precision.
        r:    Recall.
        beta: Beta parameter (default 1.0 == F1).

    Returns:
        F-beta score in [0, 1].
    """
    b2 = beta ** 2
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Single-threshold evaluation
# ---------------------------------------------------------------------------


def evaluate_at_threshold(
    y_true: Sequence[Label],
    y_scores: Sequence[Score],
    threshold: float,
    beta: float = 1.0,
) -> dict[str, float]:
    """Compute all metrics for a single decision threshold.

    Args:
        y_true:    Ground-truth labels.
        y_scores:  Predicted scores.
        threshold: Decision cutoff.
        beta:      F-beta parameter (default 1.0 = F1).

    Returns:
        Dict with keys:
            threshold, precision, recall, f1, f_beta, tp, fp, fn, tn,
            n_predicted_positive, n_actual_positive, support
    """
    counts = confusion_counts(y_true, y_scores, threshold)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    p = precision(tp, fp)
    r = recall(tp, fn)
    f1 = f1_score(p, r)
    fb = f_beta_score(p, r, beta=beta)
    return {
        "threshold":            threshold,
        "precision":            round(p, 6),
        "recall":               round(r, 6),
        "f1":                   round(f1, 6),
        "f_beta":               round(fb, 6),
        "tp":                   tp,
        "fp":                   fp,
        "fn":                   fn,
        "tn":                   tn,
        "n_predicted_positive": tp + fp,
        "n_actual_positive":    tp + fn,
        "support":              len(y_true),
    }


# ---------------------------------------------------------------------------
# Multi-threshold sweep
# ---------------------------------------------------------------------------


def evaluate_thresholds(
    y_true: Sequence[Label],
    y_scores: Sequence[Score],
    thresholds: Optional[Sequence[float]] = None,
    beta: float = 1.0,
) -> list[dict[str, float]]:
    """Evaluate metrics across a sweep of thresholds.

    Does NOT choose or recommend a best threshold -- it just reports.
    The caller decides which threshold to use based on their cost function.

    Args:
        y_true:     Ground-truth labels.
        y_scores:   Predicted scores.
        thresholds: Thresholds to evaluate.  Defaults to 0.05 steps in
                    [0.05, 0.95].  Pass your own list for finer control.
        beta:       F-beta parameter.

    Returns:
        List of dicts (one per threshold), each with the same keys as
        :func:`evaluate_at_threshold`.  Sorted by threshold ascending.

    Example::

        rows = evaluate_thresholds(y_true, scores)
        for row in rows:
            print(f"{row['threshold']:.2f}  P={row['precision']:.3f}  "
                  f"R={row['recall']:.3f}  F1={row['f1']:.3f}")
    """
    if thresholds is None:
        thresholds = [round(t / 100, 2) for t in range(5, 100, 5)]
    return [
        evaluate_at_threshold(y_true, y_scores, t, beta=beta)
        for t in sorted(thresholds)
    ]


def threshold_table_str(rows: list[dict[str, float]]) -> str:
    """Format the output of :func:`evaluate_thresholds` as a readable table.

    Args:
        rows: List of metric dicts from ``evaluate_thresholds``.

    Returns:
        Human-readable string table.
    """
    header = (
        f"{'threshold':>9}  {'precision':>9}  {'recall':>6}  "
        f"{'f1':>6}  {'tp':>6}  {'fp':>6}  {'fn':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['threshold']:>9.3f}  {r['precision']:>9.4f}  "
            f"{r['recall']:>6.4f}  {r['f1']:>6.4f}  "
            f"{r['tp']:>6}  {r['fp']:>6}  {r['fn']:>6}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: evaluate a list of (score, label) pair predictions
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: Sequence[tuple[Score, Label]],
    threshold: float = 0.5,
    beta: float = 1.0,
) -> dict[str, float]:
    """Evaluate a list of (score, label) tuples at a single threshold.

    Args:
        predictions: Sequence of ``(score, true_label)`` tuples.
        threshold:   Decision threshold (default 0.5).
        beta:        F-beta parameter.

    Returns:
        Metrics dict (same schema as :func:`evaluate_at_threshold`).
    """
    y_scores = [p[0] for p in predictions]
    y_true   = [p[1] for p in predictions]
    return evaluate_at_threshold(y_true, y_scores, threshold, beta=beta)
