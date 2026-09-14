from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def normalized_auprc(auprc: float, prevalence: float) -> float:
    """Normalize AUPRC above its random-classifier baseline."""
    auprc = float(auprc)
    prevalence = float(prevalence)
    if not np.isfinite(auprc) or not np.isfinite(prevalence):
        return float("nan")
    denominator = max(1.0 - prevalence, 1e-12)
    return float(np.clip((auprc - prevalence) / denominator, 0.0, 1.0))


def dual_metric_score(
    auroc: float,
    auprc: float,
    prevalence: float,
    auroc_weight: float = 0.60,
    auprc_weight: float = 0.40,
) -> float:
    normalized_pr = normalized_auprc(auprc, prevalence)
    if not np.isfinite(auroc) or not np.isfinite(normalized_pr):
        return float("-inf")
    return float(auroc_weight * float(auroc) + auprc_weight * normalized_pr)


def binary_ranking_metrics(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(
        average_precision_score(labels, scores)
    )


def checkpoint_passes_direct_guard(
    auroc: float,
    auprc: float,
    best_direct_auroc: float,
    best_direct_auprc: float,
    tolerance_auroc: float = 0.001,
    tolerance_auprc: float = 0.002,
) -> bool:
    return (
        float(auroc) >= float(best_direct_auroc) - float(tolerance_auroc)
        and float(auprc) >= float(best_direct_auprc) - float(tolerance_auprc)
    )


def metric_sort_key(candidate: Mapping[str, float]):
    """Dual score first, then AUROC, AUPRC, and lower NLL."""
    return (
        float(candidate["score"]),
        float(candidate["auroc"]),
        float(candidate["auprc"]),
        -float(candidate.get("nll", float("inf"))),
    )


def select_best_candidate(
    candidates: Iterable[Mapping[str, float]],
    *,
    best_direct_auroc: float | None = None,
    best_direct_auprc: float | None = None,
    tolerance_auroc: float = 0.001,
    tolerance_auprc: float = 0.002,
):
    rows = list(candidates)
    if best_direct_auroc is not None and best_direct_auprc is not None:
        guarded = [
            row
            for row in rows
            if checkpoint_passes_direct_guard(
                row["auroc"],
                row["auprc"],
                best_direct_auroc,
                best_direct_auprc,
                tolerance_auroc,
                tolerance_auprc,
            )
        ]
        if guarded:
            rows = guarded
    if not rows:
        return None
    return max(rows, key=metric_sort_key)


def _sigmoid(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def select_reason_threshold(
    reason_probabilities,
    direct_logits,
    reasoned_logits,
    labels,
    *,
    split: str,
    candidates: Sequence[float] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    min_reason_rate: float = 0.10,
    max_reason_rate: float = 0.90,
):
    """Select a routing threshold using validation labels only."""
    if split != "val":
        raise ValueError("reason threshold selection is validation-only")
    reason_probabilities = np.asarray(reason_probabilities).reshape(-1)
    direct_logits = np.asarray(direct_logits).reshape(-1)
    reasoned_logits = np.asarray(reasoned_logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    prevalence = float(np.mean(labels))
    rows = []
    for threshold in candidates:
        routed = reason_probabilities >= float(threshold)
        logits = np.where(routed, reasoned_logits, direct_logits)
        auroc, auprc = binary_ranking_metrics(labels, logits)
        probability = _sigmoid(logits)
        nll = float(log_loss(labels, probability, labels=[0, 1]))
        rows.append(
            {
                "threshold": float(threshold),
                "reason_rate": float(np.mean(routed)),
                "auroc": auroc,
                "auprc": auprc,
                "score": dual_metric_score(auroc, auprc, prevalence),
                "nll": nll,
            }
        )
    feasible = [
        row
        for row in rows
        if min_reason_rate <= row["reason_rate"] <= max_reason_rate
    ]
    selected = max(feasible or rows, key=metric_sort_key)
    return dict(selected), rows


DEFAULT_ENSEMBLE_WEIGHTS = (
    (1 / 3, 1 / 3, 1 / 3),
    (0.5, 0.3, 0.2),
    (0.6, 0.2, 0.2),
    (0.4, 0.4, 0.2),
)


def select_ensemble_weights(
    checkpoint_logits,
    labels,
    *,
    split: str,
    weight_candidates: Sequence[Sequence[float]] = DEFAULT_ENSEMBLE_WEIGHTS,
):
    """Grid-search logit ensemble weights on validation predictions only."""
    if split != "val":
        raise ValueError("ensemble weight selection is validation-only")
    logits = np.asarray(checkpoint_logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    labels = np.asarray(labels).reshape(-1)
    if logits.shape[0] != 3:
        raise ValueError("v5 ensemble expects exactly three validation checkpoints")
    if logits.shape[1] != labels.size:
        raise ValueError("ensemble logits and labels have inconsistent lengths")
    prevalence = float(np.mean(labels))
    rows = []
    for values in weight_candidates:
        weights = np.asarray(values, dtype=np.float64)
        if weights.shape != (3,) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("ensemble weights must be three non-negative values")
        weights = weights / weights.sum()
        combined = np.sum(weights[:, None] * logits, axis=0)
        auroc, auprc = binary_ranking_metrics(labels, combined)
        rows.append(
            {
                "weights": weights.tolist(),
                "auroc": auroc,
                "auprc": auprc,
                "score": dual_metric_score(auroc, auprc, prevalence),
                "nll": float(log_loss(labels, _sigmoid(combined), labels=[0, 1])),
            }
        )
    selected = max(rows, key=metric_sort_key)
    return dict(selected), rows


@dataclass(frozen=True)
class PerformanceGuardResult:
    save: bool
    both_improved: bool
    tradeoff_warning: bool


def performance_guard(final_metrics, direct_metrics) -> PerformanceGuardResult:
    score_improved = float(final_metrics["score"]) > float(direct_metrics["score"])
    auroc_improved = float(final_metrics["auroc"]) >= float(direct_metrics["auroc"])
    auprc_improved = float(final_metrics["auprc"]) >= float(direct_metrics["auprc"])
    both = auroc_improved and auprc_improved
    return PerformanceGuardResult(
        save=score_improved,
        both_improved=both,
        tradeoff_warning=score_improved and not both,
    )
