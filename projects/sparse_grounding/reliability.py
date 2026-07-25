"""Reliability signals and diagnostics for generated-view evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def hypothesis_variance(values: Any) -> dict[str, float | int]:
    """Summarize population variance across generated hypotheses."""
    array = _finite_array(values, "values")
    if array.ndim < 1 or array.shape[0] < 2:
        raise ValueError("values must contain at least two hypotheses")
    per_element = np.var(array, axis=0)
    return {
        "mean_variance": float(np.mean(per_element)),
        "p95_variance": float(np.percentile(per_element, 95)),
        "max_variance": float(np.max(per_element)),
        "element_count": int(per_element.size),
        "hypothesis_count": int(array.shape[0]),
    }


def depth_hypothesis_variance(
    depths: Any,
    *,
    valid_min: float = 0.0,
) -> dict[str, float | int]:
    """Summarize depth variance where every hypothesis has valid depth."""
    array = np.asarray(depths, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("depths must contain at least two depth maps")
    valid = np.isfinite(array) & (array > valid_min)
    common = np.all(valid, axis=0)
    if not np.any(common):
        raise ValueError("depth hypotheses have no common valid elements")
    selected = array[:, common]
    summary = hypothesis_variance(selected)
    summary["valid_element_count"] = int(np.count_nonzero(common))
    summary["valid_fraction"] = float(np.mean(common))
    return summary


def cycle_reprojection_error(
    prediction: Any,
    reference: Any,
    *,
    mask: Any | None = None,
) -> dict[str, float | int]:
    """Compute finite masked L1/RMSE reprojection errors."""
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.size == 0:
        raise ValueError("prediction and reference must have equal nonempty shape")
    valid = np.isfinite(predicted) & np.isfinite(expected)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.shape != predicted.shape:
            try:
                selected = np.broadcast_to(selected, predicted.shape)
            except ValueError as error:
                raise ValueError("mask is not broadcastable to inputs") from error
        valid &= selected.astype(bool)
    if not np.any(valid):
        raise ValueError("cycle error has no valid elements")
    residual = predicted[valid] - expected[valid]
    absolute = np.abs(residual)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "valid_element_count": int(residual.size),
    }


def real_support_ratio(
    real_weights: Any,
    generated_weights: Any,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return opacity/confidence mass supported by real observations."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    real = _finite_array(real_weights, "real_weights")
    generated = _finite_array(generated_weights, "generated_weights")
    if np.any(real < 0) or np.any(generated < 0):
        raise ValueError("support weights must be nonnegative")
    real_mass = float(np.sum(real))
    total = real_mass + float(np.sum(generated))
    if total <= epsilon:
        return 0.0
    return real_mass / (total + epsilon)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(first: Any, second: Any) -> float:
    """Compute Spearman rho with average ranks for ties."""
    left = _finite_array(first, "first").reshape(-1)
    right = _finite_array(second, "second").reshape(-1)
    if left.shape != right.shape or len(left) < 2:
        raise ValueError("inputs must have equal length of at least two")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - np.mean(left_ranks)
    right_centered = right_ranks - np.mean(right_ranks)
    denominator = float(
        np.sqrt(
            np.sum(np.square(left_centered))
            * np.sum(np.square(right_centered))
        )
    )
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for constant input")
    return float(np.sum(left_centered * right_centered) / denominator)


def binary_auroc(labels: Any, scores: Any) -> float:
    """Compute tie-aware binary AUROC via the Mann-Whitney statistic."""
    target = np.asarray(labels)
    score = _finite_array(scores, "scores").reshape(-1)
    if target.ndim != 1 or target.shape != score.shape:
        raise ValueError("labels and scores must be equal one-dimensional arrays")
    if not np.all(np.isin(target, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    target = target.astype(bool)
    positives = int(np.count_nonzero(target))
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ranks = _average_ranks(score) + 1.0
    positive_rank_sum = float(np.sum(ranks[target]))
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return statistic / (positives * negatives)


def expected_calibration_error(
    labels: Any,
    probabilities: Any,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Compute fixed-width binary ECE and retain auditable bin statistics."""
    target = np.asarray(labels)
    probability = _finite_array(probabilities, "probabilities").reshape(-1)
    if target.ndim != 1 or target.shape != probability.shape:
        raise ValueError("labels and probabilities must have equal 1D shape")
    if not np.all(np.isin(target, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    target = target.astype(np.float64)
    indices = np.minimum((probability * bin_count).astype(int), bin_count - 1)
    bins = []
    ece = 0.0
    for index in range(bin_count):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        confidence = float(np.mean(probability[selected]))
        accuracy = float(np.mean(target[selected]))
        contribution = count / len(target) * abs(confidence - accuracy)
        ece += contribution
        bins.append(
            {
                "bin_index": index,
                "lower": index / bin_count,
                "upper": (index + 1) / bin_count,
                "count": count,
                "mean_confidence": confidence,
                "positive_rate": accuracy,
                "ece_contribution": contribution,
            }
        )
    return {"ece": ece, "bin_count": bin_count, "bins": bins}


def reliability_binned_gain(
    reliability_scores: Any,
    grounding_gains: Any,
    *,
    bin_count: int = 5,
) -> list[dict[str, float | int]]:
    """Create equal-count bins ordered from least to most reliable."""
    scores = _finite_array(reliability_scores, "reliability_scores").reshape(-1)
    gains = _finite_array(grounding_gains, "grounding_gains").reshape(-1)
    if scores.shape != gains.shape:
        raise ValueError("reliability scores and gains must have equal shape")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    order = np.argsort(scores, kind="mergesort")
    groups = np.array_split(order, min(bin_count, len(order)))
    return [
        {
            "bin_index": index,
            "count": int(len(group)),
            "score_min": float(np.min(scores[group])),
            "score_max": float(np.max(scores[group])),
            "score_mean": float(np.mean(scores[group])),
            "gain_mean": float(np.mean(gains[group])),
            "beneficial_rate": float(np.mean(gains[group] > 0)),
        }
        for index, group in enumerate(groups)
    ]


def build_reliability_report(
    records: Sequence[Mapping[str, Any]],
    *,
    score_field: str,
    gain_field: str,
    higher_is_reliable: bool = True,
    probability_field: str | None = None,
    bin_count: int = 5,
) -> dict[str, Any]:
    """Build a diagnostic report from paired view-level reliability records."""
    if not records:
        raise ValueError("records cannot be empty")
    try:
        raw_scores = [record[score_field] for record in records]
        gains = [record[gain_field] for record in records]
    except KeyError as error:
        raise ValueError(f"record is missing field {error.args[0]!r}") from error
    scores = _finite_array(raw_scores, score_field).reshape(-1)
    gains_array = _finite_array(gains, gain_field).reshape(-1)
    reliability_scores = scores if higher_is_reliable else -scores
    labels = gains_array > 0
    report = {
        "schema_version": "1.0",
        "record_count": len(records),
        "score_field": score_field,
        "gain_field": gain_field,
        "higher_is_reliable": higher_is_reliable,
        "beneficial_count": int(np.count_nonzero(labels)),
        "harmful_or_neutral_count": int(len(labels) - np.count_nonzero(labels)),
        "spearman_gain": spearman_correlation(reliability_scores, gains_array),
        "beneficial_auroc": binary_auroc(labels, reliability_scores),
        "reliability_binned_gain": reliability_binned_gain(
            reliability_scores,
            gains_array,
            bin_count=bin_count,
        ),
    }
    if probability_field is not None:
        try:
            probabilities = [record[probability_field] for record in records]
        except KeyError as error:
            raise ValueError(
                f"record is missing field {error.args[0]!r}"
            ) from error
        report["calibration"] = expected_calibration_error(
            labels,
            probabilities,
            bin_count=bin_count,
        )
        report["probability_field"] = probability_field
    return report


def load_record_list(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(record, Mapping) for record in value
    ):
        raise ValueError("input must be a JSON list of objects")
    return value
