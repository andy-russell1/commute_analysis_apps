from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    DIRECTION_HIGHER,
    DIRECTION_LOWER,
    SCORING_METHOD_LOG_ROBUST_MINMAX,
    SCORING_METHOD_MINMAX,
    SCORING_METHOD_PERCENTILE_RANK,
    SCORING_METHOD_RANK,
    SCORING_METHOD_ROBUST_MINMAX,
)


def infer_direction_from_source(source: str | float | None) -> str:
    text = "" if source is None else str(source).lower()
    if "lower is better" in text:
        return DIRECTION_LOWER
    return DIRECTION_HIGHER


def normalize_direction(direction: str | None) -> str:
    text = (direction or "").strip().lower()
    if text in {"low", "lower", "lower_is_better", "ascending"}:
        return DIRECTION_LOWER
    return DIRECTION_HIGHER


def normalize_scoring_method_key(method: str | None) -> str:
    normalized = (method or SCORING_METHOD_RANK).strip().lower()
    if normalized == "percentile":
        return SCORING_METHOD_PERCENTILE_RANK
    if normalized in {
        SCORING_METHOD_RANK,
        SCORING_METHOD_PERCENTILE_RANK,
        SCORING_METHOD_MINMAX,
        SCORING_METHOD_ROBUST_MINMAX,
        SCORING_METHOD_LOG_ROBUST_MINMAX,
    }:
        return normalized
    return SCORING_METHOD_RANK


def normalize_weight_map(weight_map: dict[str, float]) -> dict[str, float]:
    if not weight_map:
        return {}
    cleaned = {k: max(float(v), 0.0) for k, v in weight_map.items()}
    total = sum(cleaned.values())
    if total <= 0:
        equal = 1.0 / len(cleaned)
        return {k: equal for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def _rank_average_ascending(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", ascending=True, na_option="keep")


def _rank_average_descending(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", ascending=False, na_option="keep")


def _rank_to_unit_interval(rank_series: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=rank_series.index, dtype=float)
    valid_mask = rank_series.notna()
    n = int(valid_mask.sum())
    if n == 0:
        return output
    if n == 1:
        output.loc[valid_mask] = 0.5
        return output
    output.loc[valid_mask] = (rank_series.loc[valid_mask] - 1.0) / float(n - 1)
    return output.clip(0.0, 1.0)


def rank_scores(values: pd.Series) -> pd.Series:
    """Rank scores in [0,1] using average rank with ties; 0=worst, 1=best."""
    ascending_rank = _rank_average_ascending(values)
    return _rank_to_unit_interval(ascending_rank)


def percentile_rank_scores(values: pd.Series) -> pd.Series:
    """Percentile rank defined as normalized average rank in [0,1]."""
    ascending_rank = _rank_average_ascending(values)
    return _rank_to_unit_interval(ascending_rank)


def _minmax_scale(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return output

    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        output.loc[valid.index] = 0.5
        return output

    output.loc[valid.index] = (valid - min_value) / (max_value - min_value)
    return output.clip(0.0, 1.0)


def minmax_scores(values: pd.Series) -> pd.Series:
    return _minmax_scale(values)


def robust_minmax_scores(values: pd.Series, robust_clip_quantiles: tuple[float, float] = (0.05, 0.95)) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return output

    q_low, q_high = robust_clip_quantiles
    q_low = float(np.clip(q_low, 0.0, 1.0))
    q_high = float(np.clip(q_high, 0.0, 1.0))
    if q_high < q_low:
        q_low, q_high = q_high, q_low

    low = float(valid.quantile(q_low))
    high = float(valid.quantile(q_high))
    if high <= low:
        output.loc[valid.index] = 0.5
        return output

    clipped = valid.clip(lower=low, upper=high)
    min_clipped = float(clipped.min())
    max_clipped = float(clipped.max())
    if max_clipped <= min_clipped:
        output.loc[valid.index] = 0.5
        return output

    output.loc[valid.index] = (clipped - min_clipped) / (max_clipped - min_clipped)
    return output.clip(0.0, 1.0)


def log_robust_minmax_scores(
    values: pd.Series,
    robust_clip_quantiles: tuple[float, float] = (0.05, 0.95),
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return output

    shift = 0.0
    min_value = float(valid.min())
    if min_value < 0:
        shift = -min_value
    transformed = np.log1p(valid + shift)
    transformed_series = pd.Series(transformed, index=valid.index, dtype=float)
    return robust_minmax_scores(transformed_series, robust_clip_quantiles=robust_clip_quantiles)


def score_series(
    values: pd.Series,
    method: str,
    direction: str,
    *,
    robust_clip_quantiles: tuple[float, float] = (0.05, 0.95),
) -> pd.Series:
    method_key = normalize_scoring_method_key(method)
    direction_key = normalize_direction(direction)

    if method_key == SCORING_METHOD_RANK:
        scores = rank_scores(values)
    elif method_key == SCORING_METHOD_PERCENTILE_RANK:
        scores = percentile_rank_scores(values)
    elif method_key == SCORING_METHOD_MINMAX:
        scores = minmax_scores(values)
    elif method_key == SCORING_METHOD_ROBUST_MINMAX:
        scores = robust_minmax_scores(values, robust_clip_quantiles=robust_clip_quantiles)
    elif method_key == SCORING_METHOD_LOG_ROBUST_MINMAX:
        scores = log_robust_minmax_scores(values, robust_clip_quantiles=robust_clip_quantiles)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    if direction_key == DIRECTION_LOWER:
        scores = 1.0 - scores
    return scores.clip(0.0, 1.0)


def compute_rank_series(values: pd.Series, direction: str) -> pd.Series:
    direction_key = normalize_direction(direction)
    if direction_key == DIRECTION_LOWER:
        return _rank_average_ascending(values)
    return _rank_average_descending(values)


def compute_micro_scores(
    raw_data: pd.DataFrame,
    city_columns: Iterable[str],
    direction_map: dict[str, str],
    method: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    city_columns = list(city_columns)
    method_key = normalize_scoring_method_key(method)

    for _, row in raw_data.iterrows():
        direction = direction_map.get(row["criterion_id"], infer_direction_from_source(row.get("source")))
        direction = normalize_direction(direction)
        values = row[city_columns]
        ranks = compute_rank_series(values, direction=direction)
        scores = score_series(values, method=method_key, direction=direction)

        for city in city_columns:
            records.append(
                {
                    "criterion_id": row["criterion_id"],
                    "macro": row["macro"],
                    "major": row["major"],
                    "micro": row["micro"],
                    "source": row.get("source"),
                    "city": city,
                    "raw_value": pd.to_numeric(values[city], errors="coerce"),
                    "rank": pd.to_numeric(ranks[city], errors="coerce"),
                    "score": pd.to_numeric(scores[city], errors="coerce"),
                    "direction": direction,
                }
            )

    return pd.DataFrame.from_records(records)


def _weighted_average(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    valid = group[value_col].notna() & group[weight_col].notna()
    if not valid.any():
        return np.nan
    weights = group.loc[valid, weight_col]
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return np.nan
    values = group.loc[valid, value_col]
    return float((values * weights).sum() / weight_sum)


def _weighted_average_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    weight_col: str,
    output_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        row[output_col] = _weighted_average(group, value_col, weight_col)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_scores(
    micro_scores: pd.DataFrame,
    weighted_criteria: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    merged = micro_scores.merge(
        weighted_criteria[
            [
                "criterion_id",
                "macro_weight",
                "major_weight",
                "minor_weight",
                "effective_micro_weight",
            ]
        ],
        on="criterion_id",
        how="inner",
    )

    major_scores = _weighted_average_by_group(
        merged,
        group_cols=["city", "macro", "major"],
        value_col="score",
        weight_col="minor_weight",
        output_col="major_score",
    )
    major_weight_table = weighted_criteria[["macro", "major", "major_weight"]].drop_duplicates()
    major_scores = major_scores.merge(major_weight_table, on=["macro", "major"], how="left")

    macro_scores = _weighted_average_by_group(
        major_scores,
        group_cols=["city", "macro"],
        value_col="major_score",
        weight_col="major_weight",
        output_col="macro_score",
    )
    macro_weight_table = weighted_criteria[["macro", "macro_weight"]].drop_duplicates()
    macro_scores = macro_scores.merge(macro_weight_table, on="macro", how="left")

    overall_scores = _weighted_average_by_group(
        macro_scores,
        group_cols=["city"],
        value_col="macro_score",
        weight_col="macro_weight",
        output_col="overall_score",
    )

    contributions = merged.copy()
    contributions["contribution"] = contributions["score"] * contributions["effective_micro_weight"]

    return {
        "micro_scores": merged,
        "major_scores": major_scores,
        "macro_scores": macro_scores,
        "overall_scores": overall_scores,
        "contributions": contributions,
    }
