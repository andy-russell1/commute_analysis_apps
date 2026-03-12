from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def bucket_slug(bucket: str) -> str:
    slug = bucket.strip().lower().replace("&", "and")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def count_column(bucket: str) -> str:
    return f"count_{bucket_slug(bucket)}"


def nearest_distance_column(bucket: str) -> str:
    return f"nearest_distance_m_{bucket_slug(bucket)}"


def normalise_weights(weights: dict[str, float], selected_buckets: Iterable[str]) -> dict[str, float]:
    buckets = list(selected_buckets)
    selected = {bucket: float(weights.get(bucket, 0.0)) for bucket in buckets}
    total = float(sum(selected.values()))
    if not buckets:
        return {}
    if total <= 0:
        equal = 1.0 / len(buckets)
        return {bucket: equal for bucket in buckets}
    return {bucket: selected[bucket] / total for bucket in buckets}


def normalise_counts(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=float)
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value == min_value:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - min_value) / (max_value - min_value)


def apply_amenity_kpi(summary_df: pd.DataFrame, selected_buckets: list[str], weights: dict[str, float]) -> pd.DataFrame:
    if summary_df.empty:
        out = summary_df.copy()
        out["amenity_kpi"] = []
        return out
    out = summary_df.copy()
    norm_weights = normalise_weights(weights=weights, selected_buckets=selected_buckets)
    kpi = np.zeros(len(out), dtype=float)
    for bucket in selected_buckets:
        cnt_col = count_column(bucket)
        norm_col = f"normalised_{bucket_slug(bucket)}"
        if cnt_col not in out.columns:
            out[cnt_col] = 0
        out[norm_col] = normalise_counts(out[cnt_col].fillna(0).astype(float))
        kpi += out[norm_col].to_numpy(dtype=float) * float(norm_weights.get(bucket, 0.0))
    out["amenity_kpi"] = np.round(kpi * 100.0, 1)
    return out
