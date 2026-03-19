from __future__ import annotations

import pandas as pd


def validate_outputs(
    *,
    canonical_lookup: pd.DataFrame,
    source_catalog: pd.DataFrame,
    lad_metrics: pd.DataFrame,
    custom_metrics: pd.DataFrame,
) -> None:
    errors: list[str] = []

    active_lookup = canonical_lookup[canonical_lookup["is_active"]].copy()
    expected_lads = sorted(active_lookup["lad_code"].dropna().unique().tolist())
    expected_custom_groups = active_lookup["custom_geography_key"].dropna().nunique()
    active_metric_keys = sorted(source_catalog.loc[source_catalog["is_active"], "metric_key"].dropna().unique().tolist())

    if lad_metrics.empty:
        errors.append("LAD-level demographics output is empty.")
    if custom_metrics.empty:
        errors.append("Custom-geography demographics output is empty.")

    duplicated_lookup = active_lookup.duplicated(subset=["custom_geography_key", "lad_code"]).any()
    if duplicated_lookup:
        errors.append("Canonical custom geography lookup contains duplicate LAD-to-custom-group mappings.")

    lad_codes_in_output = sorted(lad_metrics["lad_code"].dropna().unique().tolist())
    if lad_codes_in_output != expected_lads:
        missing = sorted(set(expected_lads) - set(lad_codes_in_output))
        extra = sorted(set(lad_codes_in_output) - set(expected_lads))
        if missing:
            errors.append(f"LAD-level output is missing expected London LAD codes: {', '.join(missing[:10])}")
        if extra:
            errors.append(f"LAD-level output contains unexpected LAD codes: {', '.join(extra[:10])}")

    custom_group_count = custom_metrics["custom_geography_key"].dropna().nunique()
    if custom_group_count != expected_custom_groups:
        errors.append(
            f"Expected {expected_custom_groups} custom regions in aggregated output, found {custom_group_count}."
        )

    metric_keys_in_outputs = sorted(set(lad_metrics["metric_key"]).union(set(custom_metrics["metric_key"])))
    unknown_metric_keys = sorted(set(metric_keys_in_outputs) - set(active_metric_keys))
    if unknown_metric_keys:
        errors.append(f"Processed outputs contain metric keys not present in the active source catalog: {', '.join(unknown_metric_keys)}")

    if custom_metrics["custom_geography_key"].isna().any():
        errors.append("Aggregated output contains null custom_geography_key values.")

    proportion_metrics = pd.concat(
        [
            lad_metrics.loc[lad_metrics["unit"].eq("proportion"), ["metric_key", "value"]],
            custom_metrics.loc[custom_metrics["unit"].eq("proportion"), ["metric_key", "value"]],
        ],
        ignore_index=True,
    )
    if not proportion_metrics.empty:
        invalid = proportion_metrics[
            pd.to_numeric(proportion_metrics["value"], errors="coerce").lt(0)
            | pd.to_numeric(proportion_metrics["value"], errors="coerce").gt(1)
        ]
        if not invalid.empty:
            bad_keys = ", ".join(sorted(invalid["metric_key"].dropna().unique().tolist()))
            errors.append(f"One or more proportion metrics fall outside the expected 0-1 range: {bad_keys}")

    count_metrics = pd.concat(
        [
            lad_metrics.loc[lad_metrics["unit"].eq("persons"), ["metric_key", "value"]],
            custom_metrics.loc[custom_metrics["unit"].eq("persons"), ["metric_key", "value"]],
        ],
        ignore_index=True,
    )
    if not count_metrics.empty:
        invalid_counts = count_metrics[pd.to_numeric(count_metrics["value"], errors="coerce").lt(0)]
        if not invalid_counts.empty:
            bad_keys = ", ".join(sorted(invalid_counts["metric_key"].dropna().unique().tolist()))
            errors.append(f"One or more population count metrics are negative: {bad_keys}")

    if lad_metrics["period"].isna().any() or custom_metrics["period"].isna().any():
        errors.append("One or more processed metrics have missing period values.")

    expected_lad_rows = len(expected_lads) * len(active_metric_keys)
    if len(lad_metrics) != expected_lad_rows:
        errors.append(f"Expected {expected_lad_rows} LAD metric rows, found {len(lad_metrics)}.")

    expected_custom_rows = expected_custom_groups * len(active_metric_keys)
    if len(custom_metrics) != expected_custom_rows:
        errors.append(f"Expected {expected_custom_rows} custom geography metric rows, found {len(custom_metrics)}.")

    if errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise ValueError(f"ONS / Nomis pipeline validation failed:\n{joined}")
