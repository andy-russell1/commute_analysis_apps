from __future__ import annotations

from pathlib import Path

import pandas as pd

from apps.talent_analytics.geography import build_legacy_lookup, legacy_lookup_path


METADATA_OVERRIDES = {
    "total_population": {
        "short_label": "Total population",
        "format_type": "whole_number",
        "default_sort_direction": "desc",
        "source_note": "Population estimates are sourced from Nomis mid-year population estimates at LAD level.",
        "display_order": 1,
    },
    "working_age_population": {
        "short_label": "Working-age population",
        "format_type": "whole_number",
        "default_sort_direction": "desc",
        "source_note": "Working-age population uses APS LAD counts for people aged 16-64 to align with labour-market rates.",
        "display_order": 2,
    },
    "employment_rate": {
        "short_label": "Employment rate",
        "format_type": "percentage_1dp",
        "default_sort_direction": "desc",
        "source_note": "Employment rate uses the APS percentage series for people aged 16-64, with custom-region aggregation reweighted from the source numerators and denominators.",
        "display_order": 3,
    },
    "unemployment_rate": {
        "short_label": "Unemployment rate",
        "format_type": "percentage_1dp",
        "default_sort_direction": "desc",
        "source_note": "Unemployment rate uses the APS percentage series for people aged 16-64, with custom-region aggregation reweighted from the source numerators and denominators.",
        "display_order": 4,
    },
    "economic_activity_rate": {
        "short_label": "Economic activity rate",
        "format_type": "percentage_1dp",
        "default_sort_direction": "desc",
        "source_note": "Economic activity rate uses the APS percentage series for people aged 16-64, with custom-region aggregation reweighted from the source numerators and denominators.",
        "display_order": 5,
    },
    "nvq4_plus_share": {
        "short_label": "NVQ4+ share",
        "format_type": "percentage_1dp",
        "default_sort_direction": "desc",
        "source_note": "NVQ4+ share uses the APS RQF percentage series for residents aged 16-64, with custom-region aggregation reweighted from the source numerators and denominators.",
        "display_order": 6,
    },
    "no_qualifications_share": {
        "short_label": "No qualifications share",
        "format_type": "percentage_1dp",
        "default_sort_direction": "desc",
        "source_note": "No qualifications share uses the APS RQF percentage series for residents aged 16-64, with custom-region aggregation reweighted from the source numerators and denominators.",
        "display_order": 7,
    },
    "professional_occupations_share": {
        "short_label": "Professional occupations share",
        "format_type": "percentage_1dp",
        "default_sort_direction": "desc",
        "source_note": "Professional occupations share uses the APS SOC2020 percentage series for professional occupations, with custom-region aggregation reweighted from the source numerators and denominators.",
        "display_order": 8,
    },
}


def build_demographics_metadata(source_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_key, overrides in METADATA_OVERRIDES.items():
        source_row = source_catalog.loc[source_catalog["metric_key"] == metric_key].iloc[0]
        rows.append(
            {
                "metric_key": metric_key,
                "metric_family": source_row["metric_family"],
                "metric_label": source_row["metric_label"],
                "short_label": overrides["short_label"],
                "unit": source_row["unit"],
                "format_type": overrides["format_type"],
                "default_sort_direction": overrides["default_sort_direction"],
                "aggregation_method": source_row["aggregation_method"],
                "source_system": source_row["source_system"],
                "source_dataset": source_row["source_dataset"],
                "source_note": overrides["source_note"],
                "is_active": "TRUE" if bool(source_row["is_active"]) else "FALSE",
                "display_order": overrides["display_order"],
            }
        )
    return pd.DataFrame(rows).sort_values("display_order").reset_index(drop=True)


def publish_outputs(
    *,
    data_root: Path,
    canonical_lookup: pd.DataFrame,
    source_catalog: pd.DataFrame,
    lad_metrics: pd.DataFrame,
    custom_metrics: pd.DataFrame,
) -> dict[str, Path]:
    shared_dir = data_root / "shared"
    processed_dir = shared_dir / "ons" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    lad_public = lad_metrics.loc[
        :,
        [
            "lad_code",
            "lad_name",
            "metric_family",
            "metric_key",
            "metric_label",
            "period",
            "value",
            "unit",
            "source_system",
            "source_dataset",
            "last_updated",
        ],
    ].sort_values(["metric_key", "lad_name"]).reset_index(drop=True)

    custom_public = custom_metrics.loc[
        :,
        [
            "custom_geography_key",
            "custom_geography_name",
            "display_order",
            "metric_family",
            "metric_key",
            "metric_label",
            "period",
            "value",
            "unit",
            "aggregation_method",
            "source_system",
            "source_dataset",
            "last_updated",
        ],
    ].sort_values(["metric_key", "display_order"]).reset_index(drop=True)

    metadata_public = build_demographics_metadata(source_catalog)
    legacy_lookup = build_legacy_lookup(canonical_lookup)

    lad_path = processed_dir / "demographics_by_lad.csv"
    custom_path = processed_dir / "demographics_by_custom_geography.csv"
    metadata_path = processed_dir / "demographics_metadata.csv"
    legacy_lookup_output_path = legacy_lookup_path(data_root)

    lad_public.to_csv(lad_path, index=False)
    custom_public.to_csv(custom_path, index=False)
    metadata_public.to_csv(metadata_path, index=False)
    legacy_lookup.to_csv(legacy_lookup_output_path, index=False)

    return {
        "demographics_by_lad": lad_path,
        "demographics_by_custom_geography": custom_path,
        "demographics_metadata": metadata_path,
        "legacy_geography_lookup": legacy_lookup_output_path,
    }
