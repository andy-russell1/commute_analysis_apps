from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ons_extract import (
    ExtractedNomisArtifacts,
    PERCENTAGE_DENOMINATOR_MEASURE,
    PERCENTAGE_NUMERATOR_MEASURE,
    PERCENTAGE_VALUE_MEASURE,
)


def load_source_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, dtype=str)
    catalog["is_active"] = catalog["is_active"].astype(str).str.upper().eq("TRUE")
    return catalog


def _catalog_row(catalog: pd.DataFrame, metric_key: str) -> pd.Series:
    row = catalog.loc[catalog["metric_key"] == metric_key]
    if row.empty:
        raise KeyError(f"Metric '{metric_key}' is missing from the ONS source catalog.")
    return row.iloc[0]


def _to_iso_timestamp(value: str) -> str:
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return value
    return parsed.isoformat()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "OBS_VALUE" in frame.columns:
        frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    return frame


def _latest_dataset_updates(extracted: ExtractedNomisArtifacts) -> dict[str, str]:
    updates: dict[str, str] = {}
    for metadata in (
        extracted.population_metadata,
        extracted.aps_counts_metadata,
        extracted.aps_percentages_metadata,
    ):
        timestamp = _to_iso_timestamp(metadata.last_updated)
        existing = updates.get(metadata.mnemonic, "")
        if timestamp and (not existing or timestamp > existing):
            updates[metadata.mnemonic] = timestamp
    return updates


def _build_percentage_source_frame(
    aps_percentages: pd.DataFrame,
    *,
    variable_code: int,
) -> pd.DataFrame:
    variable_rows = aps_percentages[aps_percentages["VARIABLE"].astype(str) == str(variable_code)].copy()
    if variable_rows.empty:
        raise ValueError(f"APS percentage extract is missing variable {variable_code}.")

    measure_rows = variable_rows[
        variable_rows["MEASURES"].astype(str).eq(PERCENTAGE_VALUE_MEASURE) & variable_rows["OBS_VALUE"].notna()
    ].copy()
    if measure_rows.empty:
        raise ValueError(f"APS percentage extract does not contain a populated period for variable {variable_code}.")

    dated_rows = measure_rows.copy()
    dated_rows["_period_sort"] = pd.to_datetime(dated_rows["DATE_CODE"], errors="coerce")
    if dated_rows["_period_sort"].notna().any():
        latest_period = str(
            dated_rows.sort_values(["_period_sort", "DATE_CODE"], ascending=[True, True]).iloc[-1]["DATE_CODE"]
        )
    else:
        latest_period = str(measure_rows["DATE_CODE"].astype(str).max())

    variable_rows = variable_rows[variable_rows["DATE_CODE"].astype(str) == latest_period].copy()

    wide = (
        variable_rows.set_index(
            ["GEOGRAPHY_CODE", "GEOGRAPHY_NAME", "DATE_NAME", "VARIABLE_NAME", "MEASURES"]
        )["OBS_VALUE"]
        .unstack("MEASURES")
        .reset_index()
        .rename(
            columns={
                "GEOGRAPHY_CODE": "lad_code",
                "GEOGRAPHY_NAME": "lad_name",
                "DATE_NAME": "period",
                PERCENTAGE_VALUE_MEASURE: "percentage_value",
                PERCENTAGE_NUMERATOR_MEASURE: "numerator_value",
                PERCENTAGE_DENOMINATOR_MEASURE: "denominator_value",
            }
        )
    )

    for column in ("percentage_value", "numerator_value", "denominator_value"):
        wide[column] = pd.to_numeric(wide[column], errors="coerce")
    wide["value"] = wide["percentage_value"] / 100.0
    return wide


def _build_metric_frame(
    *,
    base: pd.DataFrame,
    metric_key: str,
    values: pd.Series,
    numerator_values: pd.Series | None,
    denominator_values: pd.Series | None,
    catalog: pd.DataFrame,
    dataset_last_updated: dict[str, str],
) -> pd.DataFrame:
    meta = _catalog_row(catalog, metric_key)
    source_dataset = str(meta["source_dataset"])
    null_series = pd.Series([float("nan")] * len(base), index=base.index, dtype="float64")
    return pd.DataFrame(
        {
            "lad_code": base["lad_code"],
            "lad_name": base["lad_name"],
            "metric_family": meta["metric_family"],
            "metric_key": metric_key,
            "metric_label": meta["metric_label"],
            "period": base["period"],
            "value": values.astype(float),
            "unit": meta["unit"],
            "source_system": meta["source_system"],
            "source_dataset": source_dataset,
            "last_updated": dataset_last_updated.get(source_dataset, ""),
            "aggregation_method": meta["aggregation_method"],
            "denominator_metric_key": meta["denominator_metric_key"],
            "numerator_value": numerator_values.astype(float) if numerator_values is not None else null_series,
            "denominator_value": denominator_values.astype(float) if denominator_values is not None else null_series,
        }
    )


def build_lad_metrics(
    extracted: ExtractedNomisArtifacts,
    canonical_lookup: pd.DataFrame,
    source_catalog: pd.DataFrame,
) -> pd.DataFrame:
    active_catalog = source_catalog[source_catalog["is_active"]].copy()
    target_lads = sorted(canonical_lookup["lad_code"].dropna().unique().tolist())
    dataset_last_updated = _latest_dataset_updates(extracted)

    population = _read_csv(extracted.population_data_path)
    population = population[population["GEOGRAPHY_CODE"].isin(target_lads)].copy()
    population_base = population.rename(
        columns={
            "GEOGRAPHY_CODE": "lad_code",
            "GEOGRAPHY_NAME": "lad_name",
            "DATE_NAME": "period",
        }
    )[["lad_code", "lad_name", "period", "OBS_VALUE"]]

    aps_counts = _read_csv(extracted.aps_counts_data_path)
    aps_counts["CELL"] = aps_counts["CELL"].astype(str)
    aps_counts = aps_counts[aps_counts["GEOGRAPHY_CODE"].isin(target_lads)].copy()

    working_age = aps_counts[aps_counts["CELL"].isin({str(extracted.aps_counts_cell_map["working_age_population"])})].copy()
    working_age_base = working_age.rename(
        columns={
            "GEOGRAPHY_CODE": "lad_code",
            "GEOGRAPHY_NAME": "lad_name",
            "DATE_NAME": "period",
            "OBS_VALUE": "working_age_population",
        }
    )[["lad_code", "lad_name", "period", "working_age_population"]]

    aps_percentages = _read_csv(extracted.aps_percentages_data_path)
    aps_percentages["VARIABLE"] = aps_percentages["VARIABLE"].astype(str)
    aps_percentages["MEASURES"] = aps_percentages["MEASURES"].astype(str)
    aps_percentages = aps_percentages[aps_percentages["GEOGRAPHY_CODE"].isin(target_lads)].copy()

    metric_frames = [
        _build_metric_frame(
            base=population_base,
            metric_key="total_population",
            values=population_base["OBS_VALUE"],
            numerator_values=None,
            denominator_values=None,
            catalog=active_catalog,
            dataset_last_updated=dataset_last_updated,
        ),
        _build_metric_frame(
            base=working_age_base,
            metric_key="working_age_population",
            values=working_age_base["working_age_population"],
            numerator_values=None,
            denominator_values=None,
            catalog=active_catalog,
            dataset_last_updated=dataset_last_updated,
        ),
    ]

    for metric_key in (
        "employment_rate",
        "unemployment_rate",
        "economic_activity_rate",
        "nvq4_plus_share",
        "no_qualifications_share",
        "professional_occupations_share",
    ):
        percentage_frame = _build_percentage_source_frame(
            aps_percentages,
            variable_code=extracted.aps_percentages_variable_map[metric_key],
        )
        metric_frames.append(
            _build_metric_frame(
                base=percentage_frame,
                metric_key=metric_key,
                values=percentage_frame["value"],
                numerator_values=percentage_frame["numerator_value"],
                denominator_values=percentage_frame["denominator_value"],
                catalog=active_catalog,
                dataset_last_updated=dataset_last_updated,
            )
        )

    lad_metrics = pd.concat(metric_frames, ignore_index=True)
    lad_metrics = lad_metrics.sort_values(["metric_key", "lad_name"]).reset_index(drop=True)
    return lad_metrics


def aggregate_custom_geographies(lad_metrics: pd.DataFrame, canonical_lookup: pd.DataFrame) -> pd.DataFrame:
    mapping = canonical_lookup.loc[
        :,
        ["custom_geography_key", "custom_geography_name", "display_order", "lad_code"],
    ].drop_duplicates()
    merged = lad_metrics.merge(mapping, on="lad_code", how="left", validate="many_to_one")

    group_columns = [
        "custom_geography_key",
        "custom_geography_name",
        "display_order",
        "metric_family",
        "metric_key",
        "metric_label",
        "period",
        "unit",
        "aggregation_method",
        "source_system",
        "source_dataset",
        "last_updated",
    ]

    records: list[dict[str, object]] = []
    for keys, frame in merged.groupby(group_columns, dropna=False, sort=False):
        record = dict(zip(group_columns, keys))
        aggregation_method = str(record["aggregation_method"])
        if aggregation_method == "sum":
            record["value"] = float(frame["value"].sum())
        else:
            numerator = pd.to_numeric(frame["numerator_value"], errors="coerce").sum()
            denominator = pd.to_numeric(frame["denominator_value"], errors="coerce").sum()
            record["value"] = float(numerator / denominator) if denominator else pd.NA
        records.append(record)

    custom_metrics = pd.DataFrame.from_records(records)
    custom_metrics = custom_metrics.sort_values(["metric_key", "display_order"]).reset_index(drop=True)
    return custom_metrics
