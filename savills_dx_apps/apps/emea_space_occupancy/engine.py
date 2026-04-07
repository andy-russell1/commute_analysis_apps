from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ACTION_THRESHOLD_DEFAULTS,
    ATTENDANCE_MAX_RATE,
    ATTENDANCE_MIN_RATE,
    DEFAULT_ROOM_SEATS,
    MOVE_COMPLEXITY_SCORE_MAP,
    PREFERRED_SCENARIO_AUTO_KEY,
    RISK_THRESHOLD_DEFAULTS,
    SCENARIO_SCORE_WEIGHTS,
    SPACE_PLANNING_DEFAULTS,
)


STANDARD_METRIC_COLUMNS = [
    "seat_ratio_target",
    "desk_sharing_ratio_target",
    "sqm_per_person_target",
    "collaboration_area_pct_target",
    "meeting_seats_per_100_staff",
    "focus_seats_per_100_staff",
    "planning_utilisation_threshold_pct",
]


def _safe_divide(numerator: float | int, denominator: float | int, default: float = 0.0) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return float(default)
    return float(numerator) / float(denominator)


def _weighted_average(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return float(default)
    weight_values = weights.loc[valid].astype(float)
    if float(weight_values.sum()) == 0.0:
        return float(default)
    return float(np.average(values.loc[valid].astype(float), weights=weight_values))


def _series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _safe_ratio_series(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    default: float = 0.0,
) -> pd.Series:
    return np.where(
        denominator.fillna(0.0) > 0,
        numerator.fillna(0.0) / denominator.replace(0, np.nan),
        default,
    )


def _clipped_score(
    base_score: float,
    penalties: pd.Series | np.ndarray,
    *,
    floor: float = 0.0,
    ceiling: float = 100.0,
) -> pd.Series:
    return (base_score - pd.Series(penalties)).clip(lower=floor, upper=ceiling)


def _latest_standard_rows(standards: pd.DataFrame) -> pd.DataFrame:
    if standards.empty:
        return standards
    current = standards.copy()
    if "effective_date" in current.columns:
        current = current.sort_values("effective_date")
    return current.groupby(["standard_group", "applicable_to"], dropna=False).tail(1).reset_index(drop=True)


def _portfolio_standard_defaults(standards: pd.DataFrame) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for metric in STANDARD_METRIC_COLUMNS:
        series = pd.to_numeric(standards.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
        if not series.empty:
            defaults[metric] = float(series.mean())
    return defaults


def _normalise_parameter_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def get_scenario_names(clean_sheets: dict[str, pd.DataFrame]) -> list[str]:
    assumptions = clean_sheets.get("scenario_assumptions", pd.DataFrame())
    if assumptions.empty or "scenario_name" not in assumptions.columns:
        return []
    return sorted(assumptions["scenario_name"].dropna().astype(str).unique().tolist())


def build_filter_options(clean_sheets: dict[str, pd.DataFrame]) -> dict[str, list[Any]]:
    hierarchy = clean_sheets.get("portfolio_hierarchy", pd.DataFrame())
    people = clean_sheets.get("people_demand", pd.DataFrame())
    occupancy = clean_sheets.get("occupancy_utilisation", pd.DataFrame())
    return {
        "region": sorted(hierarchy.get("region", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()),
        "country": sorted(hierarchy.get("country", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()),
        "city": sorted(hierarchy.get("city", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()),
        "site_name": sorted(hierarchy.get("site_name", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()),
        "site_type": sorted(hierarchy.get("site_type", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()),
        "building_name": sorted(
            hierarchy.get("building_name", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
        ),
        "business_unit": sorted(
            people.get("business_unit", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
        ),
        "months": sorted(occupancy.get("month", pd.Series(dtype="datetime64[ns]")).dropna().unique().tolist()),
    }


def _apply_list_filter(df: pd.DataFrame, column: str, values: list[Any] | None) -> pd.DataFrame:
    if df.empty or column not in df.columns or not values:
        return df
    return df[df[column].isin(values)].copy()


def _apply_month_filter(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    if df.empty or "month" not in df.columns:
        return df
    month_range = filters.get("month_range")
    if not month_range or len(month_range) != 2:
        return df
    start, end = month_range
    if start is None or end is None:
        return df
    return df[(df["month"] >= pd.Timestamp(start)) & (df["month"] <= pd.Timestamp(end))].copy()


def filter_frames(clean_sheets: dict[str, pd.DataFrame], filters: dict[str, Any]) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for sheet_name, df in clean_sheets.items():
        current = df.copy()
        for filter_key in ["region", "country", "city", "site_name", "site_type", "building_name", "business_unit"]:
            current = _apply_list_filter(current, filter_key, filters.get(filter_key))
        current = _apply_month_filter(current, filters)
        filtered[sheet_name] = current
    return filtered


def _site_allocation_share(clean_sheets: dict[str, pd.DataFrame], filtered_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hierarchy = clean_sheets.get("portfolio_hierarchy", pd.DataFrame())
    selected_hierarchy = filtered_frames.get("portfolio_hierarchy", pd.DataFrame())
    if hierarchy.empty:
        return pd.DataFrame(columns=["site_id", "allocation_share"])
    total = hierarchy.groupby("site_id", dropna=False).agg(
        total_seats=("seat_capacity", "sum"),
        total_area=("usable_area_sqm", "sum"),
    )
    selected = selected_hierarchy.groupby("site_id", dropna=False).agg(
        selected_seats=("seat_capacity", "sum"),
        selected_area=("usable_area_sqm", "sum"),
    )
    share = selected.join(total, how="outer").fillna(0.0)
    share["allocation_share"] = np.where(
        share["total_seats"] > 0,
        share["selected_seats"] / share["total_seats"],
        np.where(share["total_area"] > 0, share["selected_area"] / share["total_area"], 1.0),
    )
    share["allocation_share"] = share["allocation_share"].clip(lower=0.0, upper=1.0)
    return share.reset_index()[["site_id", "allocation_share"]]


def _scaled_people_demand(clean_sheets: dict[str, pd.DataFrame], filtered_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    people = filtered_frames.get("people_demand", pd.DataFrame()).copy()
    if people.empty:
        return people
    share = _site_allocation_share(clean_sheets, filtered_frames)
    people = people.merge(share, on="site_id", how="left")
    people["allocation_share"] = people["allocation_share"].fillna(1.0)
    for column in ["current_headcount", "forecast_headcount_12m", "forecast_headcount_24m"]:
        if column in people.columns:
            people[column] = pd.to_numeric(people[column], errors="coerce").fillna(0.0) * people["allocation_share"]
    return people


def _scaled_occupancy(clean_sheets: dict[str, pd.DataFrame], filtered_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    occupancy = filtered_frames.get("occupancy_utilisation", pd.DataFrame()).copy()
    if occupancy.empty:
        return occupancy
    share = _site_allocation_share(clean_sheets, filtered_frames)
    occupancy = occupancy.merge(share, on="site_id", how="left")
    occupancy["allocation_share"] = occupancy["allocation_share"].fillna(1.0)
    for column in [
        "current_headcount",
        "seat_capacity",
        "avg_daily_attendance",
        "peak_daily_attendance",
        "badge_swipes",
        "desk_bookings",
        "meeting_room_bookings",
    ]:
        if column in occupancy.columns:
            occupancy[column] = pd.to_numeric(occupancy[column], errors="coerce").fillna(0.0) * occupancy["allocation_share"]
    return occupancy


def _latest_occupancy_summary(occupancy: pd.DataFrame) -> pd.DataFrame:
    if occupancy.empty:
        return pd.DataFrame(columns=["site_id"])
    ordered = occupancy.sort_values("month")
    latest = ordered.groupby("site_id", dropna=False).tail(1).copy()
    earliest = ordered.groupby("site_id", dropna=False).head(1).copy()
    earliest = earliest[["site_id", "avg_desk_utilisation_pct"]].rename(
        columns={"avg_desk_utilisation_pct": "avg_desk_utilisation_pct_start"}
    )
    latest = latest.merge(earliest, on="site_id", how="left")
    latest["avg_desk_utilisation_delta"] = (
        latest["avg_desk_utilisation_pct"] - latest["avg_desk_utilisation_pct_start"]
    ).fillna(0.0)
    return latest


def _prepare_standards(clean_sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    standards = _latest_standard_rows(clean_sheets.get("standards", pd.DataFrame()).copy())
    if standards.empty:
        return pd.DataFrame(), pd.DataFrame()
    workstyle = standards[standards["standard_group"].astype(str).str.lower() == "workstyle"].copy()
    site_type = standards[standards["standard_group"].astype(str).str.lower() == "site type"].copy()
    return workstyle, site_type


def _weighted_site_standards(clean_sheets: dict[str, pd.DataFrame], people: pd.DataFrame) -> pd.DataFrame:
    workstyle_standards, site_type_standards = _prepare_standards(clean_sheets)
    if people.empty:
        return pd.DataFrame(columns=["site_id"])

    portfolio_defaults = _portfolio_standard_defaults(pd.concat([workstyle_standards, site_type_standards], ignore_index=True))

    merged = people.merge(
        workstyle_standards,
        left_on="workstyle_category",
        right_on="applicable_to",
        how="left",
        suffixes=("", "_workstyle"),
    )
    site_type_lookup = (
        site_type_standards.set_index("applicable_to")[STANDARD_METRIC_COLUMNS].copy()
        if not site_type_standards.empty
        else pd.DataFrame(columns=STANDARD_METRIC_COLUMNS)
    )
    hierarchy = clean_sheets.get("portfolio_hierarchy", pd.DataFrame())
    site_types = hierarchy[["site_id", "site_type"]].drop_duplicates() if not hierarchy.empty else pd.DataFrame(columns=["site_id", "site_type"])
    weighted_rows = []
    for site_id, group in merged.groupby("site_id", dropna=False):
        weight_col = group["current_headcount"].fillna(0.0)
        site_type_value = site_types.loc[site_types["site_id"] == site_id, "site_type"].astype(str).iloc[0] if not site_types.empty and (site_types["site_id"] == site_id).any() else ""
        row_payload: dict[str, Any] = {"site_id": site_id}
        for metric in STANDARD_METRIC_COLUMNS:
            default_column = f"default_{metric}"
            source_column = f"{default_column}_source"
            workstyle_value = _weighted_average(group[metric], weight_col, default=np.nan)
            site_type_value_for_metric = np.nan
            if site_type_value in site_type_lookup.index:
                site_type_value_for_metric = pd.to_numeric(pd.Series([site_type_lookup.loc[site_type_value, metric]]), errors="coerce").iloc[0]
            portfolio_value = float(portfolio_defaults.get(metric, np.nan))
            if pd.notna(workstyle_value):
                row_payload[default_column] = float(workstyle_value)
                row_payload[source_column] = "Workstyle weighted mix"
            elif pd.notna(site_type_value_for_metric):
                row_payload[default_column] = float(site_type_value_for_metric)
                row_payload[source_column] = f"Site type fallback: {site_type_value}"
            else:
                row_payload[default_column] = portfolio_value
                row_payload[source_column] = "Portfolio default fallback"
        weighted_rows.append(row_payload)
    defaults = pd.DataFrame(weighted_rows)
    if defaults.empty:
        return defaults

    defaults["default_standard_precedence"] = "Workstyle weighted mix -> Site type fallback -> Portfolio default fallback"
    keep_columns = [column for column in defaults.columns if column.startswith("default_")] + ["site_id"]
    return defaults[keep_columns].drop_duplicates("site_id").reset_index(drop=True)


def build_portfolio_baseline(clean_sheets: dict[str, pd.DataFrame], filters: dict[str, Any]) -> dict[str, Any]:
    filtered = filter_frames(clean_sheets, filters)
    hierarchy = filtered.get("portfolio_hierarchy", pd.DataFrame()).copy()
    properties = filtered.get("property_metrics", pd.DataFrame()).copy()
    people = _scaled_people_demand(clean_sheets, filtered)
    occupancy = _scaled_occupancy(clean_sheets, filtered)
    space_inventory = filtered.get("space_inventory", pd.DataFrame()).copy()

    if hierarchy.empty and properties.empty and people.empty:
        empty_site = pd.DataFrame(columns=["site_id"])
        return {
            "filtered_frames": filtered,
            "site_table": empty_site,
            "building_table": pd.DataFrame(),
            "floor_table": pd.DataFrame(),
            "space_mix": pd.DataFrame(),
            "portfolio_summary": {},
            "occupancy_trend": pd.DataFrame(),
            "occupancy_latest": pd.DataFrame(),
        }

    site_dim = hierarchy[
        ["site_id", "region", "country", "city", "site_name", "site_type", "criticality", "delivery_status"]
    ].drop_duplicates("site_id")
    site_hierarchy = hierarchy.groupby("site_id", dropna=False).agg(
        floor_count=("floor_id", "nunique"),
        building_count=("building_id", "nunique"),
        hierarchy_seats=("seat_capacity", "sum"),
        hierarchy_usable_area_sqm=("usable_area_sqm", "sum"),
        hierarchy_gross_area_sqm=("gross_area_sqm", "sum"),
    )
    property_summary = properties.groupby("site_id", dropna=False).agg(
        property_seats=("seat_capacity_total", "sum"),
        property_usable_area_sqm=("usable_area_sqm", "sum"),
        property_gross_area_sqm=("gross_area_sqm", "sum"),
        annual_property_cost_eur=("annual_property_cost_eur", "sum"),
        building_count_properties=("building_id", "nunique"),
        earliest_lease_end_date=("lease_end_date", "min"),
    )
    people_summary = people.groupby("site_id", dropna=False).agg(
        current_headcount=("current_headcount", "sum"),
        forecast_headcount_12m=("forecast_headcount_12m", "sum"),
        forecast_headcount_24m=("forecast_headcount_24m", "sum"),
    )
    if not people.empty and "strategic_priority" in people.columns:
        priority_rank = {"Growth": 3, "Maintain": 2, "Consolidate": 1}
        priority_rows = []
        for site_id, group in people.groupby("site_id", dropna=False):
            ranked = group["strategic_priority"].astype(str).map(priority_rank).fillna(0)
            if ranked.empty or int(ranked.max()) == 0:
                priority_rows.append({"site_id": site_id, "dominant_strategic_priority": "Maintain"})
                continue
            priority_value = group.loc[ranked.idxmax(), "strategic_priority"]
            priority_rows.append({"site_id": site_id, "dominant_strategic_priority": str(priority_value)})
        strategic_priority = pd.DataFrame(priority_rows)
    else:
        strategic_priority = pd.DataFrame(columns=["site_id", "dominant_strategic_priority"])
    if not people.empty:
        attendance_rows = []
        for site_id, group in people.groupby("site_id", dropna=False):
            weights = group["current_headcount"].fillna(0.0)
            attendance_rows.append(
                {
                    "site_id": site_id,
                    "avg_attendance_pct_base": _weighted_average(group["avg_attendance_pct"], weights, default=0.5),
                    "peak_attendance_pct_base": _weighted_average(group["peak_attendance_pct"], weights, default=0.7),
                    "remote_ratio_pct_base": _weighted_average(group["remote_ratio_pct"], weights, default=0.4),
                }
            )
        attendance_summary = pd.DataFrame(attendance_rows)
    else:
        attendance_summary = pd.DataFrame(columns=["site_id"])

    occupancy_latest = _latest_occupancy_summary(occupancy)
    occupancy_latest_summary = occupancy_latest.groupby("site_id", dropna=False).agg(
        avg_daily_attendance=("avg_daily_attendance", "sum"),
        peak_daily_attendance=("peak_daily_attendance", "sum"),
        avg_desk_utilisation_pct=("avg_desk_utilisation_pct", "mean"),
        peak_desk_utilisation_pct=("peak_desk_utilisation_pct", "mean"),
        avg_meeting_room_utilisation_pct=("avg_meeting_room_utilisation_pct", "mean"),
        collaboration_space_utilisation_pct=("collaboration_space_utilisation_pct", "mean"),
        avg_desk_utilisation_delta=("avg_desk_utilisation_delta", "mean"),
        latest_month=("month", "max"),
    )
    if not occupancy_latest_summary.empty:
        occupancy_latest_summary["observed_avg_attendance_rate"] = _safe_ratio_series(
            occupancy_latest_summary["avg_daily_attendance"],
            people_summary.reindex(occupancy_latest_summary.index)["current_headcount"] if not people_summary.empty else pd.Series(0.0, index=occupancy_latest_summary.index),
            default=np.nan,
        )
        occupancy_latest_summary["observed_peak_attendance_rate"] = _safe_ratio_series(
            occupancy_latest_summary["peak_daily_attendance"],
            people_summary.reindex(occupancy_latest_summary.index)["current_headcount"] if not people_summary.empty else pd.Series(0.0, index=occupancy_latest_summary.index),
            default=np.nan,
        )
        occupancy_latest_summary["occupancy_supported"] = True

    meeting_space = pd.DataFrame(columns=["site_id"])
    if not space_inventory.empty:
        inventory = space_inventory.copy()
        inventory["space_type_lower"] = inventory["space_type"].astype(str).str.lower()
        inventory["space_subtype_lower"] = inventory["space_subtype"].astype(str).str.lower()
        inventory["meeting_capacity_component"] = np.where(
            inventory["space_type_lower"].str.contains("meeting"),
            inventory["capacity"].fillna(0.0),
            0.0,
        )
        inventory["focus_capacity_component"] = np.where(
            inventory["space_subtype_lower"].str.contains("focus"),
            inventory["capacity"].fillna(0.0),
            0.0,
        )
        inventory["collaboration_area_component"] = np.where(
            inventory["space_type_lower"].str.contains("collaboration"),
            inventory["area_sqm"].fillna(0.0),
            0.0,
        )
        meeting_space = inventory.groupby("site_id", dropna=False).agg(
            existing_meeting_capacity=("meeting_capacity_component", "sum"),
            existing_focus_capacity=("focus_capacity_component", "sum"),
            existing_collaboration_area_sqm=("collaboration_area_component", "sum"),
        ).reset_index()
        meeting_space["space_inventory_supported"] = True

    standards = _weighted_site_standards(clean_sheets, people)
    site_table = (
        site_dim.merge(site_hierarchy, on="site_id", how="outer")
        .merge(property_summary, on="site_id", how="left")
        .merge(people_summary, on="site_id", how="left")
        .merge(strategic_priority, on="site_id", how="left")
        .merge(attendance_summary, on="site_id", how="left")
        .merge(occupancy_latest_summary, on="site_id", how="left")
        .merge(meeting_space, on="site_id", how="left")
        .merge(standards, on="site_id", how="left")
    )
    site_table["existing_seats"] = site_table["property_seats"].fillna(site_table["hierarchy_seats"]).fillna(0.0)
    site_table["existing_usable_area_sqm"] = (
        site_table["property_usable_area_sqm"].fillna(site_table["hierarchy_usable_area_sqm"]).fillna(0.0)
    )
    site_table["existing_gross_area_sqm"] = (
        site_table["property_gross_area_sqm"].fillna(site_table["hierarchy_gross_area_sqm"]).fillna(0.0)
    )
    site_table["seat_gap_baseline"] = site_table["existing_seats"] - site_table["current_headcount"].fillna(0.0)
    site_table["sqm_per_person_baseline"] = site_table.apply(
        lambda row: _safe_divide(row["existing_usable_area_sqm"], row["current_headcount"], default=0.0), axis=1
    )
    site_table["occupancy_supported"] = (
        site_table["occupancy_supported"].fillna(False) if "occupancy_supported" in site_table.columns else False
    )
    site_table["space_inventory_supported"] = (
        site_table["space_inventory_supported"].fillna(False) if "space_inventory_supported" in site_table.columns else False
    )
    site_table["dominant_strategic_priority"] = (
        site_table["dominant_strategic_priority"].fillna("Maintain")
        if "dominant_strategic_priority" in site_table.columns
        else "Maintain"
    )
    site_table["lease_expiry_within_24m"] = (
        site_table["earliest_lease_end_date"].notna()
        & ((site_table["earliest_lease_end_date"] - pd.Timestamp.now(tz=None)).dt.days <= 730)
    )

    building_table = properties.copy()
    if not building_table.empty:
        building_table["site_building_seat_share"] = building_table.groupby("site_id")["seat_capacity_total"].transform(
            lambda series: series / series.sum() if float(series.sum()) else 0.0
        )
        building_table = building_table.merge(
            site_table[["site_id", "current_headcount", "forecast_headcount_12m", "forecast_headcount_24m"]],
            on="site_id",
            how="left",
        )
        for column in ["current_headcount", "forecast_headcount_12m", "forecast_headcount_24m"]:
            building_table[f"{column}_allocated"] = (
                building_table[column].fillna(0.0) * building_table["site_building_seat_share"].fillna(0.0)
            )
        building_table["seat_gap_baseline"] = (
            building_table["seat_capacity_total"].fillna(0.0) - building_table["current_headcount_allocated"].fillna(0.0)
        )

    floor_table = hierarchy.copy()
    if not floor_table.empty:
        floor_table["site_floor_seat_share"] = floor_table.groupby("site_id")["seat_capacity"].transform(
            lambda series: series / series.sum() if float(series.sum()) else 0.0
        )
        floor_table = floor_table.merge(site_table[["site_id", "current_headcount"]], on="site_id", how="left")
        floor_table["current_headcount_allocated"] = (
            floor_table["current_headcount"].fillna(0.0) * floor_table["site_floor_seat_share"].fillna(0.0)
        )
        floor_table["seat_gap_baseline"] = floor_table["seat_capacity"].fillna(0.0) - floor_table[
            "current_headcount_allocated"
        ].fillna(0.0)

    space_mix = pd.DataFrame()
    if not space_inventory.empty:
        space_mix = space_inventory.groupby("space_type", dropna=False).agg(
            area_sqm=("area_sqm", "sum"),
            seat_count=("seat_count", "sum"),
            room_count=("room_count", "sum"),
            capacity=("capacity", "sum"),
        ).reset_index()
        total_area = float(space_mix["area_sqm"].sum())
        space_mix["area_share_pct"] = np.where(total_area > 0, space_mix["area_sqm"] / total_area, 0.0)

    occupancy_trend = pd.DataFrame()
    if not occupancy.empty:
        occupancy_trend = occupancy.groupby("month", dropna=False).agg(
            current_headcount=("current_headcount", "sum"),
            seat_capacity=("seat_capacity", "sum"),
            avg_daily_attendance=("avg_daily_attendance", "sum"),
            peak_daily_attendance=("peak_daily_attendance", "sum"),
            avg_desk_utilisation_pct=("avg_desk_utilisation_pct", "mean"),
            peak_desk_utilisation_pct=("peak_desk_utilisation_pct", "mean"),
            avg_meeting_room_utilisation_pct=("avg_meeting_room_utilisation_pct", "mean"),
            collaboration_space_utilisation_pct=("collaboration_space_utilisation_pct", "mean"),
        ).reset_index()

    portfolio_summary = {
        "total_sites": int(site_table["site_id"].nunique()) if "site_id" in site_table.columns else 0,
        "total_buildings": int(building_table["building_id"].nunique()) if "building_id" in building_table.columns else 0,
        "total_floors": int(floor_table["floor_id"].nunique()) if "floor_id" in floor_table.columns else 0,
        "usable_area_sqm": float(site_table["existing_usable_area_sqm"].fillna(0.0).sum()),
        "total_seat_capacity": float(site_table["existing_seats"].fillna(0.0).sum()),
        "current_headcount": float(site_table["current_headcount"].fillna(0.0).sum()),
        "average_desk_utilisation_pct": float(site_table["avg_desk_utilisation_pct"].fillna(0.0).mean()),
        "portfolio_risk_sites_count": int((site_table["seat_gap_baseline"].fillna(0.0) < 0).sum()),
        "annual_property_cost_eur": float(site_table["annual_property_cost_eur"].fillna(0.0).sum()),
    }

    return {
        "filtered_frames": filtered,
        "site_table": site_table.sort_values(["region", "country", "city", "site_name"]).reset_index(drop=True),
        "building_table": building_table.sort_values(["site_name", "building_name"]).reset_index(drop=True),
        "floor_table": floor_table.sort_values(["site_name", "building_name", "floor_sequence"]).reset_index(drop=True),
        "space_mix": space_mix.sort_values("area_sqm", ascending=False).reset_index(drop=True),
        "portfolio_summary": portfolio_summary,
        "occupancy_trend": occupancy_trend.sort_values("month").reset_index(drop=True),
        "occupancy_latest": occupancy_latest,
    }


def _base_horizon_months(assumptions_df: pd.DataFrame) -> int:
    if assumptions_df.empty or "planning_horizon_months" not in assumptions_df.columns:
        return 12
    values = pd.to_numeric(assumptions_df["planning_horizon_months"], errors="coerce").dropna()
    return int(values.iloc[0]) if not values.empty else 12


def _ensure_assumption_rows(clean_sheets: dict[str, pd.DataFrame], assumptions_df: pd.DataFrame) -> pd.DataFrame:
    if assumptions_df.empty:
        return assumptions_df
    assumptions = assumptions_df.copy()
    scenario_name = str(assumptions["scenario_name"].iloc[0])
    scenario_id = str(assumptions["scenario_id"].iloc[0]) if "scenario_id" in assumptions.columns else "SCN-LIVE"
    planning_horizon_months = _base_horizon_months(assumptions)
    standards = clean_sheets.get("standards", pd.DataFrame())
    default_collaboration = float(standards.get("collaboration_area_pct_target", pd.Series([0.18])).dropna().mean())
    default_peak_threshold = float(standards.get("planning_utilisation_threshold_pct", pd.Series([0.82])).dropna().mean())
    to_add = []
    existing_names = {_normalise_parameter_name(value) for value in assumptions["parameter_name"].astype(str)}
    if "collaboration area target" not in existing_names:
        to_add.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "planning_horizon_months": planning_horizon_months,
                "parameter_category": "Planning",
                "parameter_name": "Collaboration Area Target",
                "scope_level": "Global",
                "scope_value": "All",
                "value": default_collaboration,
                "unit": "pct",
                "driver_note": "Derived from current standards mix.",
                "owner": "Session",
                "version_status": "Working",
            }
        )
    if "maximum peak utilisation threshold" not in existing_names:
        to_add.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "planning_horizon_months": planning_horizon_months,
                "parameter_category": "Risk",
                "parameter_name": "Maximum Peak Utilisation Threshold",
                "scope_level": "Global",
                "scope_value": "All",
                "value": default_peak_threshold,
                "unit": "pct",
                "driver_note": "Derived from workplace planning threshold.",
                "owner": "Session",
                "version_status": "Working",
            }
        )
    if to_add:
        assumptions = pd.concat([assumptions, pd.DataFrame(to_add)], ignore_index=True)
    return assumptions.reset_index(drop=True)


def build_working_assumptions(clean_sheets: dict[str, pd.DataFrame], scenario_name: str) -> pd.DataFrame:
    assumptions = clean_sheets.get("scenario_assumptions", pd.DataFrame()).copy()
    if assumptions.empty:
        return assumptions
    selected = assumptions[assumptions["scenario_name"].astype(str) == str(scenario_name)].copy()
    if selected.empty:
        selected = assumptions.copy()
    return _ensure_assumption_rows(clean_sheets, selected)


def _scope_priority(level: str) -> int:
    normalised = str(level or "").strip().lower()
    priorities = {
        "site": 6,
        "building": 5,
        "business unit": 4,
        "business_unit": 4,
        "city": 3,
        "country": 2,
        "regional": 1,
        "region": 1,
        "global": 0,
    }
    return priorities.get(normalised, 0)


def _scope_matches(scope_level: str, scope_value: Any, context: dict[str, Any]) -> bool:
    level = str(scope_level or "").strip().lower()
    value_text = str(scope_value or "").strip()
    if level == "global":
        return True
    if level in {"region", "regional"}:
        return str(context.get("region", "")) == value_text
    if level == "country":
        return str(context.get("country", "")) == value_text
    if level == "city":
        return str(context.get("city", "")) == value_text
    if level == "site":
        values = [item.strip() for item in value_text.split("|") if item.strip()]
        return str(context.get("site_id", "")) in values or str(context.get("site_name", "")) in values
    if level == "building":
        values = [item.strip() for item in value_text.split("|") if item.strip()]
        return str(context.get("building_name", "")) in values
    if level in {"business unit", "business_unit"}:
        return str(context.get("business_unit", "")) == value_text
    return False


def resolve_parameter_value(
    assumptions_df: pd.DataFrame,
    parameter_name: str,
    context: dict[str, Any],
    default: float | str | None = None,
) -> float | str | None:
    record = resolve_parameter_record(assumptions_df, parameter_name, context)
    if record is None:
        return default
    return record["value"]


def resolve_parameter_record(
    assumptions_df: pd.DataFrame,
    parameter_name: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if assumptions_df.empty:
        return None
    normalised_target = _normalise_parameter_name(parameter_name)
    matches = assumptions_df[
        assumptions_df["parameter_name"].map(_normalise_parameter_name) == normalised_target
    ].copy()
    if matches.empty:
        return None
    matches = matches[matches.apply(lambda row: _scope_matches(row["scope_level"], row["scope_value"], context), axis=1)]
    if matches.empty:
        return None
    matches["priority"] = matches["scope_level"].map(_scope_priority)
    matches = matches.sort_values(["priority", "value"], ascending=[False, False])
    return matches.iloc[0].to_dict()


def _parameter_source_label(record: dict[str, Any] | None, fallback: str) -> str:
    if not record:
        return fallback
    scope_level = str(record.get("scope_level", "Global"))
    scope_value = str(record.get("scope_value", "All"))
    return f"Assumption override: {scope_level} - {scope_value}"


def _resolve_numeric_parameter(
    assumptions_df: pd.DataFrame,
    parameter_name: str,
    context: dict[str, Any],
    *,
    default_value: float,
    default_source: str,
) -> tuple[float, str]:
    record = resolve_parameter_record(assumptions_df, parameter_name, context)
    if not record:
        return float(default_value), default_source
    value = pd.to_numeric(pd.Series([record.get("value")]), errors="coerce").iloc[0]
    return (float(value) if pd.notna(value) else float(default_value), _parameter_source_label(record, default_source))


def _business_unit_growth(assumptions_df: pd.DataFrame, context: dict[str, Any]) -> float:
    if assumptions_df.empty:
        return 0.0
    demand_rows = assumptions_df[
        assumptions_df["parameter_category"].astype(str).str.lower() == "demand"
    ].copy()
    if demand_rows.empty:
        return 0.0
    base_name = _normalise_parameter_name("Headcount Growth Default")
    demand_rows = demand_rows[demand_rows["parameter_name"].map(_normalise_parameter_name) != base_name]
    demand_rows = demand_rows[demand_rows.apply(lambda row: _scope_matches(row["scope_level"], row["scope_value"], context), axis=1)]
    if demand_rows.empty:
        return 0.0
    numeric = pd.to_numeric(demand_rows["value"], errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else 0.0


def _preferred_growth_hubs(assumptions_df: pd.DataFrame) -> set[str]:
    rows = assumptions_df[
        assumptions_df["parameter_name"].map(_normalise_parameter_name) == _normalise_parameter_name("Preferred Growth Hubs")
    ]
    if rows.empty:
        return set()
    values = str(rows.iloc[0]["scope_value"]).split("|")
    return {value.strip() for value in values if value.strip()}


def _growth_hub_bonus(row: pd.Series, preferred_growth_hubs: set[str], scenario_growth: float) -> float:
    if not preferred_growth_hubs or scenario_growth <= 0:
        return 0.0
    site_type = str(row.get("site_type", ""))
    site_id = str(row.get("site_id", ""))
    strategic_priority = str(row.get("strategic_priority", ""))
    if site_id in preferred_growth_hubs:
        base_bonus = {
            "HQ": SPACE_PLANNING_DEFAULTS["hq_growth_hub_bonus"],
            "Hub": SPACE_PLANNING_DEFAULTS["hub_growth_hub_bonus"],
            "Office": SPACE_PLANNING_DEFAULTS["office_growth_hub_bonus"],
        }.get(site_type, SPACE_PLANNING_DEFAULTS["office_growth_hub_bonus"])
        return base_bonus + (0.01 if strategic_priority == "Growth" else 0.0)
    if site_type == "Office" and strategic_priority == "Consolidate":
        return SPACE_PLANNING_DEFAULTS["non_hub_consolidation_drag"]
    return 0.0


def _site_type_peak_gap(site_type: str) -> float:
    mapping = {
        "HQ": SPACE_PLANNING_DEFAULTS["minimum_peak_gap_hq"],
        "Hub": SPACE_PLANNING_DEFAULTS["minimum_peak_gap_hub"],
        "Office": SPACE_PLANNING_DEFAULTS["minimum_peak_gap_office"],
    }
    return float(mapping.get(str(site_type or ""), SPACE_PLANNING_DEFAULTS["minimum_peak_gap_office"]))


def _interpolate_forecast(row: pd.Series, horizon_months: int) -> float:
    current = float(row.get("current_headcount", 0.0) or 0.0)
    fc12 = pd.to_numeric(pd.Series([row.get("forecast_headcount_12m")]), errors="coerce").iloc[0]
    fc24 = pd.to_numeric(pd.Series([row.get("forecast_headcount_24m")]), errors="coerce").iloc[0]
    if horizon_months <= 12:
        if pd.notna(fc12):
            share = max(float(horizon_months), 0.0) / 12.0
            return float(current + ((fc12 - current) * share))
        if pd.notna(fc24):
            share = max(float(horizon_months), 0.0) / 24.0
            return float(current + ((fc24 - current) * share))
        return current
    if pd.notna(fc12) and pd.notna(fc24) and 12 < horizon_months < 24:
        share = float(horizon_months - 12) / 12.0
        return float(fc12 + ((fc24 - fc12) * share))
    if pd.notna(fc24) and horizon_months >= 24:
        if pd.notna(fc12) and horizon_months > 24:
            delta_12_to_24 = fc24 - fc12
            extension_share = float(horizon_months - 24) / 12.0
            damped_extension = delta_12_to_24 * extension_share * SPACE_PLANNING_DEFAULTS["post_24m_growth_damping"]
            return float(max(fc24 + damped_extension, 0.0))
        return float(fc24)
    if pd.notna(fc12):
        extension_share = max(float(horizon_months - 12), 0.0) / 12.0
        damped_extension = (fc12 - current) * extension_share * 0.35
        return float(max(fc12 + damped_extension, 0.0))
    return current


def _base_scenario_reference_growth(clean_sheets: dict[str, pd.DataFrame]) -> float:
    assumptions = clean_sheets.get("scenario_assumptions", pd.DataFrame())
    if assumptions.empty:
        return 0.035
    base_rows = assumptions[assumptions["scenario_name"].astype(str).str.lower() == "base 2026"]
    if base_rows.empty:
        return 0.035
    value = resolve_parameter_value(base_rows, "Headcount Growth Default", {"site_id": "", "site_name": ""}, default=0.035)
    return float(value or 0.035)


def _normalise_scenario_outputs(outputs: pd.DataFrame) -> pd.DataFrame:
    if outputs.empty:
        return outputs.copy()

    normalised = outputs.copy()

    for column, default in [
        ("site_id", ""),
        ("site_name", ""),
        ("scenario_name", ""),
        ("region", ""),
        ("forecast_headcount", 0.0),
        ("required_seats", 0.0),
        ("existing_seats", 0.0),
        ("seat_gap", 0.0),
        ("required_area_sqm", 0.0),
        ("existing_usable_area_sqm", 0.0),
        ("area_gap_sqm", 0.0),
        ("action_flag", "Maintain"),
        ("action_reason", "No material change drivers were provided."),
        ("why_this_changed", "Imported scenario output."),
        ("risk_rating", "Low"),
        ("standards_compliance_score", 0.0),
        ("capacity_fit_score", np.nan),
        ("utilisation_fit_score", np.nan),
        ("implementation_simplicity_score", np.nan),
        ("consolidation_efficiency_score", np.nan),
    ]:
        if column not in normalised.columns:
            normalised[column] = default

    numeric_columns = [
        "forecast_headcount",
        "required_seats",
        "existing_seats",
        "seat_gap",
        "required_area_sqm",
        "existing_usable_area_sqm",
        "area_gap_sqm",
        "standards_compliance_score",
        "capacity_fit_score",
        "utilisation_fit_score",
        "implementation_simplicity_score",
        "consolidation_efficiency_score",
    ]
    for column in numeric_columns:
        normalised[column] = pd.to_numeric(normalised[column], errors="coerce")

    if "scenario_score" not in normalised.columns:
        component_columns = [
            "capacity_fit_score",
            "utilisation_fit_score",
            "standards_compliance_score",
            "implementation_simplicity_score",
            "consolidation_efficiency_score",
        ]
        if all(column in normalised.columns for column in component_columns):
            for column in component_columns:
                normalised[column] = pd.to_numeric(normalised[column], errors="coerce")
            normalised["scenario_score"] = (
                (normalised["capacity_fit_score"] * SCENARIO_SCORE_WEIGHTS["capacity_fit"] / 100.0)
                + (normalised["utilisation_fit_score"] * SCENARIO_SCORE_WEIGHTS["utilisation_fit"] / 100.0)
                + (normalised["standards_compliance_score"] * SCENARIO_SCORE_WEIGHTS["standards_alignment"] / 100.0)
                + (normalised["implementation_simplicity_score"] * SCENARIO_SCORE_WEIGHTS["implementation_simplicity"] / 100.0)
                + (normalised["consolidation_efficiency_score"] * SCENARIO_SCORE_WEIGHTS["consolidation_efficiency"] / 100.0)
            ).round(1)
        else:
            normalised["scenario_score"] = normalised["standards_compliance_score"].fillna(0.0).round(1)
    else:
        normalised["scenario_score"] = pd.to_numeric(normalised["scenario_score"], errors="coerce")

    for column in [
        "capacity_fit_score",
        "utilisation_fit_score",
        "implementation_simplicity_score",
        "consolidation_efficiency_score",
    ]:
        if normalised[column].isna().all():
            fallback = normalised["scenario_score"].fillna(normalised["standards_compliance_score"]).fillna(0.0)
            normalised[column] = fallback
        else:
            normalised[column] = normalised[column].fillna(normalised["scenario_score"]).fillna(
                normalised["standards_compliance_score"]
            )

    if "key_risk" not in normalised.columns:
        normalised["key_risk"] = np.select(
            [
                _series_or_default(normalised, "action_flag", "").astype(str).eq("Expand"),
                _series_or_default(normalised, "action_flag", "").astype(str).eq("Exit / Merge"),
                _series_or_default(normalised, "required_area_sqm", 0.0).fillna(0.0)
                > _series_or_default(normalised, "existing_usable_area_sqm", 0.0).fillna(0.0),
            ],
            [
                "Capacity deficit versus target demand.",
                "Demand transfer and exit sequencing required.",
                "Area uplift exceeds current usable area.",
            ],
            default="Monitor standards fit and utilisation alignment.",
        )
    else:
        normalised["key_risk"] = normalised["key_risk"].fillna("Monitor standards fit and utilisation alignment.")

    normalised["action_reason"] = normalised["action_reason"].fillna("No material change drivers were provided.")
    normalised["why_this_changed"] = normalised["why_this_changed"].fillna("Imported scenario output.")

    return normalised


def _scenario_summary_from_outputs(outputs: pd.DataFrame) -> dict[str, Any]:
    normalised = _normalise_scenario_outputs(outputs)
    forecast = pd.to_numeric(normalised["forecast_headcount"], errors="coerce")
    return {
        "forecast_headcount": float(forecast.fillna(0.0).sum()),
        "peak_attendance": float(pd.to_numeric(_series_or_default(normalised, "peak_attendance", 0.0), errors="coerce").fillna(0.0).sum()),
        "required_seats": float(pd.to_numeric(normalised["required_seats"], errors="coerce").fillna(0.0).sum()),
        "existing_seats": float(pd.to_numeric(normalised["existing_seats"], errors="coerce").fillna(0.0).sum()),
        "seat_gap": float(pd.to_numeric(normalised["seat_gap"], errors="coerce").fillna(0.0).sum()),
        "required_area_sqm": float(pd.to_numeric(normalised["required_area_sqm"], errors="coerce").fillna(0.0).sum()),
        "existing_usable_area_sqm": float(
            pd.to_numeric(_series_or_default(normalised, "existing_usable_area_sqm", 0.0), errors="coerce").fillna(0.0).sum()
        ),
        "area_gap_sqm": float(pd.to_numeric(normalised["area_gap_sqm"], errors="coerce").fillna(0.0).sum()),
        "standards_compliance_score": float(
            _weighted_average(normalised["standards_compliance_score"], forecast, default=0.0)
        ),
        "capacity_fit_score": float(_weighted_average(normalised["capacity_fit_score"], forecast, default=0.0)),
        "utilisation_fit_score": float(_weighted_average(normalised["utilisation_fit_score"], forecast, default=0.0)),
        "implementation_simplicity_score": float(
            _weighted_average(normalised["implementation_simplicity_score"], forecast, default=0.0)
        ),
        "consolidation_efficiency_score": float(
            _weighted_average(normalised["consolidation_efficiency_score"], forecast, default=0.0)
        ),
        "scenario_score": float(_weighted_average(normalised["scenario_score"], forecast, default=0.0)),
        "high_risk_sites": int((_series_or_default(normalised, "risk_rating", "").astype(str) == "High").sum()),
        "expand_sites": int((_series_or_default(normalised, "action_flag", "").astype(str) == "Expand").sum()),
        "consolidate_sites": int(
            (_series_or_default(normalised, "action_flag", "").astype(str) == "Consolidate / Release Space").sum()
        ),
        "maintain_sites": int((_series_or_default(normalised, "action_flag", "").astype(str) == "Maintain").sum()),
    }


def _risk_band(row: pd.Series) -> str:
    if (
        float(row.get("peak_threshold_delta", 0.0) or 0.0) >= RISK_THRESHOLD_DEFAULTS["high_peak_utilisation_buffer"]
        or float(row.get("seat_gap_ratio", 0.0) or 0.0) <= RISK_THRESHOLD_DEFAULTS["high_capacity_gap_ratio"]
        or float(row.get("area_gap_ratio", 0.0) or 0.0) <= RISK_THRESHOLD_DEFAULTS["high_area_gap_ratio"]
        or float(row.get("standards_compliance_score", 0.0) or 0.0) < RISK_THRESHOLD_DEFAULTS["high_score_floor"]
        or bool(row.get("capex_trigger_flag"))
        or float(row.get("transfer_shortfall_ratio", 0.0) or 0.0) > 0.20
    ):
        return "High"
    if (
        float(row.get("peak_threshold_delta", 0.0) or 0.0) >= RISK_THRESHOLD_DEFAULTS["medium_peak_utilisation_buffer"]
        or float(row.get("seat_gap_ratio", 0.0) or 0.0) <= RISK_THRESHOLD_DEFAULTS["medium_capacity_gap_ratio"]
        or float(row.get("area_gap_ratio", 0.0) or 0.0) <= RISK_THRESHOLD_DEFAULTS["medium_area_gap_ratio"]
        or float(row.get("standards_compliance_score", 0.0) or 0.0) < RISK_THRESHOLD_DEFAULTS["medium_score_floor"]
        or float(row.get("actual_transfer_ratio", 0.0) or 0.0) > 0.0
    ):
        return "Medium"
    return "Low"


def _action_reason(row: pd.Series) -> str:
    action = str(row.get("action_flag", "Maintain"))
    seat_gap_ratio = float(row.get("seat_gap_ratio", 0.0) or 0.0)
    area_gap_ratio = float(row.get("area_gap_ratio", 0.0) or 0.0)
    peak_threshold_delta = float(row.get("peak_threshold_delta", 0.0) or 0.0)
    actual_transfer_ratio = float(row.get("actual_transfer_ratio", 0.0) or 0.0)
    if action == "Expand":
        return (
            f"Peak demand sits {abs(seat_gap_ratio):.0%} above seat capacity planning coverage and "
            f"{max(peak_threshold_delta, 0.0):.0%} above the peak utilisation threshold."
        )
    if action == "Exit / Merge":
        return (
            f"{actual_transfer_ratio:.0%} of forecast demand can be re-routed into recipient hubs/HQs, making site exit or merge credible."
        )
    if action == "Consolidate / Release Space":
        return (
            f"Seat surplus of {max(seat_gap_ratio, 0.0):.0%} and area surplus of {max(area_gap_ratio, 0.0):.0%} sit alongside subdued peak pressure."
        )
    if action == "Re-stack / Rebalance":
        return "Overall capacity is workable, but standards fit, meeting/support provision, or transfer effects still require intervention."
    return "Capacity, utilisation, and standards remain inside the maintainable planning tolerance band."


def _key_risk_from_row(row: pd.Series) -> str:
    if bool(row.get("capex_trigger_flag")):
        return "Area uplift exceeds the capex trigger threshold."
    if float(row.get("transfer_shortfall_ratio", 0.0) or 0.0) > 0.20:
        return "Consolidation demand cannot be fully absorbed by the preferred recipient portfolio."
    if str(row.get("action_flag", "")) == "Expand":
        return "Capacity deficit versus planned peak demand remains material."
    if str(row.get("action_flag", "")) == "Exit / Merge":
        return "Demand transfer and exit sequencing need careful delivery governance."
    return "Monitor standards fit and utilisation alignment."


def _why_this_changed(row: pd.Series) -> str:
    drivers: list[str] = []
    current_headcount = float(row.get("current_headcount", 0.0) or 0.0)
    forecast_headcount = float(row.get("forecast_headcount", 0.0) or 0.0)
    if current_headcount > 0:
        growth_delta = (forecast_headcount - current_headcount) / current_headcount
        if abs(growth_delta) >= 0.05:
            drivers.append(f"forecast headcount {'up' if growth_delta > 0 else 'down'} {abs(growth_delta):.0%}")
    attendance_delta = float(row.get("avg_attendance_rate", 0.0) or 0.0) - float(row.get("avg_attendance_pct_base", 0.0) or 0.0)
    if abs(attendance_delta) >= 0.03 or float(row.get("anchor_days", 0.0) or 0.0) > 0.0:
        anchor_text = ""
        if float(row.get("anchor_days", 0.0) or 0.0) > 0.0:
            anchor_text = f", anchor floor {float(row.get('anchor_days', 0.0) or 0.0):.0f} day(s)"
        drivers.append(f"attendance reset to {float(row.get('avg_attendance_rate', 0.0) or 0.0):.0%}{anchor_text}")
    desk_delta = float(row.get("desk_sharing_ratio_target", 0.0) or 0.0) - float(row.get("default_desk_sharing_ratio_target", 0.0) or 0.0)
    if abs(desk_delta) >= 0.05:
        drivers.append(f"desk sharing moved to {float(row.get('desk_sharing_ratio_target', 0.0) or 0.0):.2f}x")
    sqm_delta = float(row.get("sqm_per_person_target", 0.0) or 0.0) - float(row.get("default_sqm_per_person_target", 0.0) or 0.0)
    if abs(sqm_delta) >= 0.30:
        drivers.append(f"space standard reset to {float(row.get('sqm_per_person_target', 0.0) or 0.0):.1f} sqm/person")
    if float(row.get("growth_hub_bonus", 0.0) or 0.0) > 0.0:
        drivers.append("growth hub prioritisation adds demand concentration")
    if float(row.get("transfer_in_headcount", 0.0) or 0.0) > 0.0:
        drivers.append("receives transferred demand from consolidation candidates")
    if float(row.get("transfer_out_headcount", 0.0) or 0.0) > 0.0:
        drivers.append("transfers part of its demand into recipient hubs/HQs")
    if not drivers:
        return "The scenario stays close to current demand and default planning standards for this site."
    return "; ".join(drivers[:3]).capitalize() + "."


def _compute_site_metrics(site_table: pd.DataFrame) -> pd.DataFrame:
    if site_table.empty:
        return site_table
    outputs = site_table.copy()
    outputs["space_inventory_supported"] = _series_or_default(outputs, "space_inventory_supported", False).fillna(False)
    outputs["occupancy_supported"] = _series_or_default(outputs, "occupancy_supported", False).fillna(False)
    outputs["existing_meeting_capacity"] = _series_or_default(outputs, "existing_meeting_capacity", 0.0).fillna(0.0)
    outputs["existing_focus_capacity"] = _series_or_default(outputs, "existing_focus_capacity", 0.0).fillna(0.0)
    outputs["existing_collaboration_area_sqm"] = _series_or_default(outputs, "existing_collaboration_area_sqm", 0.0).fillna(0.0)
    outputs["seat_ratio_target"] = _series_or_default(outputs, "seat_ratio_target", np.nan)
    implied_seat_ratio = 1 / outputs["desk_sharing_ratio_target"].replace(0, np.nan)
    outputs["seat_ratio_target"] = outputs["seat_ratio_target"].fillna(implied_seat_ratio).fillna(0.8)
    outputs["focus_seats_per_100_staff"] = _series_or_default(outputs, "focus_seats_per_100_staff", 8.0).fillna(8.0)

    outputs["avg_attendance_demand"] = outputs["forecast_headcount"].fillna(0.0) * outputs["avg_attendance_rate"].fillna(0.0)
    outputs["peak_attendance"] = outputs["forecast_headcount"].fillna(0.0) * outputs["peak_attendance_rate"].fillna(0.0)
    outputs["required_seats"] = np.where(
        outputs["desk_sharing_ratio_target"].fillna(0.0) > 0,
        outputs["peak_attendance"] / outputs["desk_sharing_ratio_target"].replace(0, np.nan),
        outputs["peak_attendance"],
    )
    outputs["required_area_sqm_base"] = outputs["forecast_headcount"].fillna(0.0) * outputs["sqm_per_person_target"].fillna(0.0)
    outputs["collaboration_area_required_sqm"] = (
        outputs["required_area_sqm_base"].fillna(0.0) * outputs["collaboration_area_target"].fillna(0.0)
    )
    outputs["required_meeting_seats"] = (
        outputs["forecast_headcount"].fillna(0.0) * outputs["meeting_seats_per_100_staff"].fillna(0.0) / 100.0
    )
    outputs["required_focus_seats"] = (
        outputs["forecast_headcount"].fillna(0.0) * outputs["focus_seats_per_100_staff"].fillna(0.0) / 100.0
    )
    outputs["required_meeting_rooms"] = np.ceil(outputs["required_meeting_seats"] / DEFAULT_ROOM_SEATS)

    collaboration_shortfall_ratio = _safe_ratio_series(
        (outputs["collaboration_area_required_sqm"] - outputs["existing_collaboration_area_sqm"]).clip(lower=0.0),
        outputs["collaboration_area_required_sqm"].replace(0, np.nan),
        default=0.0,
    )
    meeting_shortfall_ratio = _safe_ratio_series(
        (outputs["required_meeting_seats"] - outputs["existing_meeting_capacity"]).clip(lower=0.0),
        outputs["required_meeting_seats"].replace(0, np.nan),
        default=0.0,
    )
    focus_shortfall_ratio = _safe_ratio_series(
        (outputs["required_focus_seats"] - outputs["existing_focus_capacity"]).clip(lower=0.0),
        outputs["required_focus_seats"].replace(0, np.nan),
        default=0.0,
    )
    outputs["collaboration_shortfall_ratio"] = np.where(outputs["space_inventory_supported"], collaboration_shortfall_ratio, 0.0)
    outputs["meeting_shortfall_ratio"] = np.where(outputs["space_inventory_supported"], meeting_shortfall_ratio, 0.0)
    outputs["focus_shortfall_ratio"] = np.where(outputs["space_inventory_supported"], focus_shortfall_ratio, 0.0)

    outputs["space_support_area_uplift_factor"] = 1.0 + (
        (outputs["collaboration_shortfall_ratio"] * SPACE_PLANNING_DEFAULTS["collaboration_area_uplift_weight"])
        + (outputs["meeting_shortfall_ratio"] * SPACE_PLANNING_DEFAULTS["meeting_area_uplift_weight"])
        + (outputs["focus_shortfall_ratio"] * SPACE_PLANNING_DEFAULTS["focus_area_uplift_weight"])
    )
    outputs["required_area_sqm"] = outputs["required_area_sqm_base"].fillna(0.0) * outputs["space_support_area_uplift_factor"].fillna(1.0)
    outputs["seat_gap"] = outputs["existing_seats"].fillna(0.0) - outputs["required_seats"].fillna(0.0)
    outputs["area_gap_sqm"] = outputs["existing_usable_area_sqm"].fillna(0.0) - outputs["required_area_sqm"].fillna(0.0)
    outputs["peak_utilisation_pct_live"] = np.where(
        outputs["existing_seats"].fillna(0.0) > 0,
        outputs["peak_attendance"] / outputs["existing_seats"].replace(0, np.nan),
        np.nan,
    )

    seat_reference = np.maximum.reduce(
        [
            outputs["existing_seats"].fillna(0.0).to_numpy(),
            outputs["required_seats"].fillna(0.0).to_numpy(),
            np.ones(len(outputs)),
        ]
    )
    area_reference = np.maximum.reduce(
        [
            outputs["existing_usable_area_sqm"].fillna(0.0).to_numpy(),
            outputs["required_area_sqm"].fillna(0.0).to_numpy(),
            np.ones(len(outputs)),
        ]
    )
    outputs["seat_gap_ratio"] = outputs["seat_gap"].fillna(0.0) / seat_reference
    outputs["area_gap_ratio"] = outputs["area_gap_sqm"].fillna(0.0) / area_reference
    outputs["peak_threshold_delta"] = outputs["peak_utilisation_pct_live"].fillna(0.0) - outputs["peak_utilisation_threshold"].fillna(0.0)

    seat_deficit_ratio = (-outputs["seat_gap_ratio"]).clip(lower=0.0)
    seat_surplus_ratio = outputs["seat_gap_ratio"].clip(lower=0.0)
    area_deficit_ratio = (-outputs["area_gap_ratio"]).clip(lower=0.0)
    area_surplus_ratio = outputs["area_gap_ratio"].clip(lower=0.0)
    peak_excess_ratio = outputs["peak_threshold_delta"].clip(lower=0.0)
    under_utilisation_ratio = (
        outputs["peak_utilisation_threshold"].fillna(0.0) - 0.12 - outputs["peak_utilisation_pct_live"].fillna(0.0)
    ).clip(lower=0.0)

    seat_fit_score = (100.0 - (seat_deficit_ratio * 220.0) - (seat_surplus_ratio * 105.0)).clip(0.0, 100.0)
    area_fit_score = (100.0 - (area_deficit_ratio * 205.0) - (area_surplus_ratio * 90.0)).clip(0.0, 100.0)
    utilisation_fit_score = (100.0 - (peak_excess_ratio * 260.0) - (under_utilisation_ratio * 140.0)).clip(0.0, 100.0)
    seat_ratio_alignment_score = (
        100.0 - ((implied_seat_ratio.fillna(outputs["seat_ratio_target"]) - outputs["seat_ratio_target"]).abs() * 220.0)
    ).clip(0.0, 100.0)
    collaboration_alignment_score = (
        100.0 - (outputs["collaboration_shortfall_ratio"] * 120.0)
    ).clip(0.0, 100.0)
    meeting_alignment_score = (100.0 - (outputs["meeting_shortfall_ratio"] * 110.0)).clip(0.0, 100.0)
    focus_alignment_score = (100.0 - (outputs["focus_shortfall_ratio"] * 100.0)).clip(0.0, 100.0)

    outputs["capacity_fit_score"] = ((seat_fit_score * 0.65) + (area_fit_score * 0.35)).round(1)
    outputs["utilisation_fit_score"] = utilisation_fit_score.round(1)
    outputs["standards_compliance_score"] = (
        (seat_ratio_alignment_score * 0.30)
        + (collaboration_alignment_score * 0.25)
        + (meeting_alignment_score * 0.25)
        + (focus_alignment_score * 0.20)
    ).round(1)

    force_action_flag = outputs.get("force_action_flag", pd.Series(index=outputs.index, dtype=object)).fillna("")
    actual_transfer_ratio = outputs.get("actual_transfer_ratio", pd.Series(index=outputs.index, dtype=float)).fillna(0.0)
    transfer_target_ratio = outputs.get("transfer_target_ratio", pd.Series(index=outputs.index, dtype=float)).fillna(0.0)
    transfer_in_headcount = _series_or_default(outputs, "transfer_in_headcount", 0.0).fillna(0.0)
    transfer_out_headcount = _series_or_default(outputs, "transfer_out_headcount", 0.0).fillna(0.0)
    outputs["transfer_shortfall_ratio"] = (transfer_target_ratio - actual_transfer_ratio).clip(lower=0.0)

    deficit_mask = (
        (outputs["seat_gap_ratio"] <= ACTION_THRESHOLD_DEFAULTS["expand_seat_deficit_ratio"])
        | (outputs["area_gap_ratio"] <= ACTION_THRESHOLD_DEFAULTS["expand_area_deficit_ratio"])
        | (outputs["peak_threshold_delta"] >= ACTION_THRESHOLD_DEFAULTS["expand_peak_utilisation_buffer"])
    )
    release_mask = (
        ((outputs["seat_gap_ratio"] >= ACTION_THRESHOLD_DEFAULTS["release_seat_surplus_ratio"])
         | (outputs["area_gap_ratio"] >= ACTION_THRESHOLD_DEFAULTS["release_area_surplus_ratio"]))
        & (outputs["peak_threshold_delta"] <= -ACTION_THRESHOLD_DEFAULTS["release_utilisation_buffer"])
        & (outputs["criticality"].astype(str) != "Strategic")
    )
    maintain_mask = (
        outputs["seat_gap_ratio"].abs() <= ACTION_THRESHOLD_DEFAULTS["maintain_seat_tolerance_ratio"]
    ) & (
        outputs["area_gap_ratio"].abs() <= ACTION_THRESHOLD_DEFAULTS["maintain_area_tolerance_ratio"]
    ) & (
        outputs["standards_compliance_score"] >= ACTION_THRESHOLD_DEFAULTS["maintain_score_floor"]
    ) & (
        outputs["peak_threshold_delta"].abs() <= 0.05
    )
    outputs["action_flag"] = np.select(
        [
            force_action_flag.astype(str).eq("Exit / Merge")
            | (actual_transfer_ratio >= ACTION_THRESHOLD_DEFAULTS["exit_transfer_ratio"]),
            deficit_mask,
            release_mask,
            maintain_mask,
        ],
        [
            "Exit / Merge",
            "Expand",
            "Consolidate / Release Space",
            "Maintain",
        ],
        default="Re-stack / Rebalance",
    )

    complexity_penalty = (
        np.where(outputs["action_flag"].isin(["Expand", "Exit / Merge"]), 32.0, 0.0)
        + np.where(outputs["action_flag"].eq("Re-stack / Rebalance"), 16.0, 0.0)
        + np.where(outputs["criticality"].astype(str) == "Strategic", 18.0, 0.0)
        + np.where(outputs["building_count"].fillna(0.0) > 1, 10.0, 0.0)
        + np.where(outputs["floor_count"].fillna(0.0) > 6, 10.0, 0.0)
        + np.where(outputs["lease_expiry_within_24m"].fillna(False), 8.0, 0.0)
        + np.where((transfer_in_headcount > 0.0) | (transfer_out_headcount > 0.0), 12.0, 0.0)
    )
    implementation_score = (100.0 - complexity_penalty).clip(20.0, 95.0)
    outputs["estimated_move_complexity"] = np.select(
        [implementation_score < 50.0, implementation_score < 75.0],
        ["High", "Medium"],
        default="Low",
    )
    outputs["implementation_simplicity_score"] = implementation_score.round(1)

    release_efficiency_score = ((seat_surplus_ratio * 60.0) + (area_surplus_ratio * 40.0)).clip(0.0, 100.0)
    transfer_efficiency_score = np.where(
        transfer_target_ratio > 0,
        (actual_transfer_ratio / transfer_target_ratio.replace(0, np.nan)).clip(0.0, 1.0) * 100.0,
        np.where(transfer_out_headcount > 0.0, 60.0, 70.0),
    )
    outputs["consolidation_efficiency_score"] = np.select(
        [
            outputs["action_flag"].isin(["Consolidate / Release Space", "Exit / Merge"]),
            outputs["action_flag"].eq("Expand"),
        ],
        [
            ((release_efficiency_score * 0.55) + (transfer_efficiency_score * 0.45)).clip(0.0, 100.0),
            (45.0 - (seat_deficit_ratio * 120.0)).clip(0.0, 100.0),
        ],
        default=(55.0 + (release_efficiency_score * 0.20) - (seat_deficit_ratio * 55.0)).clip(0.0, 100.0),
    ).round(1)
    outputs["scenario_score"] = (
        (outputs["capacity_fit_score"] * SCENARIO_SCORE_WEIGHTS["capacity_fit"] / 100.0)
        + (outputs["utilisation_fit_score"] * SCENARIO_SCORE_WEIGHTS["utilisation_fit"] / 100.0)
        + (outputs["standards_compliance_score"] * SCENARIO_SCORE_WEIGHTS["standards_alignment"] / 100.0)
        + (outputs["implementation_simplicity_score"] * SCENARIO_SCORE_WEIGHTS["implementation_simplicity"] / 100.0)
        + (outputs["consolidation_efficiency_score"] * SCENARIO_SCORE_WEIGHTS["consolidation_efficiency"] / 100.0)
    ).round(1)
    outputs["capex_trigger_flag"] = (
        outputs["required_area_sqm"] > (outputs["existing_usable_area_sqm"] * (1 + outputs["capex_trigger_threshold"]))
    )
    outputs["risk_rating"] = outputs.apply(_risk_band, axis=1)
    outputs["action_reason"] = outputs.apply(_action_reason, axis=1)
    outputs["key_risk"] = outputs.apply(_key_risk_from_row, axis=1)
    outputs["why_this_changed"] = outputs.apply(_why_this_changed, axis=1)
    return outputs


def _apply_consolidation_transfers(site_table: pd.DataFrame, assumptions_df: pd.DataFrame) -> pd.DataFrame:
    if site_table.empty:
        return site_table
    exits = resolve_parameter_value(assumptions_df, "Exit Small Office Sites", {}, default=0.0)
    if not exits:
        return site_table
    hub_share = float(resolve_parameter_value(assumptions_df, "Transfer Demand to Hubs", {}, default=0.0) or 0.0)
    hq_share = float(resolve_parameter_value(assumptions_df, "Transfer Demand to HQs", {}, default=0.0) or 0.0)
    total_share = min(hub_share + hq_share, 1.0)
    if total_share <= 0:
        return site_table

    outputs = site_table.copy()
    outputs["transfer_in_headcount"] = _series_or_default(outputs, "transfer_in_headcount", 0.0).fillna(0.0)
    outputs["transfer_out_headcount"] = _series_or_default(outputs, "transfer_out_headcount", 0.0).fillna(0.0)
    outputs["transfer_target_ratio"] = _series_or_default(outputs, "transfer_target_ratio", 0.0).fillna(0.0)
    outputs["actual_transfer_ratio"] = _series_or_default(outputs, "actual_transfer_ratio", 0.0).fillna(0.0)
    outputs["consolidation_target_sites"] = _series_or_default(outputs, "consolidation_target_sites", "").fillna("")
    candidates = outputs[
        (outputs["site_type"].astype(str) == "Office")
        & (outputs["criticality"].astype(str) != "Strategic")
    ].copy()
    if candidates.empty:
        return outputs
    cost_density = candidates["annual_property_cost_eur"].fillna(0.0) / candidates["existing_usable_area_sqm"].replace(0, np.nan)
    release_opportunity = (
        candidates["seat_gap_ratio"].clip(lower=0.0).fillna(0.0) * 0.45
        + candidates["area_gap_ratio"].clip(lower=0.0).fillna(0.0) * 0.20
        + np.where(candidates["lease_expiry_within_24m"].fillna(False), 0.20, 0.0)
        + (candidates["peak_threshold_delta"].mul(-1).clip(lower=0.0).fillna(0.0) * 0.15)
        + (cost_density.rank(pct=True, method="average").fillna(0.0) * 0.20)
    )
    candidates["_release_rank"] = release_opportunity
    candidates = candidates.sort_values(
        ["_release_rank", "forecast_headcount", "annual_property_cost_eur"],
        ascending=[False, True, False],
    ).head(int(float(exits)))
    exit_site_ids = set(candidates["site_id"].astype(str))

    for _, candidate in candidates.iterrows():
        site_region = candidate.get("region")
        site_country = candidate.get("country")
        site_city = candidate.get("city")
        original_forecast = float(candidate.get("forecast_headcount", 0.0) or 0.0)
        forecast_to_move = original_forecast * total_share
        if forecast_to_move <= 0:
            continue

        remaining_to_move = forecast_to_move
        recipient_labels: list[str] = []
        for share_ratio, site_type in [(hub_share, "Hub"), (hq_share, "HQ")]:
            if share_ratio <= 0 or remaining_to_move <= 0:
                continue
            recipients = outputs[
                (~outputs["site_id"].astype(str).isin(exit_site_ids))
                & (outputs["site_type"].astype(str) == site_type)
            ].copy()
            if recipients.empty:
                continue
            same_city = recipients[(recipients["city"] == site_city) & (recipients["country"] == site_country)]
            same_country = recipients[recipients["country"] == site_country]
            same_region = recipients[recipients["region"] == site_region]
            if not same_city.empty:
                recipients = same_city
            elif not same_country.empty:
                recipients = same_country
            elif not same_region.empty:
                recipients = same_region

            recipient_capacity = (
                recipients["seat_gap"].clip(lower=0.0).fillna(0.0)
                * recipients["desk_sharing_ratio_target"].fillna(1.0)
                / recipients["peak_attendance_rate"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if float(recipient_capacity.sum()) <= 0.0:
                recipient_capacity = recipients["existing_seats"].fillna(0.0) * 0.05
            priority_weight = np.where(
                recipients["dominant_strategic_priority"].astype(str) == "Growth",
                1.20,
                1.00,
            )
            weights = (recipient_capacity.clip(lower=0.0) * priority_weight).fillna(0.0)
            if float(weights.sum()) <= 0.0:
                continue

            target_forecast = forecast_to_move * (share_ratio / total_share)
            allocatable = min(target_forecast, float(recipient_capacity.sum()))
            allocations = allocatable * (weights / weights.sum())
            for recipient_site_id, allocation in zip(recipients["site_id"], allocations, strict=False):
                if allocation <= 0:
                    continue
                outputs.loc[outputs["site_id"] == recipient_site_id, "forecast_headcount"] += float(allocation)
                outputs.loc[outputs["site_id"] == recipient_site_id, "transfer_in_headcount"] += float(allocation)
                recipient_labels.append(str(recipient_site_id))
                remaining_to_move -= float(allocation)

        moved_headcount = max(forecast_to_move - remaining_to_move, 0.0)
        if moved_headcount <= 0:
            continue
        outputs.loc[outputs["site_id"] == candidate["site_id"], "forecast_headcount"] -= moved_headcount
        outputs.loc[outputs["site_id"] == candidate["site_id"], "transfer_out_headcount"] += moved_headcount
        outputs.loc[outputs["site_id"] == candidate["site_id"], "transfer_target_ratio"] = total_share
        outputs.loc[outputs["site_id"] == candidate["site_id"], "actual_transfer_ratio"] = (
            moved_headcount / original_forecast if original_forecast > 0 else 0.0
        )
        outputs.loc[outputs["site_id"] == candidate["site_id"], "consolidation_target_sites"] = "|".join(
            sorted(set(recipient_labels))
        )
        if original_forecast > 0 and (moved_headcount / original_forecast) >= ACTION_THRESHOLD_DEFAULTS["exit_transfer_ratio"]:
            outputs.loc[outputs["site_id"] == candidate["site_id"], "force_action_flag"] = "Exit / Merge"
    return outputs


def compute_live_scenario(
    clean_sheets: dict[str, pd.DataFrame],
    assumptions_df: pd.DataFrame,
    filters: dict[str, Any],
    *,
    workbook_name: str,
    workbook_hash: str,
) -> dict[str, Any]:
    assumptions = _ensure_assumption_rows(clean_sheets, assumptions_df)
    scenario_name = str(assumptions["scenario_name"].iloc[0]) if not assumptions.empty else "Live Scenario"
    horizon_months = _base_horizon_months(assumptions)
    baseline = build_portfolio_baseline(clean_sheets, filters)
    site_table = baseline["site_table"].copy()
    people = _scaled_people_demand(clean_sheets, baseline["filtered_frames"]).copy()
    if site_table.empty or people.empty:
        return {
            "scenario_name": scenario_name,
            "origin": "Live Scenario",
            "assumptions": assumptions,
            "outputs": pd.DataFrame(),
            "summary": {},
            "warnings": ["No filtered rows are available for scenario calculation."],
            "workbook_name": workbook_name,
            "workbook_hash": workbook_hash,
        }

    reference_growth = _base_scenario_reference_growth(clean_sheets)
    preferred_growth_hubs = _preferred_growth_hubs(assumptions)
    people = people.merge(
        site_table[
            [
                "site_id",
                "region",
                "country",
                "city",
                "site_name",
                "site_type",
                "criticality",
                "dominant_strategic_priority",
                "occupancy_supported",
                "space_inventory_supported",
                "observed_avg_attendance_rate",
                "observed_peak_attendance_rate",
                "default_standard_precedence",
                "default_seat_ratio_target_source",
                "default_seat_ratio_target",
                "default_desk_sharing_ratio_target_source",
                "default_desk_sharing_ratio_target",
                "default_sqm_per_person_target_source",
                "default_sqm_per_person_target",
                "default_collaboration_area_pct_target_source",
                "default_collaboration_area_pct_target",
                "default_meeting_seats_per_100_staff_source",
                "default_meeting_seats_per_100_staff",
                "default_focus_seats_per_100_staff_source",
                "default_focus_seats_per_100_staff",
                "default_planning_utilisation_threshold_pct_source",
                "default_planning_utilisation_threshold_pct",
            ]
        ],
        on=["site_id", "region", "country", "city", "site_name", "site_type"],
        how="left",
    )

    detail_rows = []
    for _, row in people.iterrows():
        context = row.to_dict()
        growth_record = resolve_parameter_record(assumptions, "Headcount Growth Default", context)
        default_growth = float(growth_record["value"]) if growth_record else float(reference_growth)
        bu_growth = _business_unit_growth(assumptions, context)
        growth_bonus = _growth_hub_bonus(row, preferred_growth_hubs, default_growth + bu_growth)
        base_forecast = _interpolate_forecast(row, horizon_months)
        if pd.isna(base_forecast) or base_forecast <= 0:
            base_forecast = float(row.get("current_headcount", 0.0) or 0.0)
        forecast_headcount = base_forecast * (1 + (default_growth - reference_growth) + bu_growth + growth_bonus)
        forecast_headcount = max(forecast_headcount, 0.0)

        observed_avg = pd.to_numeric(pd.Series([row.get("observed_avg_attendance_rate")]), errors="coerce").iloc[0]
        observed_peak = pd.to_numeric(pd.Series([row.get("observed_peak_attendance_rate")]), errors="coerce").iloc[0]
        base_avg = float(row.get("avg_attendance_pct", 0.0) or 0.0)
        if pd.notna(observed_avg):
            base_avg = max(base_avg, min(float(observed_avg), base_avg + 0.05))
        base_peak = float(row.get("peak_attendance_pct", 0.0) or 0.0)
        if pd.notna(observed_peak):
            base_peak = max(base_peak, float(observed_peak))

        avg_uplift_record = resolve_parameter_record(assumptions, "Average Attendance Uplift", context)
        avg_uplift = float(avg_uplift_record["value"]) if avg_uplift_record else 0.0
        anchor_record = resolve_parameter_record(assumptions, "Mandatory Anchor Days", context)
        anchor_days = anchor_record["value"] if anchor_record else np.nan
        anchor_floor = float(anchor_days) / 5.0 if pd.notna(anchor_days) else 0.0
        avg_rate = max(base_avg + avg_uplift, anchor_floor)
        avg_rate = float(np.clip(avg_rate, ATTENDANCE_MIN_RATE, 1.0))

        peak_buffer_record = resolve_parameter_record(assumptions, "Peak Attendance Buffer", context)
        peak_buffer = float(peak_buffer_record["value"]) if peak_buffer_record else 0.08
        baseline_peak_gap = max(base_peak - base_avg, _site_type_peak_gap(str(row.get("site_type", ""))))
        peak_gap = max(peak_buffer, baseline_peak_gap * 0.75, max(anchor_floor - avg_rate, 0.0) + (_site_type_peak_gap(str(row.get("site_type", ""))) / 2.0))
        peak_rate = max(base_peak + max(avg_rate - base_avg, 0.0) * 0.35, avg_rate + peak_gap)
        peak_rate = float(np.clip(peak_rate, avg_rate, ATTENDANCE_MAX_RATE))

        detail_rows.append(
            {
                "site_id": row["site_id"],
                "business_unit": row.get("business_unit"),
                "current_headcount": row.get("current_headcount", 0.0),
                "forecast_headcount": forecast_headcount,
                "avg_attendance_rate": avg_rate,
                "peak_attendance_rate": peak_rate,
                "growth_hub_bonus": growth_bonus,
                "anchor_days": float(anchor_days) if pd.notna(anchor_days) else 0.0,
            }
        )

    demand_detail = pd.DataFrame(detail_rows)
    site_demand_rows = []
    for site_id, group in demand_detail.groupby("site_id", dropna=False):
        weights = group["forecast_headcount"].fillna(group["current_headcount"]).clip(lower=0.0)
        site_demand_rows.append(
            {
                "site_id": site_id,
                "forecast_headcount": float(group["forecast_headcount"].fillna(0.0).sum()),
                "avg_attendance_rate": _weighted_average(group["avg_attendance_rate"], weights, default=0.5),
                "peak_attendance_rate": _weighted_average(group["peak_attendance_rate"], weights, default=0.7),
                "growth_hub_bonus": _weighted_average(group["growth_hub_bonus"], weights, default=0.0),
                "anchor_days": float(group["anchor_days"].fillna(0.0).max()),
            }
        )
    site_demand = pd.DataFrame(site_demand_rows)

    outputs = site_table.merge(site_demand, on="site_id", how="left")
    outputs["forecast_headcount"] = outputs["forecast_headcount"].fillna(outputs["forecast_headcount_12m"]).fillna(
        outputs["current_headcount"]
    )
    outputs["avg_attendance_rate"] = outputs["avg_attendance_rate"].fillna(outputs["avg_attendance_pct_base"]).fillna(0.5)
    outputs["peak_attendance_rate"] = outputs["peak_attendance_rate"].fillna(outputs["peak_attendance_pct_base"]).fillna(0.7)

    planning_rows = []
    for _, row in outputs.iterrows():
        context = row.to_dict()
        desk_share_value, desk_share_source = _resolve_numeric_parameter(
            assumptions,
            "Desk Sharing Ratio Target",
            context,
            default_value=float(row.get("default_desk_sharing_ratio_target", 1.2) or 1.2),
            default_source=str(row.get("default_desk_sharing_ratio_target_source", "Workstyle/site type default")),
        )
        sqm_value, sqm_source = _resolve_numeric_parameter(
            assumptions,
            "sqm per Person Target",
            context,
            default_value=float(row.get("default_sqm_per_person_target", 9.0) or 9.0),
            default_source=str(row.get("default_sqm_per_person_target_source", "Workstyle/site type default")),
        )
        meeting_value, meeting_source = _resolve_numeric_parameter(
            assumptions,
            "Meeting Seats per 100 Staff",
            context,
            default_value=float(row.get("default_meeting_seats_per_100_staff", 14.0) or 14.0),
            default_source=str(row.get("default_meeting_seats_per_100_staff_source", "Workstyle/site type default")),
        )
        focus_value, focus_source = _resolve_numeric_parameter(
            assumptions,
            "Focus Seats per 100 Staff",
            context,
            default_value=float(row.get("default_focus_seats_per_100_staff", 8.0) or 8.0),
            default_source=str(row.get("default_focus_seats_per_100_staff_source", "Workstyle/site type default")),
        )
        collaboration_value, collaboration_source = _resolve_numeric_parameter(
            assumptions,
            "Collaboration Area Target",
            context,
            default_value=float(row.get("default_collaboration_area_pct_target", 0.18) or 0.18),
            default_source=str(row.get("default_collaboration_area_pct_target_source", "Workstyle/site type default")),
        )
        peak_threshold_value, peak_threshold_source = _resolve_numeric_parameter(
            assumptions,
            "Maximum Peak Utilisation Threshold",
            context,
            default_value=float(row.get("default_planning_utilisation_threshold_pct", 0.82) or 0.82),
            default_source=str(row.get("default_planning_utilisation_threshold_pct_source", "Workstyle/site type default")),
        )
        capex_value, capex_source = _resolve_numeric_parameter(
            assumptions,
            "Capex Trigger Threshold",
            context,
            default_value=0.10,
            default_source="Model default",
        )
        seat_ratio_value, seat_ratio_source = _resolve_numeric_parameter(
            assumptions,
            "Seat Ratio Target",
            context,
            default_value=float(row.get("default_seat_ratio_target", 0.8) or 0.8),
            default_source=str(row.get("default_seat_ratio_target_source", "Workstyle/site type default")),
        )
        planning_rows.append(
            {
                "site_id": row["site_id"],
                "desk_sharing_ratio_target": desk_share_value,
                "desk_sharing_ratio_target_source": desk_share_source,
                "sqm_per_person_target": sqm_value,
                "sqm_per_person_target_source": sqm_source,
                "meeting_seats_per_100_staff": meeting_value,
                "meeting_seats_per_100_staff_source": meeting_source,
                "focus_seats_per_100_staff": focus_value,
                "focus_seats_per_100_staff_source": focus_source,
                "collaboration_area_target": collaboration_value,
                "collaboration_area_target_source": collaboration_source,
                "peak_utilisation_threshold": peak_threshold_value,
                "peak_utilisation_threshold_source": peak_threshold_source,
                "capex_trigger_threshold": capex_value,
                "capex_trigger_threshold_source": capex_source,
                "seat_ratio_target": seat_ratio_value,
                "seat_ratio_target_source": seat_ratio_source,
                "standards_precedence_used": row.get("default_standard_precedence", "Workstyle weighted mix -> Site type fallback -> Portfolio default fallback"),
            }
        )
    outputs = outputs.merge(pd.DataFrame(planning_rows), on="site_id", how="left")
    outputs = _compute_site_metrics(outputs)
    outputs = _apply_consolidation_transfers(outputs, assumptions)
    outputs = _compute_site_metrics(outputs)
    outputs["scenario_name"] = scenario_name
    outputs["scenario_origin"] = "Live Scenario"
    outputs["scenario_id"] = f"LIVE-{scenario_name.upper().replace(' ', '-')}"
    summary = _scenario_summary_from_outputs(outputs)
    warnings = []
    if summary["expand_sites"] > 0:
        warnings.append(f"{summary['expand_sites']} site(s) show a material capacity deficit under the active scenario.")
    if summary["high_risk_sites"] > 0:
        warnings.append(f"{summary['high_risk_sites']} site(s) are rated High risk.")
    if not bool(outputs["occupancy_supported"].all()):
        warnings.append("Some sites are using people-demand attendance fallbacks because occupancy evidence is incomplete.")
    if not bool(outputs["space_inventory_supported"].all()):
        warnings.append("Some sites are using area-only planning assumptions because meeting/support inventory is incomplete.")
    if float(outputs["transfer_shortfall_ratio"].fillna(0.0).max()) > 0:
        warnings.append("Consolidation transfers could not be fully absorbed by preferred recipient sites for part of the portfolio.")

    return {
        "scenario_name": scenario_name,
        "origin": "Live Scenario",
        "assumptions": assumptions,
        "outputs": outputs.sort_values(["risk_rating", "scenario_score", "site_name"], ascending=[True, False, True]),
        "summary": summary,
        "warnings": warnings,
        "workbook_name": workbook_name,
        "workbook_hash": workbook_hash,
    }


def build_seed_snapshots(clean_sheets: dict[str, pd.DataFrame], workbook_name: str, workbook_hash: str) -> list[dict[str, Any]]:
    outputs = clean_sheets.get("scenario_outputs", pd.DataFrame()).copy()
    assumptions = clean_sheets.get("scenario_assumptions", pd.DataFrame()).copy()
    if outputs.empty:
        return []
    snapshots: list[dict[str, Any]] = []
    for scenario_name, scenario_outputs in outputs.groupby("scenario_name", dropna=False):
        scenario_outputs = _normalise_scenario_outputs(scenario_outputs)
        scenario_assumptions = assumptions[assumptions["scenario_name"] == scenario_name].copy()
        summary = _scenario_summary_from_outputs(scenario_outputs)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshots.append(
            {
                "snapshot_key": f"seed::{scenario_name}",
                "scenario_name": str(scenario_name),
                "origin": "Seed Scenario",
                "timestamp": timestamp,
                "calculation_timestamp": timestamp,
                "workbook_name": workbook_name,
                "workbook_hash": workbook_hash,
                "filters": {},
                "basis_scenario_name": str(scenario_name),
                "basis_origin": "Seed Scenario",
                "assumptions_used": scenario_assumptions.copy(),
                "assumption_count": int(len(scenario_assumptions.index)),
                "calculated_outputs": scenario_outputs.copy(),
                "output_site_count": int(len(scenario_outputs.index)),
                "summary": summary,
                "source": "scenario_outputs",
                "notes": "",
            }
        )
    return snapshots


def build_snapshot(
    scenario_bundle: dict[str, Any],
    *,
    active_filters: dict[str, Any],
    origin: str,
    scenario_name: str,
    basis_scenario_name: str | None = None,
    basis_origin: str | None = None,
    notes: str | None = None,
    calculation_timestamp: str | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    outputs = _normalise_scenario_outputs(scenario_bundle.get("outputs", pd.DataFrame()).copy())
    assumptions = scenario_bundle.get("assumptions", pd.DataFrame()).copy()
    snapshot_prefix = "saved" if origin == "Saved Scenario Snapshot" else "live"
    return {
        "snapshot_key": f"{snapshot_prefix}::{scenario_name}::{timestamp}",
        "scenario_name": scenario_name,
        "origin": origin,
        "timestamp": timestamp,
        "calculation_timestamp": calculation_timestamp or timestamp,
        "workbook_name": scenario_bundle.get("workbook_name"),
        "workbook_hash": scenario_bundle.get("workbook_hash"),
        "filters": active_filters,
        "basis_scenario_name": basis_scenario_name or scenario_bundle.get("scenario_name"),
        "basis_origin": basis_origin or scenario_bundle.get("origin"),
        "assumptions_used": assumptions,
        "assumption_count": int(len(assumptions.index)),
        "calculated_outputs": outputs,
        "output_site_count": int(len(outputs.index)),
        "summary": _scenario_summary_from_outputs(outputs),
        "source": "live_engine",
        "notes": str(notes or ""),
    }


def build_space_plan(clean_sheets: dict[str, pd.DataFrame], scenario_bundle: dict[str, Any], filters: dict[str, Any]) -> dict[str, pd.DataFrame]:
    baseline = build_portfolio_baseline(clean_sheets, filters)
    outputs = scenario_bundle.get("outputs", pd.DataFrame()).copy()
    hierarchy = baseline["floor_table"].copy()
    if hierarchy.empty or outputs.empty:
        return {"building_plan": pd.DataFrame(), "floor_plan": pd.DataFrame()}

    building_plan = baseline["building_table"].copy()
    if not building_plan.empty:
        building_plan["site_seat_total"] = building_plan.groupby("site_id")["seat_capacity_total"].transform("sum")
        building_plan["site_area_total"] = building_plan.groupby("site_id")["usable_area_sqm"].transform("sum")
        building_plan["site_building_count"] = building_plan.groupby("site_id")["building_id"].transform("nunique")
        building_plan["building_share"] = np.where(
            building_plan["site_seat_total"].fillna(0.0) > 0,
            building_plan["seat_capacity_total"].fillna(0.0) / building_plan["site_seat_total"].replace(0, np.nan),
            np.where(
                building_plan["site_area_total"].fillna(0.0) > 0,
                building_plan["usable_area_sqm"].fillna(0.0) / building_plan["site_area_total"].replace(0, np.nan),
                1 / building_plan["site_building_count"].replace(0, np.nan),
            ),
        )
        building_plan["allocation_basis"] = np.select(
            [
                building_plan["site_seat_total"].fillna(0.0) > 0,
                building_plan["site_area_total"].fillna(0.0) > 0,
            ],
            ["Seat capacity share", "Usable area share"],
            default="Equal building split",
        )
        building_plan["planning_confidence"] = np.select(
            [
                building_plan["site_seat_total"].fillna(0.0) > 0,
                building_plan["site_area_total"].fillna(0.0) > 0,
            ],
            ["High", "Medium"],
            default="Low",
        )
        building_plan = building_plan.merge(
            outputs[
                [
                    "site_id",
                    "scenario_name",
                    "required_seats",
                    "required_area_sqm",
                    "existing_seats",
                    "existing_usable_area_sqm",
                    "action_flag",
                    "risk_rating",
                    "action_reason",
                    "why_this_changed",
                ]
            ],
            on="site_id",
            how="left",
        )
        building_plan["target_seats"] = building_plan["required_seats"].fillna(0.0) * building_plan["building_share"].fillna(0.0)
        building_plan["target_area_sqm"] = (
            building_plan["required_area_sqm"].fillna(0.0) * building_plan["building_share"].fillna(0.0)
        )
        building_plan["seat_gap_target"] = building_plan["seat_capacity_total"].fillna(0.0) - building_plan["target_seats"].fillna(0.0)
        building_plan["implementation_flag"] = np.select(
            [
                building_plan["seat_gap_target"] < (
                    ACTION_THRESHOLD_DEFAULTS["expand_seat_deficit_ratio"] * building_plan["seat_capacity_total"].fillna(0.0)
                ),
                building_plan["seat_gap_target"] > (
                    ACTION_THRESHOLD_DEFAULTS["release_seat_surplus_ratio"] * building_plan["seat_capacity_total"].fillna(0.0)
                ),
            ],
            ["Expansion pressure", "Release candidate"],
            default="Rebalance / retain",
        )

    floor_plan = hierarchy.merge(
        outputs[
            [
                "site_id",
                "scenario_name",
                "required_seats",
                "required_area_sqm",
                "action_flag",
                "risk_rating",
                "action_reason",
                "why_this_changed",
            ]
        ],
        on="site_id",
        how="left",
    )
    floor_plan["site_floor_seat_total"] = floor_plan.groupby("site_id")["seat_capacity"].transform("sum")
    floor_plan["site_floor_area_total"] = floor_plan.groupby("site_id")["usable_area_sqm"].transform("sum")
    floor_plan["site_floor_count"] = floor_plan.groupby("site_id")["floor_id"].transform("nunique")
    floor_plan["floor_share"] = np.where(
        floor_plan["site_floor_seat_total"].fillna(0.0) > 0,
        floor_plan["seat_capacity"].fillna(0.0) / floor_plan["site_floor_seat_total"].replace(0, np.nan),
        np.where(
            floor_plan["site_floor_area_total"].fillna(0.0) > 0,
            floor_plan["usable_area_sqm"].fillna(0.0) / floor_plan["site_floor_area_total"].replace(0, np.nan),
            1 / floor_plan["site_floor_count"].replace(0, np.nan),
        ),
    )
    floor_plan["allocation_basis"] = np.select(
        [
            floor_plan["site_floor_seat_total"].fillna(0.0) > 0,
            floor_plan["site_floor_area_total"].fillna(0.0) > 0,
        ],
        ["Seat capacity share", "Usable area share"],
        default="Equal floor split",
    )
    floor_plan["planning_confidence"] = np.select(
        [
            floor_plan["site_floor_seat_total"].fillna(0.0) > 0,
            floor_plan["site_floor_area_total"].fillna(0.0) > 0,
        ],
        ["High", "Medium"],
        default="Low",
    )
    floor_plan["target_seats"] = floor_plan["required_seats"].fillna(0.0) * floor_plan["floor_share"].fillna(0.0)
    floor_plan["target_area_sqm"] = floor_plan["required_area_sqm"].fillna(0.0) * floor_plan["floor_share"].fillna(0.0)
    floor_plan["seat_gap_target"] = floor_plan["seat_capacity"].fillna(0.0) - floor_plan["target_seats"].fillna(0.0)
    floor_plan["intervention_flag"] = np.select(
        [
            floor_plan["seat_gap_target"] < (
                ACTION_THRESHOLD_DEFAULTS["expand_seat_deficit_ratio"] * floor_plan["seat_capacity"].fillna(0.0)
            ),
            floor_plan["seat_gap_target"] > (
                ACTION_THRESHOLD_DEFAULTS["release_seat_surplus_ratio"] * floor_plan["seat_capacity"].fillna(0.0)
            ),
        ],
        ["Re-stack / expand", "Release / repurpose"],
        default="Monitor / retain",
    )
    return {"building_plan": building_plan, "floor_plan": floor_plan}


def build_scenario_library(
    seed_snapshots: list[dict[str, Any]],
    saved_snapshots: list[dict[str, Any]],
    live_bundle: dict[str, Any] | None,
    manual_preferred_key: str | None = None,
    active_filters: dict[str, Any] | None = None,
    live_calculation_timestamp: str | None = None,
    basis_scenario_name: str | None = None,
    live_notes: str | None = None,
) -> list[dict[str, Any]]:
    library: list[dict[str, Any]] = []
    if live_bundle is not None:
        live_outputs = _normalise_scenario_outputs(live_bundle.get("outputs", pd.DataFrame()))
        live_assumptions = live_bundle.get("assumptions", pd.DataFrame())
        library.append(
            {
                "snapshot_key": "live::current",
                "scenario_name": live_bundle.get("scenario_name", "Live Scenario"),
                "origin": "Live Scenario",
                "summary": _scenario_summary_from_outputs(live_outputs),
                "assumptions_used": live_assumptions,
                "assumption_count": int(len(live_assumptions.index)),
                "calculated_outputs": live_outputs,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "calculation_timestamp": live_calculation_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "workbook_name": live_bundle.get("workbook_name"),
                "workbook_hash": live_bundle.get("workbook_hash"),
                "filters": active_filters or {},
                "basis_scenario_name": basis_scenario_name or live_bundle.get("scenario_name"),
                "basis_origin": "Seed Scenario" if basis_scenario_name else live_bundle.get("origin", "Live Scenario"),
                "output_site_count": int(len(live_outputs.index)),
                "source": "live_engine",
                "notes": str(live_notes or ""),
            }
        )
    library.extend(seed_snapshots)
    library.extend(saved_snapshots)
    for entry in library:
        outputs = _normalise_scenario_outputs(entry.get("calculated_outputs", pd.DataFrame()))
        entry["calculated_outputs"] = outputs
        entry["summary"] = _scenario_summary_from_outputs(outputs)
        entry["display_label"] = f"{entry['scenario_name']} [{entry['origin']}]"
        entry["manual_preferred"] = bool(manual_preferred_key and entry.get("snapshot_key") == manual_preferred_key)
    return library


def auto_recommended_scenario(library: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not library:
        return None
    ranked = sorted(
        library,
        key=lambda entry: (
            float(entry.get("summary", {}).get("scenario_score", 0.0) or 0.0),
            -float(entry.get("summary", {}).get("high_risk_sites", 0.0) or 0.0),
            -abs(float(entry.get("summary", {}).get("seat_gap", 0.0) or 0.0)),
            float(entry.get("summary", {}).get("standards_compliance_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ranked[0]


def resolve_preferred_scenario(
    library: list[dict[str, Any]],
    manual_preferred_key: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    auto = auto_recommended_scenario(library)
    if not manual_preferred_key or manual_preferred_key == PREFERRED_SCENARIO_AUTO_KEY:
        return auto, auto
    manual = next((entry for entry in library if entry.get("snapshot_key") == manual_preferred_key), auto)
    return auto, manual


def comparison_summary_table(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        summary = entry.get("summary", {})
        rows.append(
            {
                "Scenario": entry.get("scenario_name"),
                "Origin": entry.get("origin"),
                "Forecast Headcount": summary.get("forecast_headcount", 0.0),
                "Peak Attendance": summary.get("peak_attendance", 0.0),
                "Required Seats": summary.get("required_seats", 0.0),
                "Existing Seats": summary.get("existing_seats", 0.0),
                "Seat Gap": summary.get("seat_gap", 0.0),
                "Required Area SQM": summary.get("required_area_sqm", 0.0),
                "Area Gap SQM": summary.get("area_gap_sqm", 0.0),
                "Capacity Fit": summary.get("capacity_fit_score", 0.0),
                "Utilisation Fit": summary.get("utilisation_fit_score", 0.0),
                "Standards Compliance": summary.get("standards_compliance_score", 0.0),
                "Implementation Simplicity": summary.get("implementation_simplicity_score", 0.0),
                "Consolidation Efficiency": summary.get("consolidation_efficiency_score", 0.0),
                "Scenario Score": summary.get("scenario_score", 0.0),
                "High Risk Sites": summary.get("high_risk_sites", 0.0),
            }
        )
    return pd.DataFrame(rows)


def comparison_site_table(entries: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for entry in entries:
        outputs = _normalise_scenario_outputs(entry.get("calculated_outputs", pd.DataFrame()).copy())
        if outputs.empty:
            continue
        outputs = outputs[
            [
                "site_id",
                "site_name",
                "region",
                "scenario_name",
                "forecast_headcount",
                "required_seats",
                "existing_seats",
                "seat_gap",
                "required_area_sqm",
                "area_gap_sqm",
                "action_flag",
                "action_reason",
                "risk_rating",
                "key_risk",
                "why_this_changed",
                "standards_compliance_score",
                "capacity_fit_score",
                "utilisation_fit_score",
                "implementation_simplicity_score",
                "consolidation_efficiency_score",
                "scenario_score",
            ]
        ].copy()
        outputs["origin"] = entry.get("origin")
        frames.append(outputs)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decision_pack_narrative(preferred_entry: dict[str, Any] | None, baseline_entry: dict[str, Any] | None) -> str:
    if preferred_entry is None:
        return "No preferred scenario is currently selected."
    preferred_summary = preferred_entry.get("summary", {})
    preferred_name = str(preferred_entry.get("scenario_name", "Preferred Scenario"))
    if baseline_entry is None:
        return (
            f"{preferred_name} is presented as the lead scenario. It balances seat demand, area requirement, "
            "and standards alignment across the currently filtered portfolio."
        )
    baseline_summary = baseline_entry.get("summary", {})
    seat_delta = float(preferred_summary.get("seat_gap", 0.0) or 0.0) - float(baseline_summary.get("seat_gap", 0.0) or 0.0)
    area_delta = float(preferred_summary.get("area_gap_sqm", 0.0) or 0.0) - float(baseline_summary.get("area_gap_sqm", 0.0) or 0.0)
    return (
        f"{preferred_name} is the current lead option. Against the baseline view it changes the portfolio seat gap by "
        f"{seat_delta:,.0f} and the area gap by {area_delta:,.0f} sqm, while holding a transparent balance across "
        f"capacity fit ({preferred_summary.get('capacity_fit_score', 0.0):.1f}), utilisation fit "
        f"({preferred_summary.get('utilisation_fit_score', 0.0):.1f}), standards alignment "
        f"({preferred_summary.get('standards_compliance_score', 0.0):.1f}), and implementation practicality."
    )
