from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def _filter_summary(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "All portfolio filters active."
    parts: list[str] = []
    for key, label in [
        ("region", "Region"),
        ("country", "Country"),
        ("city", "City"),
        ("site_name", "Site"),
        ("site_type", "Site type"),
        ("building_name", "Building"),
        ("business_unit", "Business unit"),
    ]:
        values = filters.get(key) or []
        if values:
            parts.append(f"{label}: {', '.join(map(str, values[:2]))}{' +' if len(values) > 2 else ''}")
    month_range = filters.get("month_range")
    if month_range:
        start, end = month_range
        if start is not None and end is not None:
            parts.append(f"Period: {pd.Timestamp(start):%b %Y} to {pd.Timestamp(end):%b %Y}")
    return " | ".join(parts) if parts else "All portfolio filters active."


def dataframe_to_excel_bytes(sheet_map: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            safe_name = str(sheet_name)[:31]
            export_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            export_df.to_excel(writer, sheet_name=safe_name, index=False)
    return buffer.getvalue()


def validation_log_bytes(validation_result: dict[str, Any]) -> bytes:
    return dataframe_to_excel_bytes(
        {
            "sheet_summary": validation_result.get("sheet_summary", pd.DataFrame()),
            "issues": validation_result.get("issues", pd.DataFrame()),
            "relationships": validation_result.get("relationship_summary", pd.DataFrame()),
        }
    )


def snapshot_metadata_table(snapshots: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for snapshot in snapshots:
        assumptions_used = snapshot.get("assumptions_used", pd.DataFrame())
        outputs = snapshot.get("calculated_outputs", pd.DataFrame())
        rows.append(
            {
                "snapshot_key": snapshot.get("snapshot_key"),
                "scenario_name": snapshot.get("scenario_name"),
                "origin": snapshot.get("origin"),
                "basis_scenario_name": snapshot.get("basis_scenario_name"),
                "basis_origin": snapshot.get("basis_origin"),
                "timestamp": snapshot.get("timestamp"),
                "calculation_timestamp": snapshot.get("calculation_timestamp", snapshot.get("timestamp")),
                "workbook_name": snapshot.get("workbook_name"),
                "workbook_hash": snapshot.get("workbook_hash"),
                "filters": _filter_summary(snapshot.get("filters", {})),
                "assumption_count": snapshot.get(
                    "assumption_count",
                    int(len(assumptions_used.index)) if isinstance(assumptions_used, pd.DataFrame) else 0,
                ),
                "output_site_count": snapshot.get(
                    "output_site_count",
                    int(len(outputs.index)) if isinstance(outputs, pd.DataFrame) else 0,
                ),
                "notes_captured": "Yes" if str(snapshot.get("notes", "")).strip() else "No",
            }
        )
    return pd.DataFrame(rows)


def scenario_outputs_export_table(entries: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for entry in entries:
        outputs = entry.get("calculated_outputs", pd.DataFrame()).copy()
        if outputs.empty:
            continue
        outputs.insert(0, "calculation_timestamp", entry.get("calculation_timestamp", entry.get("timestamp")))
        outputs.insert(0, "basis_origin", entry.get("basis_origin"))
        outputs.insert(0, "basis_scenario_name", entry.get("basis_scenario_name"))
        outputs.insert(0, "origin", entry.get("origin"))
        frames.append(outputs)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assumptions_export_table(entries: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for entry in entries:
        assumptions = entry.get("assumptions_used", pd.DataFrame()).copy()
        if assumptions.empty:
            continue
        assumptions.insert(0, "calculation_timestamp", entry.get("calculation_timestamp", entry.get("timestamp")))
        assumptions.insert(0, "basis_origin", entry.get("basis_origin"))
        assumptions.insert(0, "basis_scenario_name", entry.get("basis_scenario_name"))
        assumptions.insert(0, "origin", entry.get("origin"))
        assumptions.insert(0, "scenario_name_export", entry.get("scenario_name"))
        frames.append(assumptions)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_excel_package(
    *,
    validation_result: dict[str, Any],
    baseline_site_table: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    comparison_site: pd.DataFrame,
    decision_pack_summary: pd.DataFrame,
    snapshots: list[dict[str, Any]],
) -> bytes:
    return dataframe_to_excel_bytes(
        {
            "validation_summary": validation_result.get("sheet_summary", pd.DataFrame()),
            "validation_issues": validation_result.get("issues", pd.DataFrame()),
            "baseline_sites": baseline_site_table,
            "comparison_summary": comparison_summary,
            "comparison_sites": comparison_site,
            "decision_pack": decision_pack_summary,
            "snapshot_metadata": snapshot_metadata_table(snapshots),
            "snapshot_assumptions": assumptions_export_table(snapshots),
            "snapshot_outputs": scenario_outputs_export_table(snapshots),
        }
    )
