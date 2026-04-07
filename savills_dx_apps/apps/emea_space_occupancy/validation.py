from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CORE_MODEL_SHEETS, NULL_HEAVY_THRESHOLD, SHEET_SPECS


def _sheet_severity(sheet_name: str) -> str:
    return "Critical" if SHEET_SPECS[sheet_name].blocking else "Warning"


def _coerce_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    before = series.notna().sum()
    coerced = pd.to_numeric(series, errors="coerce")
    failures = int(max(before - coerced.notna().sum(), 0))
    return coerced, failures


def _coerce_date(series: pd.Series) -> tuple[pd.Series, int]:
    before = series.notna().sum()
    coerced = pd.to_datetime(series, errors="coerce")
    failures = int(max(before - coerced.notna().sum(), 0))
    return coerced, failures


def _coerce_share(series: pd.Series) -> tuple[pd.Series, int]:
    coerced, failures = _coerce_numeric(series)
    non_null = coerced.dropna()
    if not non_null.empty and float(non_null.max()) > 1.5 and float(non_null.max()) <= 100.0:
        coerced = coerced / 100.0
    return coerced, failures


def _normalise_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        out[column] = out[column].replace({"": pd.NA})
    return out


def _register_issue(
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
    *,
    severity: str,
    sheet: str,
    category: str,
    message: str,
    records: pd.DataFrame | None = None,
) -> str:
    issue_id = f"ISS-{len(issues) + 1:03d}"
    issues.append(
        {
            "issue_id": issue_id,
            "severity": severity,
            "sheet": sheet,
            "category": category,
            "message": message,
            "affected_rows": int(len(records)) if records is not None else 0,
        }
    )
    if records is not None and not records.empty:
        issue_records[issue_id] = records.reset_index(drop=True).head(250)
    return issue_id


def _missing_value_records(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    mask = df[columns].isna().any(axis=1)
    return df.loc[mask, columns].copy()


def _coerce_sheet(
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    spec = SHEET_SPECS[sheet_name]
    clean = _normalise_text_columns(df)
    coercion_count = 0

    for column in spec.numeric_columns:
        if column not in clean.columns or column in spec.share_columns:
            continue
        clean[column], failures = _coerce_numeric(clean[column])
        coercion_count += failures
        if failures:
            _register_issue(
                issues,
                issue_records,
                severity="Warning",
                sheet=sheet_name,
                category="type_coercion",
                message=f"Column '{column}' had {failures} numeric coercion failure(s).",
                records=df.loc[pd.to_numeric(df[column], errors="coerce").isna() & df[column].notna(), [column]].copy(),
            )

    for column in spec.date_columns:
        if column not in clean.columns:
            continue
        clean[column], failures = _coerce_date(clean[column])
        coercion_count += failures
        if failures:
            _register_issue(
                issues,
                issue_records,
                severity="Warning",
                sheet=sheet_name,
                category="type_coercion",
                message=f"Column '{column}' had {failures} date coercion failure(s).",
                records=df.loc[pd.to_datetime(df[column], errors="coerce").isna() & df[column].notna(), [column]].copy(),
            )

    for column in spec.share_columns:
        if column not in clean.columns:
            continue
        clean[column], failures = _coerce_share(clean[column])
        coercion_count += failures
        if failures:
            _register_issue(
                issues,
                issue_records,
                severity="Warning",
                sheet=sheet_name,
                category="type_coercion",
                message=f"Column '{column}' had {failures} percentage coercion failure(s).",
                records=df.loc[pd.to_numeric(df[column], errors="coerce").isna() & df[column].notna(), [column]].copy(),
            )

    return clean, coercion_count


def _check_duplicates(
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> int:
    spec = SHEET_SPECS[sheet_name]
    if not spec.key_columns or any(column not in df.columns for column in spec.key_columns):
        return 0
    duplicates = df[df.duplicated(spec.key_columns, keep=False)].copy()
    if duplicates.empty:
        return 0
    _register_issue(
        issues,
        issue_records,
        severity=_sheet_severity(sheet_name),
        sheet=sheet_name,
        category="duplicates",
        message=f"Duplicate key records found using {spec.key_columns}.",
        records=duplicates[spec.key_columns + [column for column in duplicates.columns if column not in spec.key_columns][:4]],
    )
    return int(len(duplicates))


def _check_non_negative(
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> None:
    spec = SHEET_SPECS[sheet_name]
    for column in spec.non_negative_columns:
        if column not in df.columns:
            continue
        invalid = df[df[column].fillna(0) < 0].copy()
        if invalid.empty:
            continue
        _register_issue(
            issues,
            issue_records,
            severity=_sheet_severity(sheet_name),
            sheet=sheet_name,
            category="negative_values",
            message=f"Column '{column}' contains negative values where they are not expected.",
            records=invalid[[column] + [col for col in ["site_id", "building_id", "floor_id", "scenario_id"] if col in invalid.columns]],
        )


def _check_share_ranges(
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> None:
    spec = SHEET_SPECS[sheet_name]
    for column in spec.share_columns:
        if column not in df.columns:
            continue
        upper_bound = float(spec.share_upper_bounds.get(column, 1.0))
        invalid = df[(df[column].notna()) & ((df[column] < 0) | (df[column] > upper_bound))].copy()
        if invalid.empty:
            continue
        _register_issue(
            issues,
            issue_records,
            severity="Warning",
            sheet=sheet_name,
            category="share_range",
            message=f"Column '{column}' contains values outside the supported range 0 to {upper_bound:.2f}.",
            records=invalid[[column] + [col for col in ["site_id", "building_id", "floor_id", "scenario_id"] if col in invalid.columns]],
        )


def _check_required_columns(
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> list[str]:
    spec = SHEET_SPECS[sheet_name]
    missing_columns = [column for column in spec.required_columns if column not in df.columns]
    if missing_columns:
        _register_issue(
            issues,
            issue_records,
            severity=_sheet_severity(sheet_name),
            sheet=sheet_name,
            category="missing_columns",
            message=f"Missing required column(s): {', '.join(missing_columns)}.",
        )
    return missing_columns


def _check_null_heavy(
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> None:
    required_columns = [column for column in SHEET_SPECS[sheet_name].required_columns if column in df.columns]
    if not required_columns or df.empty:
        return
    missing_ratio = df[required_columns].isna().mean()
    problem_columns = missing_ratio[missing_ratio >= NULL_HEAVY_THRESHOLD]
    if problem_columns.empty:
        return
    _register_issue(
        issues,
        issue_records,
        severity="Warning",
        sheet=sheet_name,
        category="null_heavy",
        message=(
            "High missing-value ratio detected in: "
            + ", ".join(f"{column} ({ratio:.0%})" for column, ratio in problem_columns.items())
        ),
        records=_missing_value_records(df, list(problem_columns.index)),
    )


def _relationship_check(
    *,
    from_sheet: str,
    key_column: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    right_key: str,
    severity: str,
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
    relationship_rows: list[dict[str, Any]],
) -> None:
    if key_column not in left_df.columns or right_key not in right_df.columns:
        return
    invalid = left_df[left_df[key_column].notna() & ~left_df[key_column].isin(right_df[right_key].dropna())].copy()
    relationship_rows.append(
        {
            "relationship": f"{from_sheet}.{key_column} -> {right_key}",
            "status": "Fail" if not invalid.empty else "Pass",
            "failed_rows": int(len(invalid)),
        }
    )
    if invalid.empty:
        return
    _register_issue(
        issues,
        issue_records,
        severity=severity,
        sheet=from_sheet,
        category="relationships",
        message=f"Invalid relationship: {key_column} values do not exist in the hierarchy reference.",
        records=invalid[[key_column] + [col for col in ["site_id", "building_id", "floor_id", "scenario_id"] if col in invalid.columns]],
    )


def _attribute_mismatch_check(
    *,
    from_sheet: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    key_column: str,
    attributes: list[str],
    issues: list[dict[str, Any]],
    issue_records: dict[str, pd.DataFrame],
) -> None:
    if key_column not in left_df.columns or key_column not in right_df.columns:
        return
    reference = right_df[[key_column] + [column for column in attributes if column in right_df.columns]].drop_duplicates(key_column)
    merged = left_df.merge(reference, on=key_column, how="left", suffixes=("", "_expected"))
    mismatch_masks: list[pd.Series] = []
    mismatch_columns: list[str] = []
    for column in attributes:
        expected_column = f"{column}_expected"
        if column not in merged.columns or expected_column not in merged.columns:
            continue
        mask = (
            merged[column].notna()
            & merged[expected_column].notna()
            & (merged[column].astype(str) != merged[expected_column].astype(str))
        )
        if mask.any():
            mismatch_masks.append(mask)
            mismatch_columns.append(column)
    if not mismatch_masks:
        return
    combined_mask = mismatch_masks[0]
    for mask in mismatch_masks[1:]:
        combined_mask = combined_mask | mask
    mismatch_records = merged.loc[combined_mask, [key_column] + mismatch_columns + [f"{column}_expected" for column in mismatch_columns]]
    _register_issue(
        issues,
        issue_records,
        severity="Warning",
        sheet=from_sheet,
        category="hierarchy_mismatch",
        message=f"Hierarchy attribute mismatches detected for {', '.join(mismatch_columns)}.",
        records=mismatch_records,
    )


def validate_workbook(bundle: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    issue_records: dict[str, pd.DataFrame] = {}
    clean_sheets: dict[str, pd.DataFrame] = {}
    sheet_summary_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []

    for sheet_name, spec in SHEET_SPECS.items():
        raw_df = bundle["tables"].get(sheet_name)
        if raw_df is None:
            severity = _sheet_severity(sheet_name)
            _register_issue(
                issues,
                issue_records,
                severity=severity,
                sheet=sheet_name,
                category="missing_sheet",
                message=f"Required workbook sheet '{sheet_name}' is missing." if spec.blocking else f"Optional seed/support sheet '{sheet_name}' is missing.",
            )
            sheet_summary_rows.append(
                {
                    "sheet_name": sheet_name,
                    "row_count": 0,
                    "column_count": 0,
                    "duplicate_key_count": 0,
                    "missing_value_count": 0,
                    "coercion_issue_count": 0,
                    "issue_count": 1,
                    "status": "Missing",
                }
            )
            continue

        if sheet_name in bundle.get("parse_errors", {}):
            _register_issue(
                issues,
                issue_records,
                severity=_sheet_severity(sheet_name),
                sheet=sheet_name,
                category="parse_error",
                message=f"Sheet could not be parsed: {bundle['parse_errors'][sheet_name]}",
            )
            sheet_summary_rows.append(
                {
                    "sheet_name": sheet_name,
                    "row_count": 0,
                    "column_count": 0,
                    "duplicate_key_count": 0,
                    "missing_value_count": 0,
                    "coercion_issue_count": 0,
                    "issue_count": 1,
                    "status": "Parse error",
                }
            )
            continue

        missing_columns = _check_required_columns(sheet_name, raw_df, issues, issue_records)
        clean_df, coercion_count = _coerce_sheet(sheet_name, raw_df, issues, issue_records)
        clean_sheets[sheet_name] = clean_df
        duplicate_count = _check_duplicates(sheet_name, clean_df, issues, issue_records)
        _check_non_negative(sheet_name, clean_df, issues, issue_records)
        _check_share_ranges(sheet_name, clean_df, issues, issue_records)
        _check_null_heavy(sheet_name, clean_df, issues, issue_records)

        sheet_issue_count = sum(1 for issue in issues if issue["sheet"] == sheet_name)
        status = "Ready"
        if missing_columns:
            status = "Review"
        if any(issue["sheet"] == sheet_name and issue["severity"] == "Critical" for issue in issues):
            status = "Blocked"
        elif sheet_issue_count:
            status = "Review"

        sheet_summary_rows.append(
            {
                "sheet_name": sheet_name,
                "row_count": int(len(clean_df)),
                "column_count": int(len(clean_df.columns)),
                "duplicate_key_count": int(duplicate_count),
                "missing_value_count": int(clean_df.isna().sum().sum()),
                "coercion_issue_count": int(coercion_count),
                "issue_count": int(sheet_issue_count),
                "status": status,
            }
        )

    hierarchy = clean_sheets.get("portfolio_hierarchy", pd.DataFrame())
    if not hierarchy.empty:
        _relationship_check(
            from_sheet="property_metrics",
            key_column="building_id",
            left_df=clean_sheets.get("property_metrics", pd.DataFrame()),
            right_df=hierarchy,
            right_key="building_id",
            severity="Critical",
            issues=issues,
            issue_records=issue_records,
            relationship_rows=relationship_rows,
        )
        _relationship_check(
            from_sheet="space_inventory",
            key_column="floor_id",
            left_df=clean_sheets.get("space_inventory", pd.DataFrame()),
            right_df=hierarchy,
            right_key="floor_id",
            severity="Critical",
            issues=issues,
            issue_records=issue_records,
            relationship_rows=relationship_rows,
        )
        _relationship_check(
            from_sheet="people_demand",
            key_column="site_id",
            left_df=clean_sheets.get("people_demand", pd.DataFrame()),
            right_df=hierarchy,
            right_key="site_id",
            severity="Critical",
            issues=issues,
            issue_records=issue_records,
            relationship_rows=relationship_rows,
        )
        _relationship_check(
            from_sheet="occupancy_utilisation",
            key_column="site_id",
            left_df=clean_sheets.get("occupancy_utilisation", pd.DataFrame()),
            right_df=hierarchy,
            right_key="site_id",
            severity="Critical",
            issues=issues,
            issue_records=issue_records,
            relationship_rows=relationship_rows,
        )
        if "scenario_outputs" in clean_sheets:
            _relationship_check(
                from_sheet="scenario_outputs",
                key_column="site_id",
                left_df=clean_sheets.get("scenario_outputs", pd.DataFrame()),
                right_df=hierarchy,
                right_key="site_id",
                severity="Warning",
                issues=issues,
                issue_records=issue_records,
                relationship_rows=relationship_rows,
            )

        _attribute_mismatch_check(
            from_sheet="property_metrics",
            left_df=clean_sheets.get("property_metrics", pd.DataFrame()),
            right_df=hierarchy,
            key_column="building_id",
            attributes=["site_id", "region", "country", "city", "site_name", "site_type", "building_name"],
            issues=issues,
            issue_records=issue_records,
        )
        _attribute_mismatch_check(
            from_sheet="space_inventory",
            left_df=clean_sheets.get("space_inventory", pd.DataFrame()),
            right_df=hierarchy,
            key_column="floor_id",
            attributes=["site_id", "region", "country", "city", "site_name", "building_id", "building_name", "floor_name"],
            issues=issues,
            issue_records=issue_records,
        )
        _attribute_mismatch_check(
            from_sheet="people_demand",
            left_df=clean_sheets.get("people_demand", pd.DataFrame()),
            right_df=hierarchy.drop_duplicates("site_id"),
            key_column="site_id",
            attributes=["region", "country", "city", "site_name", "site_type"],
            issues=issues,
            issue_records=issue_records,
        )
        _attribute_mismatch_check(
            from_sheet="occupancy_utilisation",
            left_df=clean_sheets.get("occupancy_utilisation", pd.DataFrame()),
            right_df=hierarchy.drop_duplicates("site_id"),
            key_column="site_id",
            attributes=["region", "country", "city", "site_name", "site_type"],
            issues=issues,
            issue_records=issue_records,
        )

    assumptions = clean_sheets.get("scenario_assumptions", pd.DataFrame())
    outputs = clean_sheets.get("scenario_outputs", pd.DataFrame())
    if not assumptions.empty and not outputs.empty and {"scenario_id"}.issubset(assumptions.columns) and {"scenario_id"}.issubset(outputs.columns):
        missing_in_outputs = assumptions[~assumptions["scenario_id"].isin(outputs["scenario_id"].dropna())]
        missing_in_assumptions = outputs[~outputs["scenario_id"].isin(assumptions["scenario_id"].dropna())]
        if not missing_in_outputs.empty:
            _register_issue(
                issues,
                issue_records,
                severity="Warning",
                sheet="scenario_outputs",
                category="scenario_consistency",
                message="Some scenario IDs exist in assumptions but not in seed outputs.",
                records=missing_in_outputs[["scenario_id", "scenario_name"]].drop_duplicates(),
            )
        if not missing_in_assumptions.empty:
            _register_issue(
                issues,
                issue_records,
                severity="Warning",
                sheet="scenario_outputs",
                category="scenario_consistency",
                message="Some scenario IDs exist in seed outputs but not in assumptions.",
                records=missing_in_assumptions[["scenario_id", "scenario_name"]].drop_duplicates(),
            )

    issues_df = pd.DataFrame(issues)
    sheet_summary = pd.DataFrame(sheet_summary_rows)
    relationship_summary = pd.DataFrame(relationship_rows)
    critical_count = int((issues_df["severity"] == "Critical").sum()) if not issues_df.empty else 0
    warning_count = int((issues_df["severity"] == "Warning").sum()) if not issues_df.empty else 0
    info_count = int((issues_df["severity"] == "Info").sum()) if not issues_df.empty else 0
    quality_score = max(0, 100 - (20 * critical_count) - (5 * warning_count) - info_count)
    status = "Blocked" if critical_count else "Ready with warnings" if warning_count else "Ready"

    return {
        "clean_sheets": clean_sheets,
        "issues": issues_df.sort_values(["severity", "sheet", "issue_id"]).reset_index(drop=True)
        if not issues_df.empty
        else pd.DataFrame(columns=["issue_id", "severity", "sheet", "category", "message", "affected_rows"]),
        "issue_records": issue_records,
        "sheet_summary": sheet_summary,
        "relationship_summary": relationship_summary,
        "quality_score": int(quality_score),
        "status": status,
        "blocking": critical_count > 0,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "available_core_sheets": [sheet for sheet in CORE_MODEL_SHEETS if sheet in clean_sheets],
    }
