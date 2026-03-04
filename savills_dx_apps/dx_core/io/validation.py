from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_SITE_COLUMNS = ["officeID", "address", "office - Latitude", "office - Longitude"]


@dataclass
class ValidationResult:
    cleaned_df: pd.DataFrame
    issues_df: pd.DataFrame
    missing_columns: list[str]

    @property
    def has_errors(self) -> bool:
        return bool(self.missing_columns) or not self.issues_df.empty


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_SITE_COLUMNS) -> list[str]:
    return [col for col in required_columns if col not in df.columns]


def validate_and_clean_sites(df: pd.DataFrame) -> ValidationResult:
    missing_columns = validate_required_columns(df)
    if missing_columns:
        issues = [{"row_number": "-", "field": "columns", "issue": f"Missing column: {col}"} for col in missing_columns]
        return ValidationResult(
            cleaned_df=pd.DataFrame(columns=["officeID", "address", "lat", "lon"]),
            issues_df=pd.DataFrame(issues),
            missing_columns=missing_columns,
        )

    working = df.copy()
    working["officeID"] = working["officeID"].astype(str).str.strip()
    working["address"] = working["address"].astype(str).str.strip()

    lat_series = pd.to_numeric(working["office - Latitude"], errors="coerce")
    lon_series = pd.to_numeric(working["office - Longitude"], errors="coerce")

    issues: list[dict[str, str | int]] = []

    invalid_office = working["officeID"].eq("") | working["officeID"].eq("nan")
    invalid_address = working["address"].eq("") | working["address"].eq("nan")
    invalid_lat_numeric = lat_series.isna()
    invalid_lon_numeric = lon_series.isna()
    invalid_lat_range = (~invalid_lat_numeric) & ~lat_series.between(-90, 90)
    invalid_lon_range = (~invalid_lon_numeric) & ~lon_series.between(-180, 180)

    for idx in working.index[invalid_office]:
        issues.append({"row_number": int(idx) + 2, "field": "officeID", "issue": "officeID is empty"})
    for idx in working.index[invalid_address]:
        issues.append({"row_number": int(idx) + 2, "field": "address", "issue": "address is empty"})
    for idx in working.index[invalid_lat_numeric]:
        issues.append({"row_number": int(idx) + 2, "field": "office - Latitude", "issue": "Latitude is not numeric"})
    for idx in working.index[invalid_lon_numeric]:
        issues.append({"row_number": int(idx) + 2, "field": "office - Longitude", "issue": "Longitude is not numeric"})
    for idx in working.index[invalid_lat_range]:
        issues.append({"row_number": int(idx) + 2, "field": "office - Latitude", "issue": "Latitude out of range [-90, 90]"})
    for idx in working.index[invalid_lon_range]:
        issues.append({"row_number": int(idx) + 2, "field": "office - Longitude", "issue": "Longitude out of range [-180, 180]"})

    invalid_rows = invalid_office | invalid_address | invalid_lat_numeric | invalid_lon_numeric | invalid_lat_range | invalid_lon_range

    cleaned_df = pd.DataFrame(
        {
            "officeID": working.loc[~invalid_rows, "officeID"],
            "address": working.loc[~invalid_rows, "address"],
            "lat": lat_series.loc[~invalid_rows],
            "lon": lon_series.loc[~invalid_rows],
        }
    ).reset_index(drop=True)

    issues_df = pd.DataFrame(issues)
    if not issues_df.empty:
        issues_df = issues_df.sort_values(["row_number", "field"]).reset_index(drop=True)
    else:
        issues_df = pd.DataFrame(columns=["row_number", "field", "issue"])

    return ValidationResult(cleaned_df=cleaned_df, issues_df=issues_df, missing_columns=missing_columns)
