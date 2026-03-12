from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CRITERIA_COLUMN_NAMES, CRITERIA_SHEET_NAME, DATA_BASE_COLUMNS, DATA_SHEET_NAME


def make_criterion_id(macro: str, major: str, micro: str) -> str:
    return f"{macro}||{major}||{micro}"


def _clean_text(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text == "" else text


def _first_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.iloc[0])


def _find_header_row(raw_df: pd.DataFrame, required_tokens: list[str]) -> int:
    lowered_tokens = [token.lower() for token in required_tokens]
    for idx, row in raw_df.iterrows():
        text = " | ".join(str(v).strip().lower() for v in row.tolist() if not pd.isna(v))
        if all(token in text for token in lowered_tokens):
            return int(idx)
    raise ValueError(f"Could not find header row containing: {required_tokens}")


def parse_criteria_sheet(criteria_raw: pd.DataFrame) -> pd.DataFrame:
    if criteria_raw.shape[1] < 6:
        raise ValueError("Criteria Sheet must contain at least 6 columns (A:F).")

    header_row = _find_header_row(criteria_raw.iloc[:, :6], ["Macro Criteria", "Major Criteria", "Minor Criteria"])
    body = criteria_raw.iloc[header_row + 1 :, :6].copy()
    body.columns = CRITERIA_COLUMN_NAMES

    for col in ["macro", "major", "micro"]:
        body[col] = body[col].map(_clean_text)

    body = body.dropna(how="all").copy()
    body["macro"] = body["macro"].ffill()
    body["major"] = body["major"].ffill()

    macro_weight_map = (
        body.groupby("macro", dropna=True)["macro_weight_template"].apply(_first_numeric).to_dict()
    )
    major_weight_map = (
        body.groupby(["macro", "major"], dropna=True)["major_weight_template"].apply(_first_numeric).to_dict()
    )

    leaves = body[body["micro"].notna()].copy()
    leaves["macro_weight_template"] = leaves["macro"].map(macro_weight_map)
    leaves["major_weight_template"] = [
        major_weight_map.get((macro, major), np.nan)
        for macro, major in zip(leaves["macro"], leaves["major"], strict=False)
    ]
    leaves["minor_weight_template"] = pd.to_numeric(leaves["minor_weight_template"], errors="coerce")
    leaves["criterion_id"] = [
        make_criterion_id(macro, major, micro)
        for macro, major, micro in zip(leaves["macro"], leaves["major"], leaves["micro"], strict=False)
    ]

    return leaves[
        [
            "criterion_id",
            "macro",
            "major",
            "micro",
            "macro_weight_template",
            "major_weight_template",
            "minor_weight_template",
        ]
    ].reset_index(drop=True)


def parse_data_sheet(data_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if data_raw.shape[1] < 5:
        raise ValueError("Data Sheet must contain at least 5 columns (A:E).")

    header_row = _find_header_row(data_raw.iloc[:, :6], ["Macro", "Major", "Micro", "Source"])
    header_values = data_raw.iloc[header_row].tolist()
    columns: list[str] = []
    for idx, value in enumerate(header_values):
        cleaned = _clean_text(value)
        if pd.isna(cleaned):
            columns.append(f"column_{idx}")
        else:
            columns.append(str(cleaned))

    body = data_raw.iloc[header_row + 1 :].copy()
    body.columns = columns
    if len(columns) < 5:
        raise ValueError("Data Sheet requires source plus at least one city column.")

    rename_map = {columns[0]: "macro", columns[1]: "major", columns[2]: "micro", columns[3]: "source"}
    body = body.rename(columns=rename_map)
    city_columns = [col for col in body.columns if col not in DATA_BASE_COLUMNS]

    for col in DATA_BASE_COLUMNS:
        body[col] = body[col].map(_clean_text)

    body["macro_ffill"] = body["macro"].ffill()
    body["major_ffill"] = body["major"].ffill()
    body["micro_ffill"] = body["micro"].ffill()

    rank_mask = (
        body["source"].fillna("").str.contains("rank", case=False, regex=False)
        & body["micro"].isna()
    )

    rank_reference = body[rank_mask].copy()
    rank_reference["macro"] = rank_reference["macro_ffill"]
    rank_reference["major"] = rank_reference["major_ffill"]
    rank_reference["micro"] = rank_reference["micro_ffill"]

    raw_rows = body[~rank_mask & body["micro"].notna()].copy()
    raw_rows["macro"] = raw_rows["macro_ffill"]
    raw_rows["major"] = raw_rows["major_ffill"]

    for city in city_columns:
        raw_rows[city] = pd.to_numeric(raw_rows[city], errors="coerce")
        if city in rank_reference.columns:
            rank_reference[city] = pd.to_numeric(rank_reference[city], errors="coerce")

    raw_rows["criterion_id"] = [
        make_criterion_id(macro, major, micro)
        for macro, major, micro in zip(raw_rows["macro"], raw_rows["major"], raw_rows["micro"], strict=False)
    ]
    rank_reference["criterion_id"] = [
        make_criterion_id(macro, major, micro)
        for macro, major, micro in zip(
            rank_reference["macro"], rank_reference["major"], rank_reference["micro"], strict=False
        )
    ]

    raw_rows = raw_rows[
        ["criterion_id", "macro", "major", "micro", "source", *city_columns]
    ].reset_index(drop=True)
    rank_reference = rank_reference[
        ["criterion_id", "macro", "major", "micro", "source", *city_columns]
    ].reset_index(drop=True)

    return raw_rows, rank_reference, city_columns


def load_workbook_from_bytes(file_bytes: bytes) -> dict[str, Any]:
    workbook = BytesIO(file_bytes)
    excel = pd.ExcelFile(workbook)

    missing_sheets = [s for s in [CRITERIA_SHEET_NAME, DATA_SHEET_NAME] if s not in excel.sheet_names]
    if missing_sheets:
        raise ValueError(f"Missing required sheet(s): {', '.join(missing_sheets)}")

    criteria_raw = pd.read_excel(excel, sheet_name=CRITERIA_SHEET_NAME, header=None)
    data_raw = pd.read_excel(excel, sheet_name=DATA_SHEET_NAME, header=None)

    criteria = parse_criteria_sheet(criteria_raw)
    raw_data, rank_reference, city_columns = parse_data_sheet(data_raw)

    return {
        "criteria": criteria,
        "raw_data": raw_data,
        "rank_reference": rank_reference,
        "city_columns": city_columns,
    }


def find_default_workbook(cwd: Path) -> Path | None:
    lens_root = Path(__file__).resolve().parents[1]
    candidates = [
        cwd / "LENS.xlsx",
        cwd / "LENS_dummy_filled.xlsx",
        cwd / "LENS dummy data.xlsx",
        cwd / "app" / "assets" / "sample_template.xlsx",
        lens_root / "assets" / "LENS.xlsx",
        lens_root / "assets" / "LENS dummy data.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
