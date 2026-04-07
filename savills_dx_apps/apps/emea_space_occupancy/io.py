from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

from .config import EXPECTED_SHEETS, SHEET_SPECS, SheetSpec


def _normalise_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_header_row(raw_df: pd.DataFrame, tokens: list[str]) -> int:
    lowered = [token.strip().lower() for token in tokens if token]
    for idx, row in raw_df.iterrows():
        row_text = " | ".join(_normalise_cell(value).lower() for value in row.tolist() if _normalise_cell(value))
        if all(token in row_text for token in lowered):
            return int(idx)
    raise ValueError(f"Could not find header row containing: {tokens}")


def _deduplicate_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        base = column if column else "unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1
        output.append(base if count == 0 else f"{base}_{count + 1}")
    return output


def _extract_table(raw_df: pd.DataFrame, spec: SheetSpec) -> pd.DataFrame:
    tokens = spec.header_tokens or spec.required_columns[: min(4, len(spec.required_columns))]
    header_row = find_header_row(raw_df, tokens)
    header_values = [_normalise_cell(value) for value in raw_df.iloc[header_row].tolist()]
    columns = _deduplicate_columns(header_values)
    table = raw_df.iloc[header_row + 1 :].copy()
    table.columns = columns
    table = table.dropna(how="all").copy()
    table = table.loc[:, [column for column in table.columns if _normalise_cell(column)]].copy()
    object_columns = table.select_dtypes(include=["object"]).columns
    for column in object_columns:
        table[column] = table[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        table[column] = table[column].replace({"": pd.NA})
    return table.reset_index(drop=True)


def load_workbook_from_bytes(file_bytes: bytes, workbook_name: str = "Workbook") -> dict[str, Any]:
    excel = pd.ExcelFile(BytesIO(file_bytes))
    available_sheets = [str(name) for name in excel.sheet_names]
    raw_tables: dict[str, pd.DataFrame] = {}
    tables: dict[str, pd.DataFrame] = {}
    parse_errors: dict[str, str] = {}

    for sheet_name in EXPECTED_SHEETS:
        if sheet_name not in available_sheets:
            continue
        raw_df = pd.read_excel(excel, sheet_name=sheet_name, header=None)
        raw_tables[sheet_name] = raw_df
        spec = SHEET_SPECS[sheet_name]
        try:
            tables[sheet_name] = _extract_table(raw_df, spec)
        except Exception as exc:
            parse_errors[sheet_name] = str(exc)

    return {
        "workbook_name": workbook_name,
        "available_sheets": available_sheets,
        "raw_tables": raw_tables,
        "tables": tables,
        "parse_errors": parse_errors,
    }

