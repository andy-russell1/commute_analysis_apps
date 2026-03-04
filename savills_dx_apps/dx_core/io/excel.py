from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd


EXCEL_EXTENSIONS = {"xls", "xlsx"}
CSV_EXTENSIONS = {"csv"}


def _normalise_ext(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].strip().lower()


def is_excel_file(filename: str) -> bool:
    return _normalise_ext(filename) in EXCEL_EXTENSIONS


def list_excel_sheets(file_bytes: bytes) -> list[str]:
    excel = pd.ExcelFile(BytesIO(file_bytes))
    return [str(name) for name in excel.sheet_names]


def safe_read_excel(file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    target = 0 if not sheet_name else sheet_name
    try:
        return pd.read_excel(BytesIO(file_bytes), sheet_name=target)
    except ValueError as exc:
        raise ValueError(f"Unable to read selected worksheet: {exc}") from exc


def safe_read_upload(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    ext = _normalise_ext(filename)
    if ext in CSV_EXTENSIONS:
        return pd.read_csv(BytesIO(file_bytes))
    if ext in EXCEL_EXTENSIONS:
        return safe_read_excel(file_bytes=file_bytes, sheet_name=sheet_name)
    raise ValueError("Unsupported upload type. Please upload CSV or XLSX.")


def read_table_from_path(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    ext = _normalise_ext(path.name)
    if ext in CSV_EXTENSIONS or ext == "txt":
        return pd.read_csv(path)
    if ext in EXCEL_EXTENSIONS:
        target = 0 if not sheet_name else sheet_name
        return pd.read_excel(path, sheet_name=target)
    raise ValueError(f"Unsupported file extension for {path}")
