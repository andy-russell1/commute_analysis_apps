from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


FILENAME_RE = re.compile(
    r"^Job_Posting_Table_(?P<table_no>\d+)_"
    r"(?P<table_name>.*?)_in_(?P<geo_name>.*?)_"
    r"(?P<file_hash>[0-9a-fA-F]{2,})(?:\s*\(\d+\))?\."
    r"(?P<ext>csv|xls|xlsx)$",
    re.IGNORECASE,
)

FILENAME_GEO_FALLBACK_RE = re.compile(
    r"_in_(?P<geo_name>.+?)(?:_[0-9a-fA-F]{2,})?$",
    re.IGNORECASE,
)

SHEET_HEADER_HINTS = [
    "posting intensity",
    "unique postings",
    "occupation",
    "industry",
    "soc",
    "naics",
    "median",
    "latest 30",
]


def parse_filename(file_name: str) -> Dict[str, str]:
    m = FILENAME_RE.match(file_name)
    if not m:
        stem = file_name.rsplit(".", 1)[0]
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        geo_name = ""
        fallback = FILENAME_GEO_FALLBACK_RE.search(stem)
        if fallback:
            geo_name = fallback.group("geo_name")
        return {
            "table_no": "",
            "table_name": "",
            "geo_name": geo_name,
            "file_hash": "",
            "ext": ext,
            "raw_stem": stem,
        }
    d = m.groupdict()
    d["raw_stem"] = file_name.rsplit(".", 1)[0]
    return d


def geo_name_to_lad(geo_name: str) -> str:
    if not geo_name:
        return ""
    return geo_name.replace("_", " ").strip()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]
    return df


def _score_sheet(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return -1
    df = clean_columns(df)
    if df.shape[1] == 0:
        return -1
    non_empty_rows = int(df.dropna(how="all").shape[0])
    if non_empty_rows == 0:
        return -1
    score = 0
    score += min(non_empty_rows, 200)
    score += min(df.shape[1], 25) * 2
    cols_l = [str(c).lower() for c in df.columns]
    for hint in SHEET_HEADER_HINTS:
        if any(hint in c for c in cols_l):
            score += 50
    return score


def _find_header_row(raw: pd.DataFrame) -> Optional[int]:
    if raw.empty:
        return None
    for i in range(min(30, len(raw))):
        row = raw.iloc[i].astype(str).str.strip()
        row_l = row.str.lower()
        non_empty = row_l[row_l != ""]
        if non_empty.shape[0] < 3:
            continue
        if any(hint in " ".join(row_l.tolist()) for hint in SHEET_HEADER_HINTS):
            return i
    return None


def _select_best_sheet(sheet_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    best_df = None
    best_score = -1
    for _, df in sheet_map.items():
        score = _score_sheet(df)
        if score > best_score:
            best_score = score
            best_df = df
    if best_df is not None:
        return clean_columns(best_df)
    first = next(iter(sheet_map.values()))
    return clean_columns(first)


def _read_csv_with_header_detection(bytes_data: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(bytes_data), header=None)
    except UnicodeDecodeError:
        raw = pd.read_csv(io.BytesIO(bytes_data), header=None, encoding="latin-1")
    header_row = _find_header_row(raw)
    if header_row is None:
        header_row = 0
    try:
        df = pd.read_csv(io.BytesIO(bytes_data), header=header_row)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(bytes_data), header=header_row, encoding="latin-1")
    return clean_columns(df)


def read_any_table_bytes(file_name: str, bytes_data: bytes) -> pd.DataFrame:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext == "csv":
        return _read_csv_with_header_detection(bytes_data)
    if ext in ("xls", "xlsx"):
        sheet_map = pd.read_excel(io.BytesIO(bytes_data), sheet_name=None)
        df = _select_best_sheet(sheet_map)
        if _score_sheet(df) >= 0:
            return df
        return df
    raise ValueError("Unsupported file type: {name}".format(name=file_name))


def is_lightcast_filename(file_name: str) -> bool:
    name = file_name.lower()
    if "job_posting_table" in name:
        return True
    if "lightcast" in name:
        return True
    return False


def detect_lightcast_like(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols_l = [str(c).lower() for c in df.columns]
    for hint in SHEET_HEADER_HINTS:
        if any(hint in c for c in cols_l):
            return True
    return False


def build_master_from_files(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name, data in files:
        df = read_any_table_bytes(name, data)
        if df is None or df.empty:
            continue
        meta = parse_filename(name)
        geo_name = meta.get("geo_name", "")
        df = df.copy()
        df.insert(0, "lower district authority", geo_name_to_lad(geo_name))
        df.insert(0, "source_file", name)
        frames.append(df)
    if not frames:
        raise ValueError("No readable Lightcast tables found in the upload.")
    return pd.concat(frames, ignore_index=True, sort=False)
