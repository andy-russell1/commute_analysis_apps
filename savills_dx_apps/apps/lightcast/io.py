from __future__ import annotations

import io
import re

import pandas as pd


FILENAME_RE = re.compile(
    r"^Job_Posting_Table_(?P<table_no>\d+)_"
    r"(?P<table_name>.*?)_in_(?P<geo_name>.*?)_"
    r"(?P<file_hash>[0-9a-fA-F]{2,})(?:\s*\(\d+\))?\."
    r"(?P<ext>csv|xls|xlsx)$",
    re.IGNORECASE,
)
FILENAME_GEO_FALLBACK_RE = re.compile(r"_in_(?P<geo_name>.+?)(?:_[0-9a-fA-F]{2,})?$", re.IGNORECASE)
SHEET_HEADER_HINTS = ["posting intensity", "unique postings", "occupation", "industry", "soc", "naics", "median", "latest 30"]


def parse_filename(file_name: str) -> dict[str, str]:
    match = FILENAME_RE.match(file_name)
    if not match:
        stem = file_name.rsplit(".", 1)[0]
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        geo_name = ""
        fallback = FILENAME_GEO_FALLBACK_RE.search(stem)
        if fallback:
            geo_name = fallback.group("geo_name")
        return {"table_no": "", "table_name": "", "geo_name": geo_name, "file_hash": "", "ext": ext, "raw_stem": stem}
    d = match.groupdict()
    d["raw_stem"] = file_name.rsplit(".", 1)[0]
    return d


def geo_name_to_lad(geo_name: str) -> str:
    if not geo_name:
        return ""
    return geo_name.replace("_", " ").strip()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]


def _score_sheet(df: pd.DataFrame | None) -> int:
    if df is None or df.empty:
        return -1
    df = clean_columns(df)
    if df.shape[1] == 0:
        return -1
    non_empty_rows = int(df.dropna(how="all").shape[0])
    if non_empty_rows == 0:
        return -1
    score = min(non_empty_rows, 200) + min(df.shape[1], 25) * 2
    cols_l = [str(c).lower() for c in df.columns]
    for hint in SHEET_HEADER_HINTS:
        if any(hint in c for c in cols_l):
            score += 50
    return score


def _find_header_row(raw: pd.DataFrame) -> int | None:
    if raw.empty:
        return None
    for idx in range(min(30, len(raw))):
        row = raw.iloc[idx].astype(str).str.strip().str.lower()
        non_empty = row[row != ""]
        if non_empty.shape[0] < 3:
            continue
        if any(hint in " ".join(row.tolist()) for hint in SHEET_HEADER_HINTS):
            return idx
    return None


def _select_best_sheet(sheet_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    best_df = None
    best_score = -1
    for df in sheet_map.values():
        score = _score_sheet(df)
        if score > best_score:
            best_score = score
            best_df = df
    if best_df is not None:
        return clean_columns(best_df)
    return clean_columns(next(iter(sheet_map.values())))


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
        return _select_best_sheet(pd.read_excel(io.BytesIO(bytes_data), sheet_name=None))
    raise ValueError("Unsupported file type: {name}".format(name=file_name))


def is_lightcast_filename(file_name: str) -> bool:
    name = file_name.lower()
    return "job_posting_table" in name or "lightcast" in name


def detect_lightcast_like(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols_l = [str(c).lower() for c in df.columns]
    return any(any(hint in c for c in cols_l) for hint in SHEET_HEADER_HINTS)


def build_master_from_files(files: list[tuple[str, bytes]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
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
