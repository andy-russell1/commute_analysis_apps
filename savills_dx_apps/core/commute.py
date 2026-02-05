from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cols:
    emp_id: str
    metric: str
    value: str
    method: str
    emp_lat: str
    emp_lon: str
    office_id: str
    office_addr: str
    office_lat: str
    office_lon: str
    postcode: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    query_dt: Optional[str] = None


def _norm(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("postal code", "postcode")
    s = s.replace("post code", "postcode")
    return s


def _find_col(df: pd.DataFrame, *candidates: str, required: bool = True) -> Optional[str]:
    cols_norm = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in cols_norm:
            return cols_norm[key]

    norm_keys = list(cols_norm.keys())
    for cand in candidates:
        key = _norm(cand)
        close = get_close_matches(key, norm_keys, n=1, cutoff=0.86)
        if close:
            return cols_norm[close[0]]

    if required:
        raise KeyError(
            "Could not find required column. Tried: {cands}\nAvailable columns:\n- {cols}".format(
                cands=", ".join(candidates),
                cols="\n- ".join(df.columns.astype(str).tolist()),
            )
        )
    return None


def match_columns(df: pd.DataFrame) -> Cols:
    emp_id = _find_col(df, "EmployeeID", "Employee ID", "employee id")
    metric = _find_col(df, "Metric", "metric")
    value = _find_col(df, "Value", "value")
    method = _find_col(
        df,
        "Query Transport Method",
        "Transport Method",
        "transport method",
        "method",
        "transport",
    )

    emp_lat = _find_col(df, "Employee - Lat", "Employee - Latitude", "employee latitude", "emp lat")
    emp_lon = _find_col(df, "Employee - Long", "Employee - Longitude", "employee longitude", "emp lon", "emp long")

    office_id = _find_col(df, "OfficeID", "Office ID", "office id", "office_id")
    office_addr = _find_col(df, "Office - Address", "address", "office address")
    office_lat = _find_col(df, "Office - Lat", "Office - Latitude", "office latitude", "office lat")
    office_lon = _find_col(df, "Office - Long", "Office - Longitude", "office longitude", "office lon", "office long")

    postcode = _find_col(df, "Postcode", "Postal Code", "postcode", "Employee - Postcode", required=False)
    city = _find_col(df, "City", "city", required=False)
    country = _find_col(df, "Country", "country", required=False)
    query_dt = _find_col(df, "Query Search Datetime", "Query Seach Datetime", "query datetime", required=False)

    return Cols(
        emp_id=emp_id,
        metric=metric,
        value=value,
        method=method,
        emp_lat=emp_lat,
        emp_lon=emp_lon,
        office_id=office_id,
        office_addr=office_addr,
        office_lat=office_lat,
        office_lon=office_lon,
        postcode=postcode,
        city=city,
        country=country,
        query_dt=query_dt,
    )


def filter_travel_time_valid(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    d = df.copy()

    d[cols.metric] = d[cols.metric].astype(str).str.strip().str.lower()
    d[cols.value] = pd.to_numeric(d[cols.value], errors="coerce")

    d["Value2"] = d[cols.value].astype("float64")
    d.loc[d[cols.metric] == "distance", "Value2"] = d.loc[d[cols.metric] == "distance", cols.value] / 1000.0
    d.loc[d[cols.metric] == "travel_time", "Value2"] = d.loc[d[cols.metric] == "travel_time", cols.value] / 60.0

    d_tt = d[d[cols.metric] == "travel_time"].copy()

    for c in [cols.emp_lat, cols.emp_lon, cols.office_lat, cols.office_lon, "Value2"]:
        d_tt[c] = pd.to_numeric(d_tt[c], errors="coerce")

    if cols.query_dt and cols.query_dt in d_tt.columns:
        d_tt[cols.query_dt] = pd.to_datetime(d_tt[cols.query_dt], errors="coerce")
        d_tt = d_tt.sort_values(cols.query_dt, ascending=False)

    required = [
        cols.emp_id,
        cols.method,
        cols.emp_lat,
        cols.emp_lon,
        cols.office_id,
        cols.office_lat,
        cols.office_lon,
        "Value2",
    ]
    d_valid = d_tt.dropna(subset=required).copy()

    d_valid[cols.emp_id] = d_valid[cols.emp_id].astype(str)
    d_valid[cols.method] = d_valid[cols.method].astype(str)
    d_valid[cols.office_id] = d_valid[cols.office_id].astype(str)

    d_valid = d_valid.drop_duplicates(subset=[cols.office_id, cols.emp_id, cols.method], keep="first")

    rename_map = {
        cols.emp_id: "employeeID",
        cols.method: "method",
        cols.office_id: "officeID",
        cols.emp_lat: "lat",
        cols.emp_lon: "lon",
        cols.office_lat: "officeLat",
        cols.office_lon: "officeLon",
        cols.office_addr: "officeAddress",
    }
    if cols.postcode:
        rename_map[cols.postcode] = "postcode"
    if cols.city:
        rename_map[cols.city] = "city"
    if cols.country:
        rename_map[cols.country] = "country"

    out = d_valid.rename(columns=rename_map).copy()
    out["travel_time_min"] = pd.to_numeric(out["Value2"], errors="coerce")

    keep = [
        "employeeID",
        "postcode" if "postcode" in out.columns else None,
        "city" if "city" in out.columns else None,
        "country" if "country" in out.columns else None,
        "method",
        "travel_time_min",
        "lat",
        "lon",
        "officeID",
        "officeAddress",
        "officeLat",
        "officeLon",
    ]
    keep = [c for c in keep if c is not None]
    return out[keep].copy()


def compute_office_summary(df_valid: pd.DataFrame) -> pd.DataFrame:
    d = df_valid.copy()
    d["travel_time_min"] = pd.to_numeric(d["travel_time_min"], errors="coerce")
    d["officeShort"] = d["officeAddress"].astype(str).str.split(",").str[0].str.strip()
    d["officeShort"] = d["officeShort"].mask(
        d["officeShort"].eq(""),
        d["officeID"].astype(str),
    )

    grouped = d.groupby(["officeID", "officeShort"], dropna=False)
    summary = grouped["travel_time_min"].agg(["count", "median", "mean"]).reset_index()
    summary = summary.rename(
        columns={
            "officeShort": "Office",
            "count": "Employee Count",
            "median": "Median (mins)",
            "mean": "Mean (mins)",
        }
    )
    summary["Median (mins)"] = summary["Median (mins)"].round(1)
    summary["Mean (mins)"] = summary["Mean (mins)"].round(1)
    return summary.sort_values("Median (mins)", ascending=True, na_position="last")


def compute_band_table(df_valid: pd.DataFrame, thresholds: Tuple[float, float, float] = (30.0, 45.0, 60.0)) -> pd.DataFrame:
    t1, t2, t3 = thresholds
    d = df_valid.copy()
    d["travel_time_min"] = pd.to_numeric(d["travel_time_min"], errors="coerce")
    d["officeShort"] = d["officeAddress"].astype(str).str.split(",").str[0].str.strip()
    d["officeShort"] = d["officeShort"].mask(
        d["officeShort"].eq(""),
        d["officeID"].astype(str),
    )

    rows = []
    for office, group in d.groupby("officeShort"):
        vals = group["travel_time_min"].dropna().to_numpy(dtype="float64")
        n = int(vals.size)
        le30 = int(np.sum(vals <= t1)) if n else 0
        b30_45 = int(np.sum((vals > t1) & (vals <= t2))) if n else 0
        b45_60 = int(np.sum((vals > t2) & (vals <= t3))) if n else 0
        gt60 = int(np.sum(vals > t3)) if n else 0
        rows.append(
            {
                "Office": office,
                "Total Employees": n,
                "<=30 min": le30,
                "30-45 min": b30_45,
                "45-60 min": b45_60,
                ">60 min": gt60,
            }
        )
    return pd.DataFrame(rows).sort_values("Office")


def compute_overall_metrics(df_valid: pd.DataFrame) -> Dict[str, float]:
    d = df_valid.copy()
    d["travel_time_min"] = pd.to_numeric(d["travel_time_min"], errors="coerce")
    vals = d["travel_time_min"].dropna().to_numpy(dtype="float64")
    if vals.size == 0:
        return {"median": float("nan"), "mean": float("nan")}
    return {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
    }
