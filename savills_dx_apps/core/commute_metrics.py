from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: {0}. Available: {1}".format(missing, list(df.columns)))


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _office_labels(offices: Sequence[dict]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for o in offices:
        oid = str(o.get("officeID", ""))
        full = str(o.get("address", "")).strip()
        short = (full.split(",")[0].strip() if full else oid) or oid
        out[oid] = (short, full or oid)
    return out


def select_office_method(
    df_valid: pd.DataFrame,
    office_id: str,
    method: str,
    *,
    best_label: str = "Best",
    emp_id_col: str = "employeeID",
    office_id_col: str = "officeID",
    method_col: str = "method",
    tt_col: str = "travel_time_min",
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    _ensure_cols(df_valid, [emp_id_col, office_id_col, method_col, tt_col])
    d = df_valid[df_valid[office_id_col].astype(str) == str(office_id)].copy()
    d[tt_col] = _as_float(d[tt_col])

    if method == best_label:
        if d.empty:
            return d.iloc[0:0].copy()
        idx = d.groupby(d[emp_id_col].astype(str))[tt_col].idxmin()
        out = d.loc[idx].copy()
        out["best_method"] = out[method_col].astype(str)
        out[method_col] = best_label
    else:
        out = d[d[method_col].astype(str) == str(method)].copy()

    if min_time is not None:
        out = out[_as_float(out[tt_col]) >= float(min_time)]
    if max_time is not None:
        out = out[_as_float(out[tt_col]) <= float(max_time)]

    return out.sort_values(tt_col, ascending=True, na_position="last")


def office_stats(
    df_valid: pd.DataFrame,
    offices: Sequence[dict],
    method: str,
    *,
    best_label: str = "Best",
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    labels = _office_labels(offices)
    rows = []

    for o in offices:
        oid = str(o.get("officeID"))
        d = select_office_method(
            df_valid,
            oid,
            method,
            best_label=best_label,
            min_time=min_time,
            max_time=max_time,
        )
        vals = _as_float(d["travel_time_min"]).dropna().to_numpy(dtype="float64")
        n = int(vals.size)

        short, full = labels.get(oid, (oid, oid))
        rows.append(
            {
                "Office": short,
                "Employee Count": n,
                "Median (mins)": float(np.quantile(vals, 0.5)) if n else np.nan,
                "90th Percentile (mins)": float(np.quantile(vals, 0.9)) if n else np.nan,
                "Mean (mins)": float(np.mean(vals)) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def threshold_bands(
    df_valid: pd.DataFrame,
    offices: Sequence[dict],
    method: str,
    *,
    best_label: str = "Best",
    thresholds: Tuple[float, float, float] = (30.0, 45.0, 60.0),
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    t1, t2, t3 = thresholds
    labels = _office_labels(offices)

    rows = []
    for o in offices:
        oid = str(o.get("officeID"))
        d = select_office_method(
            df_valid,
            oid,
            method,
            best_label=best_label,
            min_time=min_time,
            max_time=max_time,
        )
        vals = _as_float(d["travel_time_min"]).dropna().to_numpy(dtype="float64")
        n = int(vals.size)

        le30_n = int(np.sum(vals <= t1)) if n else 0
        b30_45_n = int(np.sum((vals > t1) & (vals <= t2))) if n else 0
        b45_60_n = int(np.sum((vals > t2) & (vals <= t3))) if n else 0
        gt60_n = int(np.sum(vals > t3)) if n else 0

        def pct(k: int) -> float:
            return (k / n) * 100.0 if n else 0.0

        short, full = labels.get(oid, (oid, oid))
        rows.append(
            {
                "Office": short,
                "Total Employees": n,
                "<=30 min": pct(le30_n),
                "30-45 min": pct(b30_45_n),
                "45-60 min": pct(b45_60_n),
                ">60 min": pct(gt60_n),
                "<=30 min (count)": le30_n,
                "30-45 min (count)": b30_45_n,
                "45-60 min (count)": b45_60_n,
                ">60 min (count)": gt60_n,
            }
        )

    return pd.DataFrame(rows)


def explore_table(
    df_valid: pd.DataFrame,
    office_id: str,
    method: str,
    *,
    best_label: str = "Best",
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    d = select_office_method(
        df_valid,
        office_id,
        method,
        best_label=best_label,
        min_time=min_time,
        max_time=max_time,
    ).copy()

    for c in ["postcode", "city", "country", "lat", "lon"]:
        if c not in d.columns:
            d[c] = ""

    cols = ["employeeID", "postcode", "city", "country", "travel_time_min", "lat", "lon"]
    if method == best_label:
        if "best_method" not in d.columns:
            d["best_method"] = ""
        cols = ["employeeID", "postcode", "city", "country", "best_method", "travel_time_min", "lat", "lon"]

    out = d[cols].copy()
    out["travel_time_min"] = _as_float(out["travel_time_min"]).round(1)

    rename_map = {
        "employeeID": "Employee ID",
        "postcode": "Postcode",
        "city": "City",
        "country": "Country",
        "best_method": "Best Method",
        "travel_time_min": "Travel Time (mins)",
    }
    out = out.rename(columns=rename_map)
    return out


def wide_table(
    df_valid: pd.DataFrame,
    office_id: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    _ensure_cols(df_valid, ["employeeID", "officeID", "method", "travel_time_min"])
    d = df_valid[df_valid["officeID"].astype(str) == str(office_id)].copy()
    d["travel_time_min"] = _as_float(d["travel_time_min"]).round(1)

    for c in ["postcode", "city", "country"]:
        if c not in d.columns:
            d[c] = ""

    meta = d[["employeeID", "postcode", "city", "country"]].drop_duplicates(subset=["employeeID"]).copy()
    piv = (
        d[d["method"].astype(str).isin([str(m) for m in methods])]
        .pivot_table(index="employeeID", columns="method", values="travel_time_min", aggfunc="min")
        .reset_index()
    )

    out = meta.merge(piv, on="employeeID", how="left")
    for m in methods:
        if m not in out.columns:
            out[m] = np.nan

    cols = ["employeeID", "postcode", "city", "country"] + list(methods)
    out = out[cols].sort_values("employeeID")

    rename_map = {
        "employeeID": "Employee ID",
        "postcode": "Postcode",
        "city": "City",
        "country": "Country",
    }
    out = out.rename(columns=rename_map)
    return out


def wide_table_all_offices(
    df_valid: pd.DataFrame,
    offices: Sequence[dict],
    methods: Sequence[str],
) -> pd.DataFrame:
    _ensure_cols(df_valid, ["employeeID", "officeID", "method", "travel_time_min"])
    d = df_valid.copy()
    d["travel_time_min"] = _as_float(d["travel_time_min"]).round(1)

    for c in ["postcode", "city", "country"]:
        if c not in d.columns:
            d[c] = ""

    office_lookup = {
        str(o.get("officeID", "")): str(o.get("address", "")).split(",")[0].strip() for o in offices
    }
    d["office_short"] = d["officeID"].astype(str).map(office_lookup)

    meta = d[["employeeID", "postcode", "city", "country"]].drop_duplicates(subset=["employeeID"]).copy()

    piv = (
        d[d["method"].astype(str).isin([str(m) for m in methods])]
        .pivot_table(
            index="employeeID",
            columns=["office_short", "method"],
            values="travel_time_min",
            aggfunc="min",
        )
        .reset_index()
    )

    if isinstance(piv.columns, pd.MultiIndex):
        piv.columns = [
            "_".join([str(c).strip() for c in col if c != ""]).strip() if col[0] != "employeeID" else "employeeID"
            for col in piv.columns.values
        ]

    out = meta.merge(piv, on="employeeID", how="left")

    rename_map = {
        "employeeID": "Employee ID",
        "postcode": "Postcode",
        "city": "City",
        "country": "Country",
    }
    out = out.rename(columns=rename_map)
    out = out.sort_values("Employee ID")
    return out
