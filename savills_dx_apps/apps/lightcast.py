from __future__ import annotations

import io
import re
import zipfile
from typing import List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium
import branca.colormap as cm

from core.downloads import df_to_csv_bytes
from core.lightcast import (
    build_master_from_files,
    detect_lightcast_like,
    is_lightcast_filename,
    read_any_table_bytes,
)
from core.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from core.paths import DATA_DIR


POSTING_INTENSITY_COLORS = [
    "#fff5f7",
    "#fee0e6",
    "#fcbfd2",
    "#f99bbf",
    "#f768a1",
    "#dd3497",
    "#ae017e",
    "#7a0177",
    "#49006a",
]


@st.cache_data(show_spinner=False)
def load_lad_geojson(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def _extract_files(upload: UploadPayload) -> List[Tuple[str, bytes]]:
    if upload.ext == "zip":
        files: List[Tuple[str, bytes]] = []
        with zipfile.ZipFile(io.BytesIO(upload.bytes_data)) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                clean = name.replace("\\", "/")
                if "/__MACOSX" in clean or clean.startswith("__MACOSX"):
                    continue
                base = clean.split("/")[-1]
                if base.lower().endswith((".csv", ".xls", ".xlsx")):
                    files.append((base, zf.read(name)))
        return files
    return [(upload.name, upload.bytes_data)]


def _normalize_lad_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _find_skill_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == "skill or qualification":
            return c
    for c in df.columns:
        name = str(c).strip().lower()
        if "skill" in name or "qualification" in name:
            return c
    return None


def _metric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        name = str(c).lower()
        if "posting intensity" in name:
            cols.append(c)
            continue
        if "unique postings" in name or "total postings" in name:
            cols.append(c)
            continue
        if name.endswith("postings") or "postings" in name:
            cols.append(c)
    return cols


def _monthly_unique_postings_cols(df: pd.DataFrame) -> List[str]:
    pattern = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+Unique Postings$", re.IGNORECASE)
    return [c for c in df.columns if pattern.match(str(c))]


def _numeric_metric_columns(df: pd.DataFrame, cols: List[str], sample_size: int = 1000) -> List[str]:
    if not cols:
        return []
    sample = df[cols].head(sample_size).copy()
    numeric_cols = []
    for c in cols:
        series = pd.to_numeric(sample[c], errors="coerce")
        if series.notna().mean() >= 0.25:
            numeric_cols.append(c)
    return numeric_cols


def _build_step_colormap(values: pd.Series) -> cm.StepColormap:
    clean = values.dropna().astype(float)
    if clean.empty:
        return cm.StepColormap(colors=POSTING_INTENSITY_COLORS, vmin=0, vmax=1)
    quantiles = clean.quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]).tolist()
    uniq = []
    for q in quantiles:
        if not uniq or q > uniq[-1]:
            uniq.append(q)
    if len(uniq) < 2:
        uniq = [clean.min(), clean.max() + 1e-9]
    return cm.StepColormap(colors=POSTING_INTENSITY_COLORS, vmin=uniq[0], vmax=uniq[-1], index=uniq)


class LightcastPlugin(AppPlugin):
    metadata = AppMetadata(
        id="lightcast",
        name="Lightcast",
        description="Upload Lightcast exports to build a master table and LAD breakdown.",
        accepted_upload_types=["zip", "csv", "xls", "xlsx"],
        upload_label="Upload Lightcast ZIP or multiple CSV/XLS/XLSX files",
        upload_help="Upload a single ZIP or select multiple CSV/XLS/XLSX files.",
    )

    def validate(self, upload: UploadPayload) -> None:
        if upload.ext not in {"zip", "csv", "xls", "xlsx"}:
            raise ValueError("Lightcast expects ZIP, CSV, XLS, or XLSX files.")
        files = _extract_files(upload)
        if not files:
            raise ValueError("No CSV/XLS/XLSX files found in the upload.")

        name_matches = any(is_lightcast_filename(name) for name, _ in files)
        if not name_matches:
            looks_like = False
            for sample_name, sample_bytes in files[:3]:
                df = read_any_table_bytes(sample_name, sample_bytes)
                if detect_lightcast_like(df):
                    looks_like = True
                    break
            if not looks_like:
                raise ValueError(
                    "Upload does not look like a Lightcast export. Expected Job_Posting_Table files or columns like Posting Intensity/Unique Postings."
                )

    def build(self, upload: UploadPayload, log) -> AppArtifacts:
        files = _extract_files(upload)
        if not files:
            raise ValueError("No CSV/XLS/XLSX files found in the upload.")

        log("Found {0} file(s)".format(len(files)))
        log("Building Lightcast master table")
        master = build_master_from_files(files)

        if "lad_name" not in master.columns:
            if "lad" in master.columns:
                master = master.rename(columns={"lad": "lad_name"})
            elif "lower district authority" in master.columns:
                master = master.rename(columns={"lower district authority": "lad_name"})

        return {
            "master": master,
            "file_count": len(files),
        }

    def render(self, artifacts: AppArtifacts) -> None:
        master_full = artifacts["master"]

        st.title("Lightcast Data Analyst search - Breakdown by Local Authority Districts")

        if "lad_name" not in master_full.columns:
            st.error("Lightcast master is missing 'lad_name', 'lad', or 'lower district authority'.")
            return

        master = master_full
        skill_col = _find_skill_column(master_full)
        if skill_col:
            skill_values = master_full[skill_col].dropna().astype(str).str.strip()
            skill_values = skill_values[skill_values != ""]
            skill_options = ["All"] + sorted(skill_values.unique())
            selected_skill = st.sidebar.selectbox("Skill or Qualification", skill_options, index=0)
            if selected_skill != "All":
                skill_series = master_full[skill_col].astype(str).str.strip()
                master = master_full[skill_series == selected_skill]
                if master.empty:
                    st.warning("No rows found for the selected Skill or Qualification.")
                    return

        monthly_cols = _monthly_unique_postings_cols(master)
        use_row_count = False
        metric_cols = _metric_columns(master)
        metric_cols = _numeric_metric_columns(master, metric_cols)

        total_unique_option = "Total unique postings (Jan 2024-Dec 2025)"
        metric_options: List[str] = []
        if monthly_cols:
            metric_options.append(total_unique_option)
        metric_options.extend(metric_cols)

        if not metric_options:
            st.warning("No posting metric columns found; using row counts per LAD as total postings.")
            use_row_count = True
            metric_col = "_total_postings"
            agg_method = "sum"
        else:
            metric_choice = st.sidebar.selectbox("Posting Metric Column", metric_options)
            agg_method = "sum"
            if monthly_cols and metric_choice == total_unique_option:
                metric_col = "total_unique_postings"
            else:
                metric_col = metric_choice

        lad_path = str(DATA_DIR / "geo" / "lad_uk_2024.geojson")
        lad_gdf = load_lad_geojson(lad_path)

        lad_name_cols = [c for c in lad_gdf.columns if c.lower().endswith("nm") or "name" in c.lower()]
        if not lad_name_cols:
            st.error("No LAD name column found in the geojson.")
            return

        default_lad_col = "LAD24NM" if "LAD24NM" in lad_name_cols else lad_name_cols[0]
        lad_name_col = default_lad_col

        if lad_gdf.crs is None or str(lad_gdf.crs).lower() != "epsg:4326":
            lad_gdf = lad_gdf.to_crs("EPSG:4326")

        if monthly_cols and metric_col == "total_unique_postings":
            metric_df = master[["lad_name"] + monthly_cols].copy()
            for c in monthly_cols:
                metric_df[c] = pd.to_numeric(metric_df[c], errors="coerce")
            metric_df[metric_col] = metric_df[monthly_cols].sum(axis=1, skipna=True)
            metric_df["_lad_name_norm"] = _normalize_lad_name(metric_df["lad_name"])
            agg = metric_df.groupby("_lad_name_norm")[metric_col].sum().reset_index()
        elif use_row_count:
            metric_df = master[["lad_name"]].copy()
            metric_df["_lad_name_norm"] = _normalize_lad_name(metric_df["lad_name"])
            agg = metric_df.groupby("_lad_name_norm").size().reset_index(name=metric_col)
        else:
            metric_df = master[["lad_name", metric_col]].copy()
            metric_df[metric_col] = pd.to_numeric(metric_df[metric_col], errors="coerce")
            metric_df["_lad_name_norm"] = _normalize_lad_name(metric_df["lad_name"])
            agg = metric_df.groupby("_lad_name_norm")[metric_col].agg(agg_method).reset_index()

        lad_gdf = lad_gdf.copy()
        lad_gdf["_lad_name_norm"] = _normalize_lad_name(lad_gdf[lad_name_col])
        lad_joined = lad_gdf.merge(agg, on="_lad_name_norm", how="left")

        data_lads = lad_joined[pd.notna(lad_joined[metric_col])].copy()
        map_target = data_lads if not data_lads.empty else lad_joined
        center = map_target.geometry.unary_union.centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=9)

        folium.TileLayer(
            "CartoDB positron",
            name="CartoDB Positron",
            attr="Map tiles by Carto, under CC BY 3.0 - Map data OpenStreetMap contributors",
        ).add_to(m)

        colormap = _build_step_colormap(lad_joined[metric_col])
        if monthly_cols and metric_col == "total_unique_postings":
            caption_label = "Total unique postings (Jan 2024-Dec 2025)"
        elif use_row_count:
            caption_label = "Total postings (row count)"
        else:
            caption_label = "{0} ({1})".format(metric_col, agg_method)
        colormap.caption = caption_label

        folium.GeoJson(
            lad_joined,
            style_function=lambda feature: {
                "fillColor": (
                    colormap(feature["properties"].get(metric_col))
                    if feature["properties"].get(metric_col) is not None
                    and pd.notna(feature["properties"].get(metric_col))
                    else "#cccccc"
                ),
                "color": "#444444",
                "weight": 0.6,
                "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[lad_name_col, metric_col],
                aliases=["LAD", caption_label],
                localize=True,
            ),
        ).add_to(m)

        colormap.add_to(m)
        if not map_target.empty:
            bounds = map_target.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        components.html(m.get_root().render(), height=720, scrolling=True)

        with st.expander("Downloads", expanded=False):
            st.download_button(
                "Download master CSV",
                data=df_to_csv_bytes(master_full),
                file_name="lightcast_master.csv",
                mime="text/csv",
            )

            if data_lads is not None and not data_lads.empty:
                st.download_button(
                    "Download LAD metrics",
                    data=df_to_csv_bytes(agg),
                    file_name="lightcast_lad_metrics.csv",
                    mime="text/csv",
                )

        st.markdown("---")
        st.caption(
            "Data source: Lightcast Skills tab export (Occupations: LOT, Include list; Skills & Qualifications: Has ANY; Job Titles: None selected)."
        )
        st.caption(
            "Files: uploaded Lightcast export(s); LAD boundaries: assets/data/geo/lad_uk_2024.geojson."
        )



PLUGIN = LightcastPlugin()
