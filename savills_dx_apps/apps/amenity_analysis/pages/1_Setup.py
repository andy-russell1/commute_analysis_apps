from __future__ import annotations

import streamlit as st

from apps.amenity_analysis.common import (
    PRIMARY_RADIUS_KEY,
    SELECTED_RADII_KEY,
    SITES_DF_KEY,
    clear_analysis_results,
    init_amenity_state,
    navigate_to,
    safe_set_page_config,
)
from dx_core.io.excel import is_excel_file, list_excel_sheets, safe_read_upload
from dx_core.io.validation import validate_and_clean_sites


safe_set_page_config(page_title="Amenity Analysis - Setup", page_icon="📥", layout="wide")
init_amenity_state(st.session_state)

st.title("Setup")
st.markdown("Required headers: `officeID`, `address`, `office - Latitude`, `office - Longitude`")

radius_options = [200, 400, 800, 1000, 1500]
existing_radii = st.session_state.get(SELECTED_RADII_KEY, [1000])
default_radius = next((value for value in existing_radii if value in radius_options), 1000)
selected_radius = int(
    st.selectbox(
        "Analysis radius (metres)",
        options=radius_options,
        index=radius_options.index(default_radius),
        help="Single radius used for KPI scoring and maps.",
    )
)
st.session_state[SELECTED_RADII_KEY] = [selected_radius]
st.session_state[PRIMARY_RADIUS_KEY] = selected_radius

upload = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
sheet_name = None

if upload is not None:
    file_bytes = upload.getvalue()
    if is_excel_file(upload.name):
        try:
            sheet_names = list_excel_sheets(file_bytes)
            sheet_name = st.selectbox("Worksheet", options=sheet_names)
        except Exception as exc:
            st.error(f"Could not inspect workbook: {exc}")

    if st.button("Load and validate", type="primary"):
        try:
            raw_df = safe_read_upload(file_bytes=file_bytes, filename=upload.name, sheet_name=sheet_name)
        except Exception as exc:
            st.error(f"Failed to read upload: {exc}")
        else:
            validation = validate_and_clean_sites(raw_df)
            st.session_state["amenity_raw_preview_df"] = raw_df.head(200)
            st.session_state["amenity_issues_df"] = validation.issues_df
            st.session_state["amenity_missing_cols"] = validation.missing_columns
            st.session_state[SITES_DF_KEY] = validation.cleaned_df
            clear_analysis_results(st.session_state)

preview_df = st.session_state.get("amenity_raw_preview_df")
if preview_df is not None:
    st.markdown("### Preview")
    st.dataframe(preview_df, use_container_width=True)

missing_cols = st.session_state.get("amenity_missing_cols", [])
issues_df = st.session_state.get("amenity_issues_df")
cleaned_df = st.session_state[SITES_DF_KEY]

st.markdown("### Validation")
if missing_cols:
    st.error("Missing required columns: " + ", ".join(missing_cols))
else:
    st.success("All required columns are present.")

if issues_df is not None and not issues_df.empty:
    st.warning("Invalid rows were removed from analysis input.")
    st.dataframe(issues_df, use_container_width=True)

st.markdown(f"### Cleaned Sites ({len(cleaned_df)})")
if not cleaned_df.empty:
    st.dataframe(cleaned_df, use_container_width=True)

if not cleaned_df.empty and st.button("Continue to Controls"):
    navigate_to(st.session_state, route="controls", standalone_page_path="pages/2_Controls.py")
