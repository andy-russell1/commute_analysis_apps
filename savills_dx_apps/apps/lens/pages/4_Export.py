from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from apps.lens.common import safe_set_page_config

from apps.lens.core import model
from apps.lens.core.constants import MODE_ADVANCED

safe_set_page_config(page_title="Export", layout="wide")

model.render_page_header("Export Results")
context = model.render_sidebar()
model.ensure_context_ready(context)
model.ensure_data_validation(context, prefix="Input data has blocking errors.")
bundle = model.ensure_results_bundle(model.get_results_bundle(context))

parsed = context["parsed"]
criteria_df = parsed["criteria"]
macro_w, major_w, minor_w = model.get_active_weight_tables(criteria_df)

city_scores = bundle["city_scores"].sort_values("overall_score", ascending=False).copy()

st.subheader("Downloads")
st.caption("Use CSV for reporting and Excel for full audit and model traceability.")

csv_bytes = city_scores.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV (City-level outputs)",
    data=csv_bytes,
    file_name="lens_city_scores.csv",
    mime="text/csv",
    use_container_width=True,
)

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    macro_w.to_excel(writer, sheet_name="weights_macro", index=False)
    major_w.to_excel(writer, sheet_name="weights_major", index=False)
    minor_w.to_excel(writer, sheet_name="weights_minor", index=False)
    parsed["raw_data"].to_excel(writer, sheet_name="micro_raw_values", index=False)
    bundle["micro_scores"].to_excel(writer, sheet_name="micro_scores_long", index=False)
    bundle["major_scores"].to_excel(writer, sheet_name="major_scores", index=False)
    bundle["macro_scores"].to_excel(writer, sheet_name="macro_scores", index=False)
    city_scores.to_excel(writer, sheet_name="city_scores", index=False)

st.download_button(
    label="Download Excel (Weights + Raw + Scores)",
    data=excel_buffer.getvalue(),
    file_name="lens_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

if context["mode"] == MODE_ADVANCED:
    with st.expander("Preview export table (advanced)", expanded=False):
        st.dataframe(model.format_table_for_display(city_scores, decimals=3), use_container_width=True, hide_index=True)

