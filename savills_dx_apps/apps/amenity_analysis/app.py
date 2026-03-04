from __future__ import annotations

import streamlit as st

from apps.amenity_analysis.common import (
    ANALYSIS_MESSAGES_KEY,
    EMBEDDED_MODE_KEY,
    RESULTS_BY_RADIUS_KEY,
    SELECTED_METRICS_KEY,
    SITES_DF_KEY,
    init_amenity_state,
    navigate_to,
    safe_set_page_config,
    set_embedded_mode,
)
from core.session import APP_KEY, STEP_KEY


safe_set_page_config(page_title="Amenity Analysis", page_icon="📍", layout="wide")
init_amenity_state(st.session_state)

st.title("Amenity Analysis")
st.write("Amenity KPI using OSM amenities and optional local NaPTAN transport data.")

sites_df = st.session_state[SITES_DF_KEY]
results_by_radius = st.session_state[RESULTS_BY_RADIUS_KEY]

c1, c2, c3 = st.columns(3)
c1.metric("Valid offices loaded", int(len(sites_df)))
c2.metric("Radii analysed", int(len(results_by_radius)))
c3.metric("Selected metrics", int(len(st.session_state[SELECTED_METRICS_KEY])))

if st.session_state[ANALYSIS_MESSAGES_KEY]:
    st.warning("Some data sources returned warnings. Check Overview for details.")

if st.button("Go to Setup", type="primary"):
    navigate_to(st.session_state, route="setup", standalone_page_path="pages/1_Setup.py")

if st.session_state.get(EMBEDDED_MODE_KEY):
    if st.button("Back to App Hub"):
        set_embedded_mode(st.session_state, enabled=False)
        st.session_state[APP_KEY] = None
        st.session_state[STEP_KEY] = 1
        st.rerun()
