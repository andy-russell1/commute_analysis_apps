from __future__ import annotations

import importlib
import sys

import streamlit as st

from apps.amenity_analysis.common import get_embedded_route, init_amenity_state, set_embedded_mode


_ROUTE_MODULES = {
    "app": "apps.amenity_analysis.app",
    "setup": "apps.amenity_analysis.pages.1_Setup",
    "controls": "apps.amenity_analysis.pages.2_Controls",
    "overview": "apps.amenity_analysis.pages.3_Overview",
    "drilldown": "apps.amenity_analysis.pages.4_Location_Drilldown",
}


def run_amenity_analysis() -> None:
    init_amenity_state(st.session_state)
    set_embedded_mode(st.session_state, enabled=True)
    route = get_embedded_route(st.session_state)
    module_name = _ROUTE_MODULES.get(route, _ROUTE_MODULES["app"])
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        return
    importlib.import_module(module_name)
