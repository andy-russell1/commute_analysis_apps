from __future__ import annotations

import streamlit as st

from . import state
from .pages.assumptions_manager import render_page as render_assumptions_manager
from .pages.data_upload_validation import render_page as render_data_upload_validation
from .pages.decision_pack import render_page as render_decision_pack
from .pages.exports_audit import render_page as render_exports_audit
from .pages.home import render_page as render_home
from .pages.occupancy_utilisation import render_page as render_occupancy_utilisation
from .pages.portfolio_baseline import render_page as render_portfolio_baseline
from .pages.scenario_builder import render_page as render_scenario_builder
from .pages.scenario_comparison import render_page as render_scenario_comparison
from .pages.space_planning_outputs import render_page as render_space_planning_outputs


ROUTES = {
    "home": render_home,
    "data_upload_validation": render_data_upload_validation,
    "portfolio_baseline": render_portfolio_baseline,
    "occupancy_utilisation": render_occupancy_utilisation,
    "assumptions_manager": render_assumptions_manager,
    "scenario_builder": render_scenario_builder,
    "scenario_comparison": render_scenario_comparison,
    "space_planning_outputs": render_space_planning_outputs,
    "decision_pack": render_decision_pack,
    "exports_audit": render_exports_audit,
}


def run_emea_space_occupancy() -> None:
    state.init_emea_state(st.session_state)
    state.set_embedded_mode(st.session_state, enabled=True)
    route = state.get_route(st.session_state)
    ROUTES.get(route, ROUTES["home"])()

