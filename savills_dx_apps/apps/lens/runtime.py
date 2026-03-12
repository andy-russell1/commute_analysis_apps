from __future__ import annotations

import streamlit as st

from apps.lens.common import get_embedded_route, init_lens_state, set_embedded_mode
from apps.lens.export_view import render_page as render_export_page
from apps.lens.home_view import render_page as render_home_page
from apps.lens.matrix_view import render_page as render_matrix_page
from apps.lens.methodology_view import render_page as render_methodology_page
from apps.lens.results_view import render_page as render_results_page
from apps.lens.weights_view import render_page as render_weights_page


_ROUTES = {
    "app": render_home_page,
    "weights": render_weights_page,
    "results": render_results_page,
    "matrix": render_matrix_page,
    "export": render_export_page,
    "methodology": render_methodology_page,
}


def run_lens() -> None:
    init_lens_state(st.session_state)
    set_embedded_mode(st.session_state, enabled=True)
    route = get_embedded_route(st.session_state)
    _ROUTES.get(route, _ROUTES["app"])()
