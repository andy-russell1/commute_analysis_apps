from __future__ import annotations

import streamlit as st

from shared.ui.cards import render_module_card
from shared.ui.page_header import render_page_header
from shell import registry, state


def _open_module(module_id: str, module_type: str) -> None:
    state.set_selected_module_id(module_id)
    state.set_step(2 if module_type == "wizard" else 1)
    st.rerun()


def render_home() -> None:
    st.markdown(
        """
        <div class="dx-home-surface">
        """,
        unsafe_allow_html=True,
    )

    render_page_header("Digital Experience App Hub", "Select a module to begin analysis.", logo_width=170)

    modules = registry.get_modules()
    section_order = registry.ordered_sections(modules)

    for section in section_order:
        st.markdown(f'<h3 class="dx-section-title">{section}</h3>', unsafe_allow_html=True)
        section_modules = [module for module in modules if module.metadata.section == section]
        for idx in range(0, len(section_modules), 2):
            row_modules = section_modules[idx : idx + 2]
            cols = st.columns(2)
            for col_idx, module in enumerate(row_modules):
                with cols[col_idx]:
                    if render_module_card(
                        module.metadata,
                        button_key=f"open_{module.metadata.id}",
                    ):
                        _open_module(module.metadata.id, module.metadata.module_type)

    st.markdown("</div>", unsafe_allow_html=True)
