from __future__ import annotations

import streamlit as st

from shell.registry_models import ModuleMetadata


def render_module_card(metadata: ModuleMetadata, *, button_key: str) -> bool:
    with st.container(border=True):
        st.subheader(metadata.name)
        st.write(metadata.description)
        if metadata.tags:
            st.caption("Tags: " + ", ".join(metadata.tags))
        return st.button(metadata.button_label, key=button_key, type="primary")

