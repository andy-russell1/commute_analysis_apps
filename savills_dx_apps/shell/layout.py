from __future__ import annotations

import streamlit as st

from shell import branding


def configure_page() -> None:
    st.set_page_config(page_title="Savills DX Apps", layout="wide")


def render_sidebar_branding() -> None:
    return None


def render_sidebar_actions(selected_name: str | None) -> dict[str, bool]:
    actions = {"go_home": False, "restart": False}
    with st.sidebar:
        if selected_name:
            st.divider()
            st.caption("Current module")
            st.write(f"**{selected_name}**")
            if st.button("Back to App Hub", key="dx_back_to_home", use_container_width=True):
                actions["go_home"] = True

        st.divider()
        if st.button("Restart Session", key="dx_restart", use_container_width=True):
            actions["restart"] = True

    return actions


def render_page_branding() -> None:
    logo_path = branding.combined_logo_path()
    if logo_path is None:
        logo_path = branding.savills_logo_path()
    if logo_path is None:
        return

    spacer_col, logo_col = st.columns([6.0, 1.2])
    with spacer_col:
        st.write("")
    with logo_col:
        st.image(str(logo_path), use_container_width=True)
