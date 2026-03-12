from __future__ import annotations

from typing import Any

import streamlit as st


EMBEDDED_MODE_KEY = "lens_embedded_mode"
EMBEDDED_ROUTE_KEY = "lens_embedded_route"


def init_lens_state(session_state: Any) -> None:
    session_state.setdefault(EMBEDDED_MODE_KEY, False)
    session_state.setdefault(EMBEDDED_ROUTE_KEY, "app")


def safe_set_page_config(page_title: str, page_icon: str = "L", layout: str = "wide") -> None:
    try:
        st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    except Exception:
        return


def is_embedded_mode(session_state: Any) -> bool:
    return bool(session_state.get(EMBEDDED_MODE_KEY, False))


def set_embedded_mode(session_state: Any, enabled: bool) -> None:
    session_state[EMBEDDED_MODE_KEY] = bool(enabled)


def get_embedded_route(session_state: Any) -> str:
    return str(session_state.get(EMBEDDED_ROUTE_KEY, "app"))


def set_embedded_route(session_state: Any, route: str) -> None:
    session_state[EMBEDDED_ROUTE_KEY] = str(route)


def navigate_to(session_state: Any, route: str, standalone_page_path: str) -> None:
    if is_embedded_mode(session_state):
        set_embedded_route(session_state, route)
        st.rerun()
    else:
        st.switch_page(standalone_page_path)


def render_nav_link(
    label: str,
    *,
    route: str,
    standalone_page_path: str,
    key: str,
    sidebar: bool = False,
) -> None:
    if sidebar:
        container = st.sidebar
        if is_embedded_mode(st.session_state):
            if container.button(label, key=key, use_container_width=True):
                set_embedded_route(st.session_state, route)
                st.rerun()
        else:
            container.page_link(standalone_page_path, label=label)
        return

    if is_embedded_mode(st.session_state):
        if st.button(label, key=key):
            set_embedded_route(st.session_state, route)
            st.rerun()
    else:
        st.page_link(standalone_page_path, label=label)

