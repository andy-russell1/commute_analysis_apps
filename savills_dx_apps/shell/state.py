from __future__ import annotations

import streamlit as st

from shared.runtime.session import APP_KEY, STEP_KEY, clear_all_states


def init_shell_state() -> None:
    st.session_state.setdefault(APP_KEY, None)
    st.session_state.setdefault(STEP_KEY, 1)


def get_selected_module_id() -> str | None:
    return st.session_state.get(APP_KEY)


def set_selected_module_id(module_id: str | None) -> None:
    st.session_state[APP_KEY] = module_id


def get_step() -> int:
    return int(st.session_state.get(STEP_KEY, 1))


def set_step(step: int) -> None:
    st.session_state[STEP_KEY] = int(step)


def go_home() -> None:
    set_selected_module_id(None)
    set_step(1)


def reset_all() -> None:
    clear_all_states()
    init_shell_state()
