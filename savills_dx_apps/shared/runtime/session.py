from __future__ import annotations

import hashlib
from typing import Any, Dict

import streamlit as st


STATE_KEY = "dx_app_state"
STEP_KEY = "dx_wizard_step"
APP_KEY = "dx_selected_app"


def get_state() -> Dict[str, Dict[str, Any]]:
    return st.session_state.setdefault(STATE_KEY, {})


def get_app_state(app_id: str) -> Dict[str, Any]:
    state = get_state()
    app_state = state.setdefault(app_id, {})
    app_state.setdefault("upload_bytes", None)
    app_state.setdefault("upload_name", None)
    app_state.setdefault("upload_ext", None)
    app_state.setdefault("upload_sig", None)
    app_state.setdefault("status", "idle")
    app_state.setdefault("error", None)
    app_state.setdefault("artifacts", None)
    app_state.setdefault("logs", [])
    return app_state


def build_upload_signature(name: str, bytes_data: bytes) -> str:
    h = hashlib.md5()
    h.update(name.encode("utf-8", errors="ignore"))
    h.update(bytes_data)
    return h.hexdigest()


def reset_app_state(app_id: str) -> None:
    state = get_state()
    state[app_id] = {
        "upload_bytes": None,
        "upload_name": None,
        "upload_ext": None,
        "upload_sig": None,
        "status": "idle",
        "error": None,
        "artifacts": None,
        "logs": [],
    }


def clear_all_states() -> None:
    st.session_state.pop(STATE_KEY, None)
    st.session_state.pop(STEP_KEY, None)
    st.session_state.pop(APP_KEY, None)


def append_log(app_state: Dict[str, Any], message: str) -> None:
    logs = app_state.setdefault("logs", [])
    logs.append(message)
