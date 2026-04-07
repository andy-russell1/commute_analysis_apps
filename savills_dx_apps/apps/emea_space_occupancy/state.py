from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from . import engine, io, validation
from .config import DEFAULT_WORKBOOK_PATH, NAV_ITEMS, PREFERRED_SCENARIO_AUTO_KEY


EMBEDDED_MODE_KEY = "eso_embedded_mode"
ROUTE_KEY = "eso_route"
WORKBOOK_BYTES_KEY = "eso_workbook_bytes"
WORKBOOK_NAME_KEY = "eso_workbook_name"
WORKBOOK_HASH_KEY = "eso_workbook_hash"
WORKBOOK_STATE_KEY = "eso_workbook_state"
FILTERS_KEY = "eso_filters"
ACTIVE_SCENARIO_KEY = "eso_active_scenario_name"
WORKING_SCENARIO_NAME_KEY = "eso_working_scenario_name"
WORKING_NOTES_KEY = "eso_working_notes"
WORKING_ASSUMPTIONS_KEY = "eso_working_assumptions"
SEED_SNAPSHOTS_KEY = "eso_seed_snapshots"
SAVED_SNAPSHOTS_KEY = "eso_saved_snapshots"
DRAFTS_KEY = "eso_saved_drafts"
PREFERRED_SCENARIO_KEY = "eso_preferred_scenario_key"
LAST_RUN_KEY = "eso_last_run_timestamp"


def _hash_bytes(file_bytes: bytes) -> str:
    digest = hashlib.md5()
    digest.update(file_bytes)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def prepare_workbook_cached(file_bytes: bytes, workbook_name: str) -> dict[str, Any]:
    parsed = io.load_workbook_from_bytes(file_bytes, workbook_name)
    validation_result = validation.validate_workbook(parsed)
    return {
        "parsed": parsed,
        "validation": validation_result,
    }


def _default_filter_state(clean_sheets: dict[str, Any]) -> dict[str, Any]:
    options = engine.build_filter_options(clean_sheets)
    months = options.get("months", [])
    month_range = None
    if months:
        month_range = (months[0], months[-1])
    return {
        "region": [],
        "country": [],
        "city": [],
        "site_name": [],
        "site_type": [],
        "building_name": [],
        "business_unit": [],
        "month_range": month_range,
    }


def init_emea_state(session_state: Any) -> None:
    session_state.setdefault(EMBEDDED_MODE_KEY, False)
    session_state.setdefault(ROUTE_KEY, NAV_ITEMS[0][1])
    session_state.setdefault(SAVED_SNAPSHOTS_KEY, [])
    session_state.setdefault(SEED_SNAPSHOTS_KEY, [])
    session_state.setdefault(DRAFTS_KEY, [])
    session_state.setdefault(PREFERRED_SCENARIO_KEY, PREFERRED_SCENARIO_AUTO_KEY)
    session_state.setdefault(LAST_RUN_KEY, None)


def set_embedded_mode(session_state: Any, enabled: bool) -> None:
    session_state[EMBEDDED_MODE_KEY] = bool(enabled)


def get_route(session_state: Any) -> str:
    return str(session_state.get(ROUTE_KEY, NAV_ITEMS[0][1]))


def set_route(session_state: Any, route: str) -> None:
    session_state[ROUTE_KEY] = str(route)


def _load_default_workbook() -> tuple[bytes | None, str | None]:
    if not DEFAULT_WORKBOOK_PATH.exists():
        return None, None
    return DEFAULT_WORKBOOK_PATH.read_bytes(), DEFAULT_WORKBOOK_PATH.name


def refresh_workbook_state(session_state: Any) -> None:
    file_bytes = session_state.get(WORKBOOK_BYTES_KEY)
    workbook_name = session_state.get(WORKBOOK_NAME_KEY, "Workbook")
    if not file_bytes:
        return
    workbook_hash = _hash_bytes(file_bytes)
    prepared = prepare_workbook_cached(file_bytes, workbook_name)
    validation_result = prepared["validation"]
    clean_sheets = validation_result["clean_sheets"]

    session_state[WORKBOOK_HASH_KEY] = workbook_hash
    session_state[WORKBOOK_STATE_KEY] = prepared
    session_state[SEED_SNAPSHOTS_KEY] = engine.build_seed_snapshots(clean_sheets, workbook_name, workbook_hash)
    session_state[SAVED_SNAPSHOTS_KEY] = []
    session_state[DRAFTS_KEY] = []
    session_state[PREFERRED_SCENARIO_KEY] = PREFERRED_SCENARIO_AUTO_KEY
    session_state[FILTERS_KEY] = _default_filter_state(clean_sheets)

    scenario_names = engine.get_scenario_names(clean_sheets)
    active_scenario = scenario_names[0] if scenario_names else "Live Scenario"
    session_state[ACTIVE_SCENARIO_KEY] = active_scenario
    session_state[WORKING_SCENARIO_NAME_KEY] = active_scenario
    session_state[WORKING_NOTES_KEY] = ""
    session_state[WORKING_ASSUMPTIONS_KEY] = engine.build_working_assumptions(clean_sheets, active_scenario)
    session_state[LAST_RUN_KEY] = None


def ensure_workbook_loaded(session_state: Any) -> None:
    if WORKBOOK_BYTES_KEY not in session_state or session_state.get(WORKBOOK_BYTES_KEY) is None:
        file_bytes, workbook_name = _load_default_workbook()
        if file_bytes is None:
            return
        session_state[WORKBOOK_BYTES_KEY] = file_bytes
        session_state[WORKBOOK_NAME_KEY] = workbook_name
    current_hash = _hash_bytes(session_state[WORKBOOK_BYTES_KEY])
    if current_hash != session_state.get(WORKBOOK_HASH_KEY):
        refresh_workbook_state(session_state)


def set_workbook_override(session_state: Any, *, file_bytes: bytes, workbook_name: str) -> None:
    session_state[WORKBOOK_BYTES_KEY] = file_bytes
    session_state[WORKBOOK_NAME_KEY] = workbook_name
    refresh_workbook_state(session_state)


def reset_to_demo_workbook(session_state: Any) -> None:
    file_bytes, workbook_name = _load_default_workbook()
    if file_bytes is None:
        return
    session_state[WORKBOOK_BYTES_KEY] = file_bytes
    session_state[WORKBOOK_NAME_KEY] = workbook_name
    refresh_workbook_state(session_state)


def workbook_context(session_state: Any) -> dict[str, Any]:
    ensure_workbook_loaded(session_state)
    return session_state.get(WORKBOOK_STATE_KEY, {})


def validation_context(session_state: Any) -> dict[str, Any]:
    return workbook_context(session_state).get("validation", {})


def clean_sheets(session_state: Any) -> dict[str, Any]:
    return validation_context(session_state).get("clean_sheets", {})


def filter_state(session_state: Any) -> dict[str, Any]:
    session_state.setdefault(FILTERS_KEY, _default_filter_state(clean_sheets(session_state)))
    return session_state[FILTERS_KEY]


def active_scenario_name(session_state: Any) -> str:
    session_state.setdefault(ACTIVE_SCENARIO_KEY, "Live Scenario")
    return str(session_state[ACTIVE_SCENARIO_KEY])


def set_active_scenario(session_state: Any, scenario_name: str) -> None:
    session_state[ACTIVE_SCENARIO_KEY] = scenario_name
    session_state[WORKING_SCENARIO_NAME_KEY] = scenario_name
    session_state[WORKING_ASSUMPTIONS_KEY] = engine.build_working_assumptions(clean_sheets(session_state), scenario_name)
    session_state[WORKING_NOTES_KEY] = ""
    session_state[LAST_RUN_KEY] = None


def working_assumptions(session_state: Any):
    session_state.setdefault(
        WORKING_ASSUMPTIONS_KEY,
        engine.build_working_assumptions(clean_sheets(session_state), active_scenario_name(session_state)),
    )
    return session_state[WORKING_ASSUMPTIONS_KEY]


def set_working_assumptions(session_state: Any, assumptions_df) -> None:
    session_state[WORKING_ASSUMPTIONS_KEY] = assumptions_df
    session_state[LAST_RUN_KEY] = None


def save_assumption_draft(session_state: Any, draft_name: str) -> None:
    drafts = list(session_state.get(DRAFTS_KEY, []))
    drafts = [draft for draft in drafts if draft.get("draft_name") != draft_name]
    drafts.append(
        {
            "draft_name": draft_name,
            "timestamp": st.session_state.get(LAST_RUN_KEY),
            "scenario_name": session_state.get(WORKING_SCENARIO_NAME_KEY, draft_name),
            "assumptions": working_assumptions(session_state).copy(),
            "notes": session_state.get(WORKING_NOTES_KEY, ""),
        }
    )
    session_state[DRAFTS_KEY] = drafts


def load_assumption_draft(session_state: Any, draft_name: str) -> None:
    drafts = session_state.get(DRAFTS_KEY, [])
    draft = next((item for item in drafts if item.get("draft_name") == draft_name), None)
    if not draft:
        return
    session_state[WORKING_SCENARIO_NAME_KEY] = draft.get("draft_name", draft_name)
    session_state[WORKING_ASSUMPTIONS_KEY] = draft.get("assumptions").copy()
    session_state[WORKING_NOTES_KEY] = draft.get("notes", "")
    session_state[LAST_RUN_KEY] = None


def add_saved_snapshot(session_state: Any, snapshot: dict[str, Any]) -> None:
    snapshots = list(session_state.get(SAVED_SNAPSHOTS_KEY, []))
    snapshots.append(snapshot)
    session_state[SAVED_SNAPSHOTS_KEY] = snapshots


def saved_snapshots(session_state: Any) -> list[dict[str, Any]]:
    return list(session_state.get(SAVED_SNAPSHOTS_KEY, []))


def seed_snapshots(session_state: Any) -> list[dict[str, Any]]:
    return list(session_state.get(SEED_SNAPSHOTS_KEY, []))


def draft_options(session_state: Any) -> list[str]:
    return [draft.get("draft_name") for draft in session_state.get(DRAFTS_KEY, [])]


def set_preferred_scenario(session_state: Any, snapshot_key: str) -> None:
    session_state[PREFERRED_SCENARIO_KEY] = snapshot_key


def preferred_scenario_key(session_state: Any) -> str:
    return str(session_state.get(PREFERRED_SCENARIO_KEY, PREFERRED_SCENARIO_AUTO_KEY))

