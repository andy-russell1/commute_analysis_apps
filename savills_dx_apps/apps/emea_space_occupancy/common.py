from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from shared.ui.page_header import render_page_header

from . import engine, state
from .config import NAV_ITEMS


def safe_set_page_config(page_title: str, layout: str = "wide") -> None:
    try:
        st.set_page_config(page_title=page_title, layout=layout)
    except Exception:
        return


def render_dashboard_chrome() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] { display: none; }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] button[kind] {
            border-radius: 0.85rem;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            padding: 0.15rem 0 0.25rem 0;
        }
        .eso-context-banner {
            margin: 0.2rem 0 1.1rem 0;
            padding: 0.95rem 1.05rem;
            border: 1px solid rgba(38, 42, 67, 0.16);
            border-radius: 1rem;
            background: linear-gradient(135deg, rgba(38,42,67,0.08), rgba(255,223,0,0.08));
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
        }
        .eso-context-banner strong {
            display: block;
            margin-bottom: 0.26rem;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.72;
        }
        .eso-hero,
        .eso-callout,
        .eso-mini-card,
        .eso-empty-state {
            border-radius: 1rem;
            border: 1px solid rgba(38, 42, 67, 0.14);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .eso-hero {
            margin: 0.15rem 0 1rem 0;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            background: linear-gradient(145deg, rgba(38,42,67,0.08), rgba(255,255,255,0.03));
        }
        .eso-hero__eyebrow,
        .eso-section-heading__eyebrow {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            opacity: 0.68;
            margin-bottom: 0.3rem;
        }
        .eso-hero__title {
            font-size: 1.35rem;
            font-weight: 650;
            line-height: 1.15;
            margin-bottom: 0.45rem;
        }
        .eso-hero__body,
        .eso-callout__body,
        .eso-mini-card__body,
        .eso-empty-state__body,
        .eso-section-heading__body {
            font-size: 0.96rem;
            line-height: 1.5;
            opacity: 0.9;
        }
        .eso-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }
        .eso-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(38, 42, 67, 0.14);
            background: rgba(255, 255, 255, 0.5);
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1;
        }
        .eso-badge__label {
            opacity: 0.62;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.7rem;
        }
        .eso-callout {
            margin: 0.15rem 0 0.9rem 0;
            padding: 0.95rem 1rem;
            background: rgba(38, 42, 67, 0.045);
        }
        .eso-callout__title,
        .eso-section-heading__title,
        .eso-mini-card__title,
        .eso-empty-state__title {
            font-size: 1rem;
            font-weight: 650;
            line-height: 1.2;
            margin-bottom: 0.28rem;
        }
        .eso-bullet-list {
            margin: 0.2rem 0 0 1rem;
            padding: 0;
        }
        .eso-bullet-list li {
            margin: 0.22rem 0;
        }
        .eso-mini-card {
            height: 100%;
            padding: 0.95rem 1rem;
            background: rgba(38, 42, 67, 0.04);
        }
        .eso-mini-card__value {
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }
        .eso-empty-state {
            margin: 0.15rem 0 0.8rem 0;
            padding: 1rem 1.05rem;
            background: rgba(38, 42, 67, 0.04);
        }
        .eso-section-heading {
            margin: 0 0 0.7rem 0;
        }
        .eso-tone-accent {
            border-color: rgba(255, 223, 0, 0.34);
            background: linear-gradient(145deg, rgba(255,223,0,0.16), rgba(38,42,67,0.06));
        }
        .eso-tone-success {
            border-color: rgba(53, 208, 127, 0.3);
            background: linear-gradient(145deg, rgba(53,208,127,0.14), rgba(38,42,67,0.04));
        }
        .eso-tone-warning {
            border-color: rgba(255, 187, 0, 0.3);
            background: linear-gradient(145deg, rgba(255,187,0,0.14), rgba(38,42,67,0.04));
        }
        .eso-tone-critical {
            border-color: rgba(242, 106, 115, 0.32);
            background: linear-gradient(145deg, rgba(242,106,115,0.12), rgba(38,42,67,0.04));
        }
        .eso-tone-info,
        .eso-tone-default {
            background: rgba(38, 42, 67, 0.045);
        }
        div[data-testid="stMetric"] {
            background: rgba(38, 42, 67, 0.06);
            border: 1px solid rgba(38, 42, 67, 0.12);
            border-radius: 1rem;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            opacity: 0.72;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 0.9rem;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_context_banner(title: str, sentence: str) -> None:
    st.markdown(
        f"<div class='eso-context-banner'><strong>{title}</strong>{sentence}</div>",
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, body: str) -> None:
    st.markdown(f"<div class='eso-insight-card'><h4>{title}</h4><div>{body}</div></div>", unsafe_allow_html=True)


def _render_nav_link(label: str, route: str, *, key: str) -> None:
    is_active = state.get_route(st.session_state) == route
    if st.button(label, key=key, use_container_width=True, type="primary" if is_active else "secondary"):
        state.set_route(st.session_state, route)
        st.rerun()


def _filter_summary(filters: dict[str, Any]) -> str:
    parts = []
    for key, label in [
        ("region", "Region"),
        ("country", "Country"),
        ("city", "City"),
        ("site_name", "Site"),
        ("site_type", "Site type"),
        ("building_name", "Building"),
        ("business_unit", "Business unit"),
    ]:
        values = filters.get(key) or []
        if values:
            parts.append(f"{label}: {', '.join(map(str, values[:3]))}{' +' if len(values) > 3 else ''}")
    month_range = filters.get("month_range")
    if month_range:
        start, end = month_range
        if start is not None and end is not None:
            parts.append(f"Months: {pd.Timestamp(start).strftime('%b %Y')} to {pd.Timestamp(end).strftime('%b %Y')}")
    return " | ".join(parts) if parts else "All portfolio filters active."


def render_sidebar() -> dict[str, Any]:
    render_dashboard_chrome()
    state.ensure_workbook_loaded(st.session_state)

    current_route = state.get_route(st.session_state)
    current_index = next((index for index, (_, route) in enumerate(NAV_ITEMS, start=1) if route == current_route), 1)
    current_label = next((label for label, route in NAV_ITEMS if route == current_route), NAV_ITEMS[0][0])
    workbook_state = state.workbook_context(st.session_state)
    validation_result = state.validation_context(st.session_state)
    clean_sheets = state.clean_sheets(st.session_state)
    workbook_name = st.session_state.get(state.WORKBOOK_NAME_KEY, "Bundled demo workbook")

    with st.sidebar:
        with st.container(border=True):
            st.caption("Decision Flow")
            st.markdown(f"**Page {current_index} of {len(NAV_ITEMS)}**")
            st.caption(current_label)
            for index, (label, route) in enumerate(NAV_ITEMS, start=1):
                _render_nav_link(f"{index}. {label}", route, key=f"eso_nav_{route}")
            st.caption("Embedded mode: navigation remains inside Savills DX.")

        with st.container(border=True):
            st.caption("Current Workbook")
            st.markdown(f"**{workbook_name}**")
            if validation_result:
                st.caption(
                    f"Validation: {validation_result.get('status', 'Unknown')} | "
                    f"Score {validation_result.get('quality_score', 0)}"
                )
            uploaded = st.file_uploader("Replace workbook", type=["xlsx"], key="eso_sidebar_upload")
            if uploaded is not None:
                if uploaded.name != st.session_state.get(state.WORKBOOK_NAME_KEY) or uploaded.getvalue() != st.session_state.get(
                    state.WORKBOOK_BYTES_KEY
                ):
                    state.set_workbook_override(
                        st.session_state,
                        file_bytes=uploaded.getvalue(),
                        workbook_name=uploaded.name,
                    )
                    st.rerun()
            if st.button("Reset to bundled demo", key="eso_reset_demo", use_container_width=True):
                state.reset_to_demo_workbook(st.session_state)
                st.rerun()

    if not clean_sheets:
        return {
            "ready": False,
            "workbook_state": workbook_state,
            "validation": validation_result,
            "clean_sheets": clean_sheets,
            "current_route": current_route,
            "current_page_label": current_label,
            "current_page_index": current_index,
            "total_pages": len(NAV_ITEMS),
        }

    scenario_names = engine.get_scenario_names(clean_sheets)
    active_scenario = state.active_scenario_name(st.session_state)
    with st.sidebar:
        with st.container(border=True):
            st.caption("Planning Context")
            if scenario_names:
                selected_scenario = st.selectbox(
                    "Active scenario",
                    options=scenario_names,
                    index=scenario_names.index(active_scenario) if active_scenario in scenario_names else 0,
                    key="eso_sidebar_active_scenario",
                )
                if selected_scenario != active_scenario:
                    state.set_active_scenario(st.session_state, selected_scenario)
                    st.rerun()

    filters = state.filter_state(st.session_state)
    filter_options = engine.build_filter_options(clean_sheets)
    with st.sidebar:
        with st.expander("Global filters", expanded=False):
            st.caption("Filters persist across every page in this studio.")
            for filter_key, label in [
                ("region", "Region"),
                ("country", "Country"),
                ("city", "City"),
                ("site_name", "Site"),
                ("site_type", "Site type"),
                ("building_name", "Building"),
                ("business_unit", "Business unit"),
            ]:
                filters[filter_key] = st.multiselect(
                    label,
                    options=filter_options.get(filter_key, []),
                    default=filters.get(filter_key, []),
                    key=f"eso_filter_{filter_key}",
                )

            months = filter_options.get("months", [])
            if months:
                current_range = filters.get("month_range") or (months[0], months[-1])
                filters["month_range"] = st.select_slider(
                    "Reporting period",
                    options=months,
                    value=current_range,
                    format_func=lambda value: pd.Timestamp(value).strftime("%b %Y"),
                    key="eso_filter_month_range",
                )
        st.caption(_filter_summary(filters))

    return {
        "ready": True,
        "workbook_state": workbook_state,
        "validation": validation_result,
        "clean_sheets": clean_sheets,
        "filters": filters,
        "workbook_name": workbook_name,
        "workbook_hash": st.session_state.get(state.WORKBOOK_HASH_KEY, ""),
        "scenario_names": scenario_names,
        "active_scenario": state.active_scenario_name(st.session_state),
        "working_scenario_name": st.session_state.get(state.WORKING_SCENARIO_NAME_KEY, state.active_scenario_name(st.session_state)),
        "working_notes": st.session_state.get(state.WORKING_NOTES_KEY, ""),
        "working_assumptions": state.working_assumptions(st.session_state),
        "seed_snapshots": state.seed_snapshots(st.session_state),
        "saved_snapshots": state.saved_snapshots(st.session_state),
        "draft_names": state.draft_options(st.session_state),
        "preferred_scenario_key": state.preferred_scenario_key(st.session_state),
        "current_route": current_route,
        "current_page_label": current_label,
        "current_page_index": current_index,
        "total_pages": len(NAV_ITEMS),
        "last_run": st.session_state.get(state.LAST_RUN_KEY),
    }


def prepare_page(title: str, caption: str) -> dict[str, Any]:
    safe_set_page_config(title)
    render_page_header(title, caption)
    context = render_sidebar()
    if context.get("ready"):
        render_context_banner(
            "Active Context",
            f"Scenario: {context['active_scenario']} | Workbook: {context['workbook_name']} | {_filter_summary(context['filters'])}",
        )
    return context


def build_live_bundle(context: dict[str, Any]) -> dict[str, Any]:
    return engine.compute_live_scenario(
        context["clean_sheets"],
        context["working_assumptions"],
        context["filters"],
        workbook_name=context["workbook_name"],
        workbook_hash=context["workbook_hash"],
    )


def build_scenario_library(context: dict[str, Any], live_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return engine.build_scenario_library(
        context.get("seed_snapshots", []),
        context.get("saved_snapshots", []),
        live_bundle,
        manual_preferred_key=context.get("preferred_scenario_key"),
        active_filters=context.get("filters", {}),
        live_calculation_timestamp=context.get("last_run"),
        basis_scenario_name=context.get("active_scenario"),
        live_notes=context.get("working_notes", ""),
    )
